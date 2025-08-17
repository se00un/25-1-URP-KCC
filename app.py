import os, sys, importlib
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")
try:
    import setuptools
    sys.modules['distutils'] = importlib.import_module('setuptools._distutils')
except Exception:
    pass

import json, platform, math, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import streamlit as st
import shap
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.inspection import permutation_importance
from html import escape
import base64

# ========== 전역 스타일 ==========
st.set_page_config(page_title="쪼꼬미디어", page_icon="🍫", layout="wide")
st.markdown("""
<style>
.caption-large { font-size: 1.0rem; line-height: 1.55; color:#222;
  background:#f8f9fb; border:1px solid #e6e8ee; border-radius:8px; padding:12px 14px; margin:8px 0 24px 0; }
.caption-large ul{ margin:0 0 0 1.2rem; padding-left:0.8rem; }
</style>
""", unsafe_allow_html=True)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({"axes.titlesize":14, "axes.labelsize":12, "legend.fontsize":11})

# ========== 상수/세션 ==========
load_dotenv()
AGE = "6-8세"
CLUSTER_COL = "cluster_6-8세"
OUTLIER_COL = "is_outlier_6-8세"

for k, v in [("analysis_ready", False), ("show_detail", False), ("ctx", None), ("user_input", None)]:
    if k not in st.session_state: st.session_state[k] = v

CLUSTER_CSV   = os.getenv("CLUSTER_CSV",   "media_summary_cluster.csv")
CENTROIDS_CSV = os.getenv("CENTROIDS_CSV", "media_summary_centroids.csv")

INPUT_BOUNDS = {
    "총_주중_이용시간": (0, 720),
    "총_주말_이용시간": (0, 1440),
    "평균_주중_빈도":   (0, 5),
    "평균_주말_빈도":   (0, 2),
    "편중_HHI_주중":    (0, 1),
    "편중_HHI_주말":    (0, 1),
    "B1":               (0, 10),
    "C1":               (0, 10),
}

CLUSTER_META = {0: {"type":"저이용 집중형"}, 1: {"type":"Heavy 멀티 유저형"}, 2: {"type":"주말 전용형"}}

# ========== 유틸 ==========
def get_openai_api_key() -> str:
    try:
        if getattr(st, "secrets", None) and st.secrets.get("OPENAI_API_KEY", "").strip():
            return st.secrets["OPENAI_API_KEY"].strip()
    except Exception:
        pass
    for name in ("OPENAI_API_KEY", "API_KEY"):
        v = os.getenv(name, "").strip()
        if v: return v
    return ""

def render_bullets(points: List[str]):
    if not points: return
    bullets = "".join(f"<li>{escape(p)}</li>" for p in points if p)
    st.markdown(f"<div class='caption-large'><ul>{bullets}</ul></div>", unsafe_allow_html=True)

def wrap_xticklabels(ax, width: int = 6, rotation: int = 15):
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    ax.set_xticklabels([textwrap.fill(lbl, width=width) for lbl in labels],
                       rotation=rotation, ha="right")

def add_outside_legend(ax, loc="upper left"):
    ax.legend(loc=loc, bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)

def nice_feat_name(k: str) -> str:
    return {
        "총_주중_이용시간":"평일 총 사용시간",
        "총_주말_이용시간":"주말 총 사용시간",
        "평균_주중_빈도":"평일 평균 접속 빈도",
        "평균_주말_빈도":"주말 평균 접속 빈도",
        "편중_HHI_주중":"평일 편중(HHI)",
        "편중_HHI_주말":"주말 편중(HHI)",
        "B1":"TV 시작시기 점수",
        "C1":"스마트폰 시작시기 점수",
    }.get(k, k)

def scale_0_1(v, lo, hi):
    if hi <= lo: return 0.0
    return (max(lo, min(hi, v)) - lo)/(hi-lo)

def polar_radar(ax, categories, values, label=None, fill_alpha=0.25, lw=2):
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals = values + values[:1]
    ax.plot(angles, vals, linewidth=lw, label=label)
    ax.fill(angles, vals, alpha=fill_alpha)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1.0)

# 설명 보조
def soft_level(p: float) -> str:
    return ("상대적으로 높은 편" if p>=85 else
            "조금 높은 편" if p>=65 else
            "보통" if p>=35 else
            "조금 낮은 편" if p>=15 else "상대적으로 낮은 편")

def direction_phrase(d: str) -> str:
    return "해당 유형 가능성을 높였어요" if d=="↑" else ("해당 유형 가능성을 조금 낮췄어요" if d=="↓" else "영향이 크지 않았어요")

def margin_size_phrase(m: float) -> str:
    return "작은 편(경계에 가까움)" if m<0.3 else ("보통" if m<1.0 else "넉넉한 편")

def _norm_margin(m: float) -> float: return max(0.0, min(1.0, m/1.2))
def _norm_proba(p: float) -> float:  return max(0.0, min(1.0, (p-0.5)/0.5))
def compute_confidence(margin: float, proba_max: float) -> dict:
    score = 0.5*_norm_margin(margin) + 0.5*_norm_proba(proba_max)
    score_pct = int(round(score*100))
    if score < 0.4:  return {"label":"유동적", "hint":"비슷한 다른 유형과 겹쳐 있어 변화 가능성이 있어요.", "score":score_pct}
    if score < 0.75: return {"label":"보통", "hint":"이 유형으로 볼 근거가 충분해요.", "score":score_pct}
    return {"label":"확실", "hint":"아이의 데이터가 이 유형과 잘 맞아떨어집니다.", "score":score_pct}

def feature_unit_map() -> Dict[str, str]:
    return {
        "총_주중_이용시간": "분", "총_주말_이용시간": "분",
        "평균_주중_빈도": "회/주", "평균_주말_빈도": "회/주말",
        "편중_HHI_주중": "HHI", "편중_HHI_주말": "HHI",
    }

def render_two_col(fig_or_render_fn, bullets: List[str], plot_ratio=(7,3)):
    col_plot, col_text = st.columns(plot_ratio, vertical_alignment="top")
    with col_plot:
        if callable(fig_or_render_fn):
            fig_or_render_fn()  # 내부에서 st.pyplot 호출
        else:
            st.pyplot(fig_or_render_fn, use_container_width=True)
    with col_text:
        render_bullets(bullets)

def _img_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

def render_cluster_banner(cluster_idx: int, cluster_name: str):
    img_map = {
        0: "pic/Cluster0.png",
        1: "pic/Cluster1.png",
        2: "pic/Cluster2.png",
    }
    img_path = img_map.get(cluster_idx)
    if not os.path.exists(img_path):
        pass

    b64 = _img_b64(img_path)

    st.markdown("""
    <style>
      .cluster-banner {
        --img: 240px;               
        --title-size: 1.8rem;     
        --subtitle-size: 0.95rem;
        display: flex; align-items: center; gap: 16px;
        background: #eaf7ef; border: 1px solid #cfead7;
        border-radius: 12px; padding: 14px 18px; margin: 6px 0 12px 0;
      }
      .cluster-banner img {
        width: var(--img); height: auto; border-radius: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,.06);
      }
      .cluster-banner .txt .title {
        font-size: var(--title-size); font-weight: 800; color: #19533a; margin: 0;
      }
      .cluster-banner .txt .sub {
        font-size: var(--subtitle-size); color: #3b5f4f; margin-top: 4px;
      }
      /* 큰 화면에서는 더 크게 */
      @media (min-width: 992px) {
        .cluster-banner { --img: 120px; --title-size: 2.1rem; }
      }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="cluster-banner">
          <img src="data:image/png;base64,{b64}" alt="cluster">
          <div class="txt">
            <div class="title">우리 아이의 유형은 <b>[{escape(cluster_name)}]</b> 입니다!</div>
            <div class="sub">아래에서 또래 비교와 맞춤 해석을 확인해 보세요.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# 데이터 로딩
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(CLUSTER_CSV), pd.read_csv(CENTROIDS_CSV)

df_cluster, df_centroid = load_data()

# RF Surrogate
@st.cache_resource(show_spinner=True)
def train_surrogate_multiclass_6_8(
    df_cluster: pd.DataFrame,
    cluster_col: str = CLUSTER_COL,
    outlier_col: str = OUTLIER_COL,
    feature_candidates: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 500,
    max_depth: Optional[int] = None
):
    if feature_candidates is None:
        feature_candidates = ["총_주중_이용시간","총_주말_이용시간","평균_주중_빈도","평균_주말_빈도","편중_HHI_주중","편중_HHI_주말","B1","C1"]
    features = [c for c in feature_candidates if c in df_cluster.columns]

    sub = df_cluster[df_cluster["연령대"] == AGE].copy()
    sub = sub.dropna(subset=features + [cluster_col])
    sub[cluster_col] = sub[cluster_col].astype(int)
    classes_sorted = np.sort(sub[cluster_col].unique()).tolist()

    sample_weight = None
    if outlier_col in sub.columns:
        w = np.ones(len(sub), dtype=float)
        w[sub[outlier_col] == 1] = 0.5
        sample_weight = w

    X = sub[features].copy(); y = sub[cluster_col].copy()
    X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
        X, y, sample_weight, test_size=test_size, random_state=random_state, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                random_state=random_state, class_weight="balanced_subsample", n_jobs=-1)
    rf.fit(X_tr, y_tr, sample_weight=sw_tr)
    y_pred = rf.predict(X_te)
    metrics = {
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_te, y_pred, average="weighted")),
        "report": classification_report(y_te, y_pred, digits=4, labels=classes_sorted)
    }
    cm = confusion_matrix(y_te, y_pred, labels=classes_sorted)

    return rf, {"age_group": AGE, "features_used": features, "classes": classes_sorted,
                "metrics": metrics, "confusion_matrix": cm.tolist()}, (X_tr, X_te, y_tr, y_te)

rf_mc, res_mc, splits_mc = train_surrogate_multiclass_6_8(df_cluster)

# 학습 분포/센트로이드
features = res_mc["features_used"]
sub_train = df_cluster[(df_cluster["연령대"] == AGE)].copy()
if OUTLIER_COL in sub_train.columns:
    sub_train = sub_train[sub_train[OUTLIER_COL] != 1]
sub_train = sub_train.dropna(subset=features + [CLUSTER_COL])

centroids = sub_train.groupby(CLUSTER_COL)[features].mean().sort_index()
mu, sigma = sub_train[features].mean(), sub_train[features].std(ddof=0).replace(0, 1.0)
dist_map = {f: sub_train[f] for f in features}

# SHAP
X_tr, X_te, y_tr, y_te = splits_mc
feat_order = list(getattr(rf_mc, "feature_names_in_", features))
X_tr_fix = X_tr[feat_order].copy()
explainer = shap.Explainer(rf_mc, X_tr_fix, algorithm="tree")

# 컨텍스트 & 카드
def hhi_from_list(values: List[float]) -> float:
    tot = sum(values)
    if tot <= 0: return 0.0
    shares = [v/tot for v in values]
    return float(sum(s*s for s in shares))

def build_ctx_from_user_input(user_input: dict, topk:int=3) -> dict:
    missing = [f for f in feat_order if f not in user_input]
    if missing: raise ValueError(f"필수 입력 누락: {missing}")
    for f, (lo, hi) in INPUT_BOUNDS.items():
        if f in user_input:
            v = float(user_input[f])
            if not (lo <= v <= hi):
                raise ValueError(f"{f} 값 {v}가 범위({lo}~{hi})를 벗어났습니다.")
    x = {f: float(user_input[f]) for f in feat_order}
    x_df = pd.DataFrame([x], columns=feat_order)

    proba = rf_mc.predict_proba(x_df)[0]
    pred_idx = int(np.argmax(proba))
    pred = int(rf_mc.classes_[pred_idx])
    pred_name = CLUSTER_META.get(pred, {}).get("type", f"Class {pred}")

    exp_one = explainer(x_df, check_additivity=False)
    sv = exp_one.values[0, :, :]
    sv_pred = sv[:, pred_idx]
    order = np.argsort(-np.abs(sv_pred))[:topk]

    top_contribs = []
    for i in order:
        f = feat_order[i]; val = x[f]
        direction = "↑" if sv_pred[i] > 0 else "↓" if sv_pred[i] < 0 else "·"
        top_contribs.append({
            "feature": f, "value": float(val),
            "percentile": float((dist_map[f] <= val).mean()*100.0),
            "shap": float(abs(sv_pred[i])), "direction": direction
        })

    x_z = (x_df.iloc[0] - mu) / sigma
    cen_z = ((centroids - mu) / sigma)
    dists = ((cen_z - x_z)**2).sum(axis=1).pow(0.5)
    closest = int(dists.idxmin()); second = int(dists.nsmallest(2).index[-1])
    margin  = float(dists[second] - dists[closest])

    proto_cmp = [{"feature": f, "child": float(x[f]),
                  f"centroid_{closest}": float(centroids.loc[closest, f]),
                  "delta_to_centroid": float(x[f] - centroids.loc[closest, f])}
                 for f in feat_order]

    return {"age_group": AGE, "pred_cluster": pred, "pred_cluster_name": pred_name,
            "proba_by_class": {int(c): float(p) for c, p in zip(rf_mc.classes_, proba)},
            "topk_shap": top_contribs,
            "prototype_compare": {"nearest_cluster": closest, "second_cluster": second,
                                  "margin_distance": margin, "details": proto_cmp}
           }

def compute_cards_from_ctx(ctx: Dict[str, Any], topk:int=3) -> Dict[str, Any]:
    pred_name = ctx["pred_cluster_name"]
    proba     = ctx["proba_by_class"].get(ctx["pred_cluster"], 0.0)
    margin    = ctx["prototype_compare"]["margin_distance"]
    conf = compute_confidence(margin, proba)
    summary = f"예측 유형은 [{pred_name}] (확률 {proba:.0%})입니다. "

    evidence = []
    for t in ctx["topk_shap"][:topk]:
        name = nice_feat_name(t["feature"])
        evidence.append({"title": name,
                         "body": f"{name} {t['value']:.1f} ({soft_level(t['percentile'])}) — {direction_phrase(t['direction'])}."})

    strengths, suggestions = [], []
    featmap = {e["feature"]: e for e in ctx.get("topk_shap", [])}
    def pct_of(feat): return featmap.get(feat, {}).get("percentile", None)

    if (p := pct_of("편중_HHI_주중")) is not None and p <= 35: strengths.append("평일에는 다양한 매체를 고르게 사용하는 점이 좋아요.")
    if (p := pct_of("편중_HHI_주말")) is not None and p <= 35: strengths.append("주말에도 한쪽으로 치우치지 않고 다양한 매체를 이용하고 있어요.")
    if (p := pct_of("평균_주중_빈도")) is not None and 35 <= p <= 64: strengths.append("평일 접속 빈도가 과하지도 부족하지도 않아 균형적이에요.")
    if (p := pct_of("총_주중_이용시간")) is not None and 35 <= p <= 64: strengths.append("평일 총 사용시간이 적정 범위예요.")
    if (p := pct_of("총_주말_이용시간")) is not None and p <= 64: strengths.append("주말에는 과몰입 없이 비교적 짧게 이용하고 있어요.")

    if (p := pct_of("평균_주중_빈도")) is not None and p >= 65: suggestions.append("평일 접속 빈도가 높은 편이에요. 짧은 휴식 타이머나 ‘끝나는 시간’을 정해보면 좋아요.")
    if (p := pct_of("총_주중_이용시간")) is not None and p >= 65: suggestions.append("평일 총 사용시간이 높은 편이라, 과목·활동 전환과 짧은 스트레칭을 사이사이 넣어보세요.")
    if (p := pct_of("총_주말_이용시간")) is not None and p >= 65: suggestions.append("주말에는 20~30분 단위의 즐거운 루틴으로 과몰입을 예방해 보세요.")
    if (p := pct_of("편중_HHI_주중")) is not None and p >= 65: suggestions.append("평일에 한두 매체로 치우치는 경향이 있어요. 매체를 가볍게 섞어 다양성을 늘려보세요.")
    if (p := pct_of("편중_HHI_주말")) is not None and p >= 65: suggestions.append("주말에 한쪽 매체로 몰리지 않도록, 보고 싶은 것을 미리 골라 다양한 매체를 선택해 보세요.")

    if not strengths: strengths.append("꾸준함과 다양성 측면에서 좋은 점이 보입니다.")
    if not suggestions: suggestions.append("지금의 좋은 균형을 유지하면서 작은 루틴을 더해보면 좋아요.")

    margin_note = (f"분류 확실도는 {conf['label']}({conf['score']}%)예요. {conf['hint']} "
                   f"예측된 유형과 그다음으로 가까운 유형 사이의 간격(마진)은 {margin:.2f}로 {margin_size_phrase(margin)}입니다.")
    proba_max = max(ctx["proba_by_class"].values()) if ctx["proba_by_class"] else 0.0
    parent_line = f"{pred_name}(약 {proba_max:.0%})"

    return {
        "parent_line": {"title":"🐻", "body": parent_line},
        "summary": {"title":"요약", "body": summary},
        "confidence": {"title":"분류 확실도","badge": conf["label"],"score": conf["score"],"hint": conf["hint"]},
        "evidence": {"title":"근거(Top-3)", "items": [{"title":e["title"],"body":e["body"]} for e in evidence]},
        "strengths": {"title":"강점", "items": [{"body": s} for s in strengths]},
        "suggestions": {"title":"작게 시작하는 제안", "items": [{"body": s} for s in suggestions]},
        "margin": {"title":"추가 메모", "body": margin_note},
        "meta": {"pred_cluster": ctx["pred_cluster"],"pred_cluster_name": ctx["pred_cluster_name"],
                 "proba": proba_max,"age_group": ctx["age_group"]}
    }

def build_parent_text(cards: Dict[str, Any]) -> str:
    g = cards.get
    lines = []
    parent_line = g("parent_line", {}).get("body")
    if parent_line: lines.append(f"[🐻] {parent_line}")
    summary = g("summary", {}).get("body")
    if summary: lines.append(f"[요약] {summary}")

    def bullets(prefix: str, arr: List[Dict[str,str]]) -> Optional[str]:
        items = [i.get("body","").strip() for i in arr if i.get("body")]
        if not items: return None
        return prefix + "\n" + "\n".join([f"  - {s}" for s in items])

    ev = bullets("[📈 근거]", g("evidence", {}).get("items", []))
    stg = bullets("[💪 강점]",  g("strengths", {}).get("items", []))
    sug = bullets("[💭 제안]",  g("suggestions", {}).get("items", []))
    for blk in [ev, stg, sug]:
        if blk: lines.append(blk)

    margin_body = g("margin", {}).get("body")
    if margin_body: lines.append("[📋 메모] " + margin_body)
    return "\n".join(lines)

# === LLM: 카드 폴리싱 ===
def polish_cards_with_openai(cards: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    SYSTEM_MSG = ("너는 보호자 대상 앱의 문장 다듬기 보조자야. "
                  "입력 JSON의 구조/키/숫자/백분위/확률/점수는 바꾸지 말고, 문장만 자연스럽고 따뜻하게 다듬어줘.")
    prompt = ("다음 JSON의 문장(body/title)을 부모를 위해 친절하고 친근하게 다듬어주세요. "
              "구조와 모든 수치(확률·백분위·score 등)는 그대로 유지하세요.\n"
              f"```json\n{json.dumps(cards, ensure_ascii=False)}\n```")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content": SYSTEM_MSG},
                  {"role":"user","content": prompt}],
        response_format={"type":"json_object"},
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)


def make_centroid_diff(ctx: Dict[str, Any],
                       user_input: Dict[str, float],
                       centroids: pd.DataFrame,
                       feature_keys: List[str]) -> List[Dict[str, float]]:
    cluster = ctx["pred_cluster"]
    rows = []
    for k in feature_keys:
        child = float(user_input[k])
        cen   = float(centroids.loc[cluster, k])
        rows.append({"feature": k, "child": child, "centroid": cen, "delta": child - cen})
    return rows


# === LLM: 그래프 해석(불릿 2~3줄)만 ===
def make_figure_captions(api_key: str,
                         pred_cluster_name: str,
                         proba_max: float,
                         margin: float,
                         pi_top: List[Dict[str, Any]],
                         shap_top_for_child: List[Dict[str, Any]],
                         percentiles_overview: Dict[str, float],
                         centroid_diff: List[Dict[str, float]],
                         peer_compare: List[Dict[str, float]],   # ← 이미 추가했다면 유지
                         units: Dict[str, str],                   # ← 단위 전달
                         metrics_report: str) -> dict:
    client = OpenAI(api_key=api_key)

    SYSTEM = (
      "너는 보호자에게 결과를 '짧은 해석 불릿 3~4줄'로 전달하는 도우미야. "
      "숫자는 과장 없이, 낙인/비난 없이, 간단명료하게. "
      "각 문장은 20~50자 내외로."
    )

    # 섹션별 작성 지침을 명확히 고지
    GUIDE = {
        "BOX": (
            "- percentiles_overview로 '또래 상 몇 분위'를 말합니다.\n"
            "- peer_compare의 diff를 써서 '또래 평균 대비 ±값(분/회수/HHI)'을 반드시 포함합니다.\n"
            "- centroid_diff의 delta로 '예측된 유형 평균과의 차이'도 함께 말합니다.\n"
            "- 즉, '또래보다 ○○만큼 많고/적고, 유형 내에서는 △△ 수준'을 두 줄 안에 담습니다."
        ),
        "RADAR_BAR": (
            "- centroid_diff의 'child vs centroid' 차이를 사용합니다.\n"
            "- 큰 양(+) 차이면 해당 지표가 {pred} 특성과 얼마나 다른지/닮았는지 설명합니다.\n"
            "- 이 차이가 유형 확률({proba})에 준 방향(↑/↓)을 짧게 언급합니다."
        ),
        "PI": (
            "- pi_top 상위 특징이 왜 모델에서 중요한지 '원리'로 설명합니다.\n"
            "- 예: 편중(HHI)은 한 매체 쏠림을 뜻해 사용 패턴을 가르는 핵심 축입니다."
        ),
        "SHAP": (
            "- shap_top_for_child에서 feature/percentile/방향을 활용합니다.\n"
            "- '{{feature}} {{percentile:.0f}}p, 방향 {{direction}}'처럼 구체 수치 포함,\n"
            "- 그 방향이 {pred} 가능성에 '올림/내림'으로 작용했음을 짧게."
        ),
        "ROC_CM": (
            "- metrics_report를 바탕으로 과대/과소 판정 위험을 1문장으로 요약합니다.\n"
            "- '정확도/균형수치/오분류 경향'을 한 문장에 담고, 과도한 기술용어는 피합니다."
        ),
    }

    USER = (
      "아래 입력을 바탕으로 그래프별 해석 불릿(각 3~4줄)을 만들어줘.\n"
      "JSON으로만 응답:\n"
      "{"
      "\"box_points\":[],\"radar_points\":[],\"bar_points\":[],"
      "\"pi_points\":[],\"shap_points\":[],\"roc_points\":[],\"cm_points\":[]"
      "}\n\n"
      f"- 예측 유형: {pred_cluster_name}\n"
      f"- 최대 예측 확률: {proba_max:.0%}\n"
      f"- 최근접-차근접 마진: {margin:.2f}\n"
      f"- percentiles_overview(또래 대비 pctl): {json.dumps(percentiles_overview, ensure_ascii=False)}\n"
      f"- peer_compare(child/peer_mean/diff): {json.dumps(peer_compare, ensure_ascii=False)}\n"
      f"- centroid_diff(child/centroid/delta): {json.dumps(centroid_diff, ensure_ascii=False)}\n"
      f"- units(feature→단위): {json.dumps(units, ensure_ascii=False)}\n"   # ← 단위
      f"- Permutation Importance Top-k: {json.dumps(pi_top, ensure_ascii=False)}\n"
      f"- 우리 아이 SHAP Top-k: {json.dumps(shap_top_for_child, ensure_ascii=False)}\n"
      f"- 모델 성능 리포트(요약 가능):\n{metrics_report}\n\n"
      "섹션별 작성 규칙:\n"
      f"[BOX]\n{GUIDE['BOX']}\n\n"
      f"[RADAR/BAR]\n{GUIDE['RADAR_BAR']}\n\n"
      f"[PI]\n{GUIDE['PI']}\n\n"
      f"[SHAP]\n{GUIDE['SHAP'].format(pred=pred_cluster_name)}\n\n"
      f"[ROC/CM]\n{GUIDE['ROC_CM']}\n"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content": SYSTEM}, {"role":"user","content": USER}],
        response_format={"type":"json_object"},
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)


# ========== UI — 헤더 & 폼 ==========
st.image("pic/logo.png", width=200)
st.title("쪼꼬미디어 🐻📱")
st.write("우리 아이의 미디어 이용 습관을 함께 지켜봐요!")

with st.form("user_input_form"):
    st.subheader("👨‍👩‍👧 자녀 정보 입력")
    이름 = st.text_input("자녀 이름")
    나이 = st.number_input("나이 (3~11세)", min_value=3, max_value=11, step=1, value=8)

    st.subheader("📺 주중/주말 기기 이용 시간 (분)")
    TV_주중 = st.number_input("TV (주중)", min_value=0)
    컴퓨터_주중 = st.number_input("컴퓨터 (주중)", min_value=0)
    스마트폰_주중 = st.number_input("스마트폰 (주중)", min_value=0)
    태블릿_주중 = st.number_input("태블릿 (주중)", min_value=0)
    TV_주말 = st.number_input("TV (주말)", min_value=0)
    컴퓨터_주말 = st.number_input("컴퓨터 (주말)", min_value=0)
    스마트폰_주말 = st.number_input("스마트폰 (주말)", min_value=0)
    태블릿_주말 = st.number_input("태블릿 (주말)", min_value=0)

    st.subheader("📊 주중/주말 이용 빈도 (0~7일 기준)")
    TV빈도_주중 = st.number_input("TV 빈도 (주중)", min_value=0, max_value=5)
    컴퓨터빈도_주중 = st.number_input("컴퓨터 빈도 (주중)", min_value=0, max_value=5)
    스마트폰빈도_주중 = st.number_input("스마트폰 빈도 (주중)", min_value=0, max_value=5)
    태블릿빈도_주중 = st.number_input("태블릿 빈도 (주중)", min_value=0, max_value=5)
    TV빈도_주말 = st.number_input("TV 빈도 (주말)", min_value=0, max_value=2)
    컴퓨터빈도_주말 = st.number_input("컴퓨터 빈도 (주말)", min_value=0, max_value=2)
    스마트폰빈도_주말 = st.number_input("스마트폰 빈도 (주말)", min_value=0, max_value=2)
    태블릿빈도_주말 = st.number_input("태블릿 빈도 (주말)", min_value=0, max_value=2)

    st.subheader("🕒 미디어 이용 시작 시기")
    TV_시작시기 = st.number_input("TV 이용 시작 나이", min_value=0)
    스마트폰_시작시기 = st.number_input("스마트폰 이용 시작 나이", min_value=0)

    submitted = st.form_submit_button("분석하기 🔍")

# ========== 파이프라인 ==========
if submitted:
    if not (6 <= 나이 <= 8):
        st.warning("현재 버전은 6–8세만 분석합니다. 나이를 6–8세로 설정해 주세요.")
        st.stop()

    총_주중_이용시간 = float(TV_주중 + 컴퓨터_주중 + 스마트폰_주중 + 태블릿_주중)
    총_주말_이용시간 = float(TV_주말 + 컴퓨터_주말 + 스마트폰_주말 + 태블릿_주말)
    평균_주중_빈도   = float(np.mean([TV빈도_주중, 컴퓨터빈도_주중, 스마트폰빈도_주중, 태블릿빈도_주중]))
    평균_주말_빈도   = float(np.mean([TV빈도_주말, 컴퓨터빈도_주말, 스마트폰빈도_주말, 태블릿빈도_주말]))
    편중_HHI_주중    = hhi_from_list([TV_주중, 컴퓨터_주중, 스마트폰_주중, 태블릿_주중])
    편중_HHI_주말    = hhi_from_list([TV_주말, 컴퓨터_주말, 스마트폰_주말, 태블릿_주말])

    B1 = float(np.clip(11 - TV_시작시기, 1, 10))
    C1 = float(np.clip(11 - 스마트폰_시작시기, 1, 10))

    st.session_state.user_input = {
        "총_주중_이용시간": 총_주중_이용시간, "총_주말_이용시간": 총_주말_이용시간,
        "평균_주중_빈도": 평균_주중_빈도, "평균_주말_빈도": 평균_주말_빈도,
        "편중_HHI_주중": 편중_HHI_주중, "편중_HHI_주말": 편중_HHI_주말,
        "B1": B1, "C1": C1,
    }
    st.session_state.ctx = build_ctx_from_user_input(st.session_state.user_input, topk=3)
    st.session_state.analysis_ready = True
    st.session_state.show_detail = False

if st.session_state.analysis_ready and st.session_state.ctx is not None:
    ctx = st.session_state.ctx
    render_cluster_banner(ctx['pred_cluster'], ctx['pred_cluster_name'])

    if st.button("자세히 분석하기 🔎", key="btn_detail", use_container_width=True):
        st.session_state.show_detail = True

if st.session_state.show_detail:
    ctx = st.session_state.ctx
    user_input = st.session_state.user_input

    api_key = get_openai_api_key()
    if not api_key:
        st.error("서버 측 OpenAI API 키가 설정되지 않았습니다. "
                 ".env(OPENAI_API_KEY 또는 API_KEY) 또는 .streamlit/secrets.toml 의 OPENAI_API_KEY를 설정해 주세요.")
        st.stop()

    # ========== 카드 LLM 폴리싱 ==========
    cards = compute_cards_from_ctx(ctx, topk=3)
    try:
        cards = polish_cards_with_openai(cards, api_key)
        st.success("LLM 폴리싱 완료")
    except Exception as e:
        st.error(f"LLM 폴리싱 실패: {e}")
        st.stop()

    # ========== PI/SHAP/퍼센타일 집계 ==========
    with st.spinner("Permutation Importance 계산중…"):
        perm_acc = permutation_importance(
            rf_mc, X_te[features], y_te,
            n_repeats=20, random_state=42, n_jobs=-1
        )
        df_pi_acc = (
            pd.DataFrame({
                "feature": features,
                "importance_mean": perm_acc.importances_mean,
                "importance_std":  perm_acc.importances_std
            })
            .sort_values("importance_mean", ascending=False)
        )

    with st.spinner("SHAP 랭킹 계산중…"):
        X_te_fix = X_te[feat_order].copy()
        exp = explainer(X_te_fix, check_additivity=False)
        vals = exp.values
        mean_abs_all = np.abs(vals).mean(axis=(0, 2))
        shap_rank = (
            pd.DataFrame({"feature": feat_order, "mean_abs_shap": mean_abs_all})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    pi_top = (
        df_pi_acc[["feature", "importance_mean"]]
        .head(5).round(4).to_dict(orient="records")
    )
    child_shap_top = ctx["topk_shap"]

    percentiles_overview = {
        k: float((dist_map[k] <= user_input[k]).mean() * 100.0)
        for k in ["총_주중_이용시간","총_주말_이용시간","평균_주중_빈도","평균_주말_빈도","편중_HHI_주중","편중_HHI_주말"]
    }

    radar_labels = ["총_주중_이용시간","총_주말_이용시간","평균_주중_빈도","평균_주말_빈도","편중_HHI_주중","편중_HHI_주말"]

    # 또래 평균과의 비교치(막대/해석용)
    peer_compare = []
    for k in radar_labels:
        peer_mean = float(dist_map[k].mean())
        peer_compare.append({
            "feature": k,
            "child": float(user_input[k]),
            "peer_mean": peer_mean,
            "diff": float(user_input[k] - peer_mean)
        })

    centroid_diff = make_centroid_diff(ctx, user_input, centroids, radar_labels)
    units = feature_unit_map()  # 헬퍼에 정의되어 있다고 가정

    # ========== 그래프 해석 불릿 LLM 생성 (한 번만 호출) ==========
    try:
        cap_points = make_figure_captions(
            api_key=api_key,
            pred_cluster_name=ctx["pred_cluster_name"],
            proba_max=max(ctx["proba_by_class"].values()),
            margin=ctx["prototype_compare"]["margin_distance"],
            pi_top=pi_top,
            shap_top_for_child=child_shap_top,
            percentiles_overview=percentiles_overview,
            centroid_diff=centroid_diff,
            peer_compare=peer_compare,
            units=units,
            metrics_report=res_mc["metrics"]["report"]
        )
    except Exception as e:
        st.error(f"LLM 캡션 생성 실패: {e}")
        st.stop()

    # ========== 요약 텍스트 ==========
    st.subheader("🧾 요약 텍스트")
    st.code(build_parent_text(cards), language="markdown")

    # ========== 또래 비교 (박스플롯) ==========
    st.divider()
    st.subheader("📊 또래 비교 (박스플롯)")

    def _draw_box_grid():
        df_group = df_cluster[df_cluster["연령대"] == AGE].dropna(subset=features + [CLUSTER_COL]).copy()
        child_point = {
            k: user_input[k]
            for k in ["총_주중_이용시간","총_주말_이용시간","평균_주중_빈도","평균_주말_빈도","편중_HHI_주중","편중_HHI_주말"]
        }
        pairs = [
            ('총_주중_이용시간', "주중 총 이용시간 (분)"),
            ('총_주말_이용시간', "주말 총 이용시간 (분)"),
            ('평균_주중_빈도',   "평균 이용 빈도(주중)"),
            ('평균_주말_빈도',   "평균 이용 빈도(주말)"),
            ('편중_HHI_주중',    "매체 편중도(HHI, 주중)"),
            ('편중_HHI_주말',    "매체 편중도(HHI, 주말)")
        ]
        grid = st.columns(3)
        for i, (col, label) in enumerate(pairs):
            with grid[i % 3]:
                fig, ax = plt.subplots(figsize=(4.6, 3.8))
                sns.boxplot(data=df_group, y=col, ax=ax, width=0.4, fliersize=3, linewidth=1.0)
                ax.scatter(x=0, y=child_point[col], s=60, marker='D', zorder=10, label='우리 아이')
                ax.set_title(label); ax.set_xlabel(""); ax.set_ylabel("")
                ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
                add_outside_legend(ax)
                st.pyplot(fig, use_container_width=True)

    # 왼쪽: 6개 박스플롯 / 오른쪽: LLM 해석
    render_two_col(_draw_box_grid, cap_points.get("box_points", []), plot_ratio=(7,3))

    # ========== 우리 아이 vs. 센트로이드 (레이더) ==========
    st.subheader("📈 우리 아이 vs. 예측 클러스터 센트로이드 (레이더/막대)")

    bounds_map = {"총_주중_이용시간": (0,720), "총_주말_이용시간": (0,1440),
                  "평균_주중_빈도": (0,5), "평균_주말_빈도": (0,2),
                  "편중_HHI_주중": (0,1), "편중_HHI_주말": (0,1)}
    user_scaled  = [scale_0_1(user_input[k], *bounds_map[k]) for k in radar_labels]
    proto_scaled = [scale_0_1(centroids.loc[ctx["pred_cluster"], k], *bounds_map[k]) for k in radar_labels]

    fig_r, ax_r = plt.subplots(figsize=(5.8, 5.8), subplot_kw=dict(polar=True))
    polar_radar(ax_r, [nice_feat_name(k) for k in radar_labels], user_scaled,  label="우리 아이")
    polar_radar(ax_r, [nice_feat_name(k) for k in radar_labels], proto_scaled, label=f"{ctx['pred_cluster_name']}", fill_alpha=0.15, lw=1.5)
    add_outside_legend(ax_r)
    render_two_col(fig_r, cap_points.get("radar_points", []), plot_ratio=(7,3))

    # ========== 지표별 값 비교 (막대) ==========
    df_bar = pd.DataFrame({
        "지표": [nice_feat_name(k) for k in radar_labels],
        "우리 아이": [user_input[k] for k in radar_labels],
        "클러스터 평균": [centroids.loc[ctx["pred_cluster"], k] for k in radar_labels],
    })

    def build_bar_fallback(peer_compare, centroid_diff, units, topk: int = 2) -> List[str]:
        # 또래 평균 대비 차이(|diff|)가 큰 순서로 상위 k개
        pc = sorted(peer_compare, key=lambda x: abs(x["diff"]), reverse=True)[:topk]
        cd_map = {d["feature"]: d for d in centroid_diff}
        bullets: List[str] = []
        for row in pc:
            f = row["feature"]; u = units.get(f, "")
            peer = float(row["diff"])
            cen  = float(cd_map.get(f, {}).get("delta", 0.0))
            name = nice_feat_name(f)
            fmt = ".0f" if u in ("분", "회/주", "회/주말") else ".2f"
            bullets.append(f"{name}: 또래 평균보다 {peer:+{fmt}}{u}, 유형 평균보다 {cen:+{fmt}}{u}.")
        if not bullets:
            bullets = ["또래/유형 평균과의 차이가 크지 않습니다. 균형적으로 이용하고 있어요."]
        return bullets

    bar_points = cap_points.get("bar_points", [])
    if not bar_points:
        bar_points = build_bar_fallback(peer_compare, centroid_diff, units)

    fig2, ax2 = plt.subplots(figsize=(6.8, 4.2))
    idx = np.arange(len(df_bar)); width = 0.4
    ax2.bar(idx - width/2, df_bar["우리 아이"], width, label="우리 아이")
    ax2.bar(idx + width/2, df_bar["클러스터 평균"], width, label="클러스터 평균")
    ax2.set_xticks(idx); ax2.set_xticklabels(df_bar["지표"])
    wrap_xticklabels(ax2, width=6, rotation=15)
    ax2.set_title("지표별 값 비교", pad=8)
    ax2.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    add_outside_legend(ax2)
    render_two_col(fig2, bar_points, plot_ratio=(7,3))

    # ========== 내부 검증 ==========
    st.divider()
    show_internal = st.toggle("🧪 모델 내부 확인하기", value=False)
    if show_internal:
        st.subheader("모델 내부 검증(요약)")
        st.text(res_mc["metrics"]["report"])

        classes = res_mc["classes"]; class_names = [str(c) for c in classes]
        y_pred = rf_mc.predict(X_te)

        # Confusion Matrix
        cm_norm = confusion_matrix(y_te, y_pred, labels=classes, normalize="true")
        fig_cmn, ax_cmn = plt.subplots(figsize=(5.4, 4.6))
        ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)\
            .plot(ax=ax_cmn, cmap="Blues", colorbar=True, values_format=".2f")
        ax_cmn.set_title("Confusion Matrix (Normalized)")
        render_two_col(fig_cmn, cap_points.get("cm_points", []))

        # ROC
        proba_te = rf_mc.predict_proba(X_te)
        fig_roc, ax_roc = plt.subplots(figsize=(6.0, 4.6))
        y_true_ovr = np.zeros((len(y_te), len(classes)), dtype=int)
        for i, c in enumerate(classes):
            y_true_ovr[:, i] = (y_te == c).astype(int)
            fpr, tpr, _ = roc_curve(y_true_ovr[:, i], proba_te[:, i])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"Class {c} (AUC={roc_auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves (One-vs-Rest)")
        add_outside_legend(ax_roc)
        render_two_col(fig_roc, cap_points.get("roc_points", []))

        # PI
        with st.expander("🧩 Permutation Importance (Accuracy 기준)", expanded=True):
            fig_pi, ax_pi = plt.subplots(figsize=(5.8, 4.2))
            ax_pi.barh(df_pi_acc["feature"], df_pi_acc["importance_mean"])
            ax_pi.invert_yaxis(); ax_pi.set_title("중요도(특성을 섞었을 때 정확도 하락량)")
            render_two_col(fig_pi, cap_points.get("pi_points", []))

        # SHAP
        with st.expander("🔎 SHAP 중요도(평균 절대값)", expanded=True):
            fig_bar, ax_bar = plt.subplots(figsize=(5.8, 4.2))
            ax_bar.barh(shap_rank["feature"], shap_rank["mean_abs_shap"])
            ax_bar.invert_yaxis(); ax_bar.set_title("Mean |SHAP| (모든 클래스 평균)")
            ax_bar.set_xlabel("mean |SHAP|")
            render_two_col(fig_bar, cap_points.get("shap_points", []))
