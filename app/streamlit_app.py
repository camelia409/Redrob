"""Minimal Streamlit demo for judges. Runs a simplified BM25 + features pipeline
on a small candidate sample."""
import os
import sys
from pathlib import Path

import orjson
import pandas as pd
import streamlit as st
import yaml

# Resolve project root: if streamlit_app.py lives next to src/ (HF Space),
# use its directory; otherwise assume it is inside an app/ subfolder.
APP_DIR = Path(__file__).resolve().parent
if (APP_DIR / "src").exists():
    ROOT = APP_DIR
else:
    ROOT = APP_DIR.parent

sys.path.insert(0, str(ROOT))

from src.ingestion.loader import load_candidates_sample
from src.ingestion.jd import get_jd_text
from src.ranking.baselines import BM25Ranker
from src.features.extractor import FeatureExtractor
from src.ranking.weighted_reranker import WeightedReranker
from src.validation.honeypots import honeypot_score_gates_only, run_all_checks
from src.reasoning.generator import generate

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")


def load_weights(path: Path) -> dict:
    """Load the weighted-sum feature weights from the YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["weights_v1"]


st.title("🎯 Redrob Candidate Ranker — Live Demo")
st.caption("India Runs Data & AI Challenge 2026 — Team TBD")

# JD summary
with st.expander("📄 Job Description", expanded=False):
    jd_text = get_jd_text()
    st.text_area("JD", jd_text[:2000] + "\n\n[truncated for display]", height=200)

st.markdown("---")

# Input: bundled sample or upload
col1, col2 = st.columns([1, 3])
with col1:
    input_mode = st.radio("Candidate source", ["Bundled 50-sample", "Upload JSONL"])

if input_mode == "Bundled 50-sample":
    candidates = load_candidates_sample(50)
    st.info(f"Loaded {len(candidates)} bundled sample candidates.")
else:
    uploaded = st.file_uploader(
        "Upload candidates.jsonl (≤100 candidates)", type=["jsonl"]
    )
    if uploaded is None:
        st.warning("Upload a JSONL file to proceed.")
        st.stop()
    candidates = []
    for line in uploaded:
        line = line.strip()
        if not line:
            continue
        candidates.append(orjson.loads(line))
    if len(candidates) > 100:
        st.error(f"Max 100 candidates in demo (got {len(candidates)}).")
        st.stop()
    st.success(f"Loaded {len(candidates)} candidates from upload.")

st.markdown("---")

if st.button("🚀 Rank candidates", type="primary"):
    with st.spinner("Ranking..."):
        # BM25
        bm25 = BM25Ranker(jd_text=get_jd_text())
        bm25_out = bm25.rank(candidates)
        bm25_map = dict(bm25_out)

        # Honeypot scores (gate-only)
        hp_scores = {c["candidate_id"]: honeypot_score_gates_only(c) for c in candidates}

        # Demo dense channel is skipped; use BM25 score as a proxy so the
        # feature extractor still receives a dense signal.
        dense_map = bm25_map.copy()

        fe = FeatureExtractor()
        df = fe.extract_batch(candidates, bm25_map, dense_map, hp_scores)
        df["honeypot_score"] = df["candidate_id"].map(hp_scores)

        # Weighted rerank
        weights = load_weights(ROOT / "configs" / "reranker_weights_v1.yaml")
        reranker = WeightedReranker(weights, score_col="reranker_score")
        df["reranker_score"] = reranker.score(df)

        # Honeypot hard gate: candidates with 3+ gate flags sink to the bottom.
        df.loc[df["honeypot_score"] >= 3, "reranker_score"] -= 1e6
        df = df.sort_values(
            by=["reranker_score", "candidate_id"], ascending=[False, True]
        ).reset_index(drop=True)

        top_n = min(10, len(candidates))
        top = df.head(top_n).copy()

    st.success(f"✅ Ranked {len(candidates)} candidates in seconds.")

    # Display top-10
    st.subheader(f"Top {top_n} Candidates")
    candidate_index = {c["candidate_id"]: c for c in candidates}
    for i, row in top.iterrows():
        cid = row["candidate_id"]
        c = candidate_index[cid]
        p = c.get("profile", {})

        skills = c.get("skills", [])
        skill_names = []
        for s in skills[:3]:
            if isinstance(s, dict):
                skill_names.append(s.get("name", "?"))
            else:
                skill_names.append(str(s))
        skills_text = ", ".join(skill_names) if skill_names else "—"

        hp_flags = [k for k, (t, _) in run_all_checks(c).items() if t]

        with st.container():
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f"**#{i + 1} `{cid}` — "
                    f"{p.get('current_title', '?')} at {p.get('current_company', '?')}**"
                )
                st.caption(
                    f"YoE {p.get('years_of_experience', 0):.1f} · "
                    f"{p.get('location', '?')}, {p.get('country', '?')}"
                )
                st.markdown(f"_Skills:_ {skills_text}")
                try:
                    features_row = row.to_dict()
                    reason = generate(
                        c,
                        rank=i + 1,
                        score=float(row["reranker_score"]),
                        features_row=features_row,
                        honeypot_flags=hp_flags,
                    )
                    st.markdown(f"💬 **Reasoning:** {reason}")
                except Exception as e:
                    st.markdown(f"💬 _(reasoning error: {e})_")
            with c2:
                st.metric("Score", f"{row['reranker_score']:.3f}")
                if hp_flags:
                    st.warning(f"HP flags: {', '.join(hp_flags[:2])}")
        st.markdown("---")

    # Feature-contribution breakdown toggle
    show_breakdown = st.toggle("Show feature contribution breakdown", value=False)
    if show_breakdown:
        selected_cid = st.selectbox(
            "Select a candidate", top["candidate_id"].tolist()
        )
        selected_row = top[top["candidate_id"] == selected_cid].iloc[0]
        breakdown = []
        for feat, wt in weights.items():
            val = selected_row.get(feat, 0.0)
            breakdown.append(
                {
                    "feature": feat,
                    "value": float(val),
                    "weight": float(wt),
                    "contribution": float(val) * float(wt),
                }
            )
        breakdown_df = pd.DataFrame(breakdown).sort_values(
            by="contribution", key=lambda s: s.abs(), ascending=False
        )
        st.dataframe(
            breakdown_df.style.format({
                "value": "{:.3f}",
                "weight": "{:.3f}",
                "contribution": "{:.3f}",
            }),
            use_container_width=True,
        )

    # Download
    csv = top.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download top-10 as CSV", csv, "demo_ranking.csv", "text/csv"
    )

st.markdown("---")
st.caption("Full pipeline (100K candidates) available in the GitHub repository.")
