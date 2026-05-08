"""Movie Recommender — Streamlit UI.

Loads four pre-computed artifacts at startup:
  artifacts/movies_with_netflix.parquet — movie metadata + features (notebooks 01 + 04)
  artifacts/embeddings.npy              — (4803, 384) L2-normalized vectors (notebook 03)
  artifacts/model.joblib                — fitted XGBoost rating regressor (notebook 02)
  + sentence-transformers/all-MiniLM-L6-v2 — for encoding the user's query

Every query: encode -> cosine similarity vs all movies -> top 50 -> XGBoost re-rank -> top N.
Per-query latency on CPU: ~50-150 ms once warm.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ART = Path(__file__).parent / "artifacts"

# Feature columns the saved pipeline expects, in the order it was trained on.
# The model includes Netflix Prize features (4 aggregates + 32 SVD factors) when
# available; rows without Netflix coverage get NaN that's handled by the
# pipeline's median imputer.
FEATURE_COLS = [
    # TMDB numeric
    "release_year", "runtime",
    "log_budget", "log_revenue", "log_popularity", "log_vote_count",
    "n_genres", "n_keywords", "cast_size",
    # Netflix Prize aggregates
    "netflix_mean_rating", "netflix_rating_count",
    "netflix_rating_std",  "netflix_log_count",
    # Netflix Prize SVD latent factors
    *[f"netflix_svd_{i:02d}" for i in range(32)],
    # Categorical
    "primary_genre", "original_language",
    "director", "lead_actor",
]

# Friendly labels for SHAP feature display
FRIENDLY_NAMES = {
    "log_vote_count":      "Vote count (log)",
    "log_budget":          "Budget (log)",
    "log_revenue":         "Revenue (log)",
    "log_popularity":      "TMDB popularity (log)",
    "release_year":        "Release year",
    "runtime":             "Runtime",
    "n_genres":            "Number of genres",
    "n_keywords":          "Number of keywords",
    "cast_size":           "Cast size",
    "netflix_mean_rating": "Netflix mean rating",
    "netflix_rating_count": "Netflix rating count",
    "netflix_rating_std":  "Netflix rating dispersion",
    "netflix_log_count":   "Netflix rating count (log)",
    "director":            "Director track record",
    "lead_actor":          "Lead actor track record",
}

EXAMPLE_QUERIES = [
    "dark sci-fi with a twist ending",
    "heartwarming animated movie about friendship",
    "war drama set in vietnam",
    "revenge thriller with a female lead",
    "The Dark Knight",
    "feel-good romantic comedy",
]

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached artifact loaders — run once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the recommender (one-time, ~5 sec)...")
def load_artifacts():
    if (ART / "movies_with_netflix.parquet").exists():
        parquet_path = ART / "movies_with_netflix.parquet"
    elif (ART / "movies_clean.parquet").exists():
        parquet_path = ART / "movies_clean.parquet"
    else:
        st.error("No movie parquet found in artifacts/. Run notebook 01 first.")
        st.stop()

    missing = [p.name for p in [ART / "embeddings.npy", ART / "model.joblib"]
               if not p.exists()]
    if missing:
        st.error(
            "Missing artifacts: " + ", ".join(missing)
            + ". Run notebooks 02 and 03 to produce them."
        )
        st.stop()

    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    embeddings = np.load(ART / "embeddings.npy")
    ranker = joblib.load(ART / "model.joblib")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # If the parquet lacks columns the model expects, add NaN columns.
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    return df, embeddings, ranker, encoder


@st.cache_resource(show_spinner=False)
def make_explainer(_ranker):
    """Tree-based SHAP explainer attached to the booster inside the saved pipeline."""
    return shap.TreeExplainer(_ranker.named_steps["model"])


# ---------------------------------------------------------------------------
# Core recommendation logic
# ---------------------------------------------------------------------------

def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-9)


def recommend(
    query: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    ranker,
    encoder: SentenceTransformer,
    candidate_k: int = 50,
    final_k: int = 8,
    sim_weight: float = 0.6,
    year_range: tuple[int, int] | None = None,
    min_pred_rating: float = 0.0,
) -> pd.DataFrame:
    """Encode query -> filter eligible rows -> top-K by similarity ->
    XGBoost re-rank -> drop rows below min_pred_rating -> return top final_k.
    """
    qv = encoder.encode([query], normalize_embeddings=True)[0]
    sims = embeddings @ qv  # (n_movies,)

    # Year filter — only consider movies whose release_year is in [lo, hi].
    # Movies with NaN release_year are kept (rare, can't filter what we don't know).
    eligible = np.ones(len(df), dtype=bool)
    if year_range is not None:
        lo, hi = year_range
        years = df["release_year"]
        # Allow NaN years through, exclude only those clearly outside the window.
        eligible &= years.isna() | ((years >= lo) & (years <= hi))

    # Pick top candidate_k from the eligible set, ranked by similarity
    eligible_sims = np.where(eligible, sims, -np.inf)
    cand_idx = np.argpartition(-eligible_sims, min(candidate_k, eligible.sum()) - 1)[:candidate_k]
    cand_idx = cand_idx[np.argsort(-eligible_sims[cand_idx])]
    # Trim any -inf (happens if fewer than candidate_k are eligible)
    cand_idx = cand_idx[eligible_sims[cand_idx] > -np.inf]
    if len(cand_idx) == 0:
        return pd.DataFrame()

    cand = df.iloc[cand_idx].copy()
    cand["similarity"] = sims[cand_idx]
    cand["pred_rating"] = ranker.predict(cand[FEATURE_COLS])

    # Predicted-rating filter
    if min_pred_rating > 0.0:
        cand = cand[cand["pred_rating"] >= min_pred_rating]
    if len(cand) == 0:
        return pd.DataFrame()

    sim_n  = _minmax(cand["similarity"])
    rate_n = _minmax(cand["pred_rating"])
    cand["score"] = sim_weight * sim_n + (1.0 - sim_weight) * rate_n

    return cand.nlargest(final_k, "score").reset_index(drop=True)


def explain_predictions(results: pd.DataFrame, ranker, explainer) -> list[list[tuple[str, float]]]:
    """Return per-result list of cleaned (feature_name, shap_value) pairs sorted
    by absolute contribution. Aggregates the 32 Netflix SVD factors into one
    combined "audience taste" entry for readability.
    """
    if len(results) == 0:
        return []
    prep = ranker.named_steps["prep"]
    X = results[FEATURE_COLS]
    X_trans = prep.transform(X)
    feature_names = list(prep.get_feature_names_out())
    shap_values = explainer.shap_values(X_trans)
    out = []
    for row in shap_values:
        pairs = list(zip(feature_names, row.tolist()))
        # Aggregate the 32 SVD factors into one user-friendly bucket
        svd_sum = sum(v for n, v in pairs if "netflix_svd_" in n)
        pairs = [(n, v) for n, v in pairs if "netflix_svd_" not in n]
        if abs(svd_sum) > 0.005:
            pairs.append(("Netflix audience-taste profile (SVD factors)", svd_sum))
        # Clean prefixes & swap in friendly names
        cleaned = []
        for n, v in pairs:
            n = n.replace("num__", "").replace("low_cat__", "").replace("high_cat__", "")
            if n.startswith("primary_genre_"):
                n = "Primary genre: " + n[len("primary_genre_"):]
            elif n.startswith("original_language_"):
                n = "Language: " + n[len("original_language_"):]
            else:
                n = FRIENDLY_NAMES.get(n, n)
            cleaned.append((n, v))
        cleaned.sort(key=lambda kv: abs(kv[1]), reverse=True)
        out.append(cleaned)
    return out


# ---------------------------------------------------------------------------
# Optional: LLM-generated "why this matched" explanations (Google Gemini)
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"


def generate_explanations(query: str, results: pd.DataFrame, api_key: str) -> list[str | None]:
    try:
        from google import genai
    except ImportError:
        st.warning("google-genai not installed. Run `pip install google-genai`.")
        return [None] * len(results)

    listing = "\n".join(
        f"{i + 1}. {r['title']} ({int(r['release_year']) if pd.notna(r['release_year']) else '?'}) "
        f"— {r['primary_genre']} — {(r['overview'] or '')[:240]}"
        for i, r in results.iterrows()
    )
    prompt = (
        f'A user searched for: "{query}"\n\n'
        f"Below are {len(results)} movie recommendations. For EACH movie, write ONE concise "
        f"sentence (max 22 words) explaining why it matches the query. "
        f"Output exactly {len(results)} numbered lines, no extra text.\n\n"
        f"{listing}"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = response.text or ""
    except Exception as e:
        st.warning(f"Gemini call failed: {e}. Showing results without explanations.")
        return [None] * len(results)

    explanations: list[str | None] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split(". ", 1) if ". " in line else line.split(") ", 1)
        explanations.append(parts[1].strip() if len(parts) == 2 else line)
    while len(explanations) < len(results):
        explanations.append(None)
    return explanations[: len(results)]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

df, embeddings, ranker, encoder = load_artifacts()
explainer = make_explainer(ranker)

# Year-range bounds from data (used to set slider limits)
years_present = df["release_year"].dropna().astype(int)
YEAR_MIN = int(years_present.min())
YEAR_MAX = int(years_present.max())

with st.sidebar:
    st.header("Settings")
    final_k = st.slider("Number of recommendations", 3, 15, 8)
    sim_weight = st.slider(
        "Similarity vs. predicted rating",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="Higher = match the query more strictly. Lower = prefer higher-rated movies.",
    )
    st.caption(f"Blend: {int(sim_weight*100)}% similarity / {int((1-sim_weight)*100)}% predicted rating")

    st.markdown("---")
    st.subheader("Filters")
    year_range = st.slider(
        "Release year",
        min_value=YEAR_MIN, max_value=YEAR_MAX,
        value=(YEAR_MIN, YEAR_MAX),
        help="Restrict recommendations to movies released in this window.",
    )
    min_pred_rating = st.slider(
        "Minimum predicted rating",
        min_value=0.0, max_value=10.0, value=0.0, step=0.5,
        help="Hide candidates with predicted TMDB rating below this threshold. "
             "Set to 0 to disable.",
    )

    st.markdown("---")
    use_llm = st.checkbox(
        "Generate AI explanations",
        value=False,
        help="Optional. Uses the Google Gemini API for a one-line 'why this matched' per movie.",
    )
    api_key = ""
    if use_llm:
        api_key = st.text_input(
            "Google Gemini API key", type="password",
            placeholder="AIza…",
            help="Get a free key at aistudio.google.com/apikey. Sent only to Google, never logged.",
        )

    st.markdown("---")
    with st.expander("How it works"):
        st.markdown(
            "1. Your query is embedded with `sentence-transformers/all-MiniLM-L6-v2`.\n"
            "2. The top 50 movies passing your filters are pulled by cosine similarity.\n"
            "3. An XGBoost model predicts each candidate's TMDB rating.\n"
            "4. SHAP values tell you which features drove each prediction.\n"
            "5. Similarity and predicted rating are blended into a final score.\n\n"
            f"Catalog: **{len(df):,}** movies. Embedding dim: **{embeddings.shape[1]}**."
        )

st.title("🎬 Movie Recommender")
st.caption(
    "Built for MIS 3080 Final Project. Search by movie title, genre, theme, or "
    "free-form description — the recommender combines semantic similarity (AI) with "
    "predicted rating (ML) to surface relevant, high-quality matches."
)

query = st.text_input(
    "What are you in the mood for?",
    placeholder="e.g. dark sci-fi with a twist ending, or just a movie title",
    key="query_input_widget",
)

st.write("**Try one of these:**")
example_cols = st.columns(len(EXAMPLE_QUERIES))
for col, ex in zip(example_cols, EXAMPLE_QUERIES):
    if col.button(ex, use_container_width=True):
        # Write directly to the text-input widget's session-state key.
        # Streamlit reads this on the next rerun, BEFORE the widget renders,
        # so the box appears pre-filled with the chip's text.
        st.session_state["query_input_widget"] = ex
        st.rerun()


# ---------------------------------------------------------------------------
# Run search
# ---------------------------------------------------------------------------

if query:
    with st.spinner("Searching…"):
        results = recommend(
            query,
            df=df, embeddings=embeddings, ranker=ranker, encoder=encoder,
            final_k=final_k, sim_weight=sim_weight,
            year_range=year_range,
            min_pred_rating=min_pred_rating,
        )

    if len(results) == 0:
        st.warning(
            "No movies passed the active filters. Try widening the year range or "
            "lowering the minimum predicted rating."
        )
    else:
        # Per-prediction SHAP for every shown result
        shap_per_result = explain_predictions(results, ranker, explainer)
        base_rate = float(explainer.expected_value)  # global mean predicted rating

        explanations: list[str | None] = [None] * len(results)
        if use_llm and api_key.strip():
            with st.spinner("Writing explanations…"):
                explanations = generate_explanations(query, results, api_key.strip())

        # Active-filter summary so the user knows what's filtering
        active_filters = []
        if year_range != (YEAR_MIN, YEAR_MAX):
            active_filters.append(f"years {year_range[0]}–{year_range[1]}")
        if min_pred_rating > 0.0:
            active_filters.append(f"predicted ≥ {min_pred_rating}")
        filter_note = (" (filters: " + ", ".join(active_filters) + ")") if active_filters else ""

        st.subheader(f"Top {len(results)} matches for: *{query}*{filter_note}")

        for i, row in results.iterrows():
            with st.container(border=True):
                left, right = st.columns([3, 1])
                with left:
                    year = int(row["release_year"]) if pd.notna(row["release_year"]) else "?"
                    st.markdown(f"### {i + 1}. {row['title']} ({year})")
                    meta_parts = [f"**{row['primary_genre']}**"]
                    if pd.notna(row.get("director")):
                        meta_parts.append(f"directed by {row['director']}")
                    if isinstance(row.get("top_cast"), (list, np.ndarray)) and len(row["top_cast"]):
                        meta_parts.append(f"starring {', '.join(row['top_cast'][:3])}")
                    st.markdown(" · ".join(meta_parts))
                    if pd.notna(row.get("overview")):
                        st.write(row["overview"])
                    if explanations[i]:
                        st.info(f"**Why this matched:** {explanations[i]}")

                    # Inline SHAP summary: top 3 contributing features
                    top3 = shap_per_result[i][:3]
                    summary_parts = []
                    for name, val in top3:
                        sign = "+" if val >= 0 else "−"
                        summary_parts.append(f"{sign}{abs(val):.2f} {name}")
                    st.caption("**Why this rating?** " + " · ".join(summary_parts))

                    # Full SHAP detail in an expander
                    with st.expander("Show full feature contributions"):
                        shap_df = pd.DataFrame(
                            shap_per_result[i][:12],
                            columns=["feature", "shap_value"],
                        )
                        st.caption(
                            f"Predicted rating = base ({base_rate:.2f}) + sum of feature contributions. "
                            "Positive values pushed the prediction up; negative pushed it down."
                        )
                        st.dataframe(
                            shap_df.assign(shap_value=shap_df["shap_value"].round(3)),
                            use_container_width=True,
                            hide_index=True,
                        )
                with right:
                    st.metric("TMDB rating", f"{row['vote_average']:.1f}" if pd.notna(row["vote_average"]) else "–")
                    st.metric("Predicted", f"{row['pred_rating']:.1f}")
                    st.metric("Similarity", f"{row['similarity']:.2f}")
                    st.metric("Score", f"{row['score']:.2f}")

        with st.expander("Raw results table"):
            st.dataframe(
                results[["title", "release_year", "primary_genre", "director",
                         "vote_average", "pred_rating", "similarity", "score"]]
                .round(3),
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("Enter a query above or click an example to see recommendations.")
