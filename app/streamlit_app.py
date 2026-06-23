import os
import sys
import random
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommend import get_recommendations, get_user_stats, load_data, get_item_name

st.set_page_config(
    page_title="Amazon Video Games Recommender",
    page_icon="🎮",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f0f1a; }

    .hero-title {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub { color: #888; font-size: 1rem; margin-bottom: 1.5rem; }

    .card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d4e; border-radius: 16px;
        padding: 1.4rem 1.6rem; margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card:hover { transform: translateY(-2px); border-color: #667eea; }
    .card-rank { font-size: 0.75rem; color: #667eea; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1px; }
    .card-title { font-size: 1.1rem; font-weight: 700; color: #e8e8ff; margin: 0.3rem 0; }
    .card-rating { color: #ffd700; font-size: 1rem; margin-bottom: 0.4rem; }
    .card-reason { color: #aaa; font-size: 0.88rem; line-height: 1.5;
        border-left: 3px solid #667eea; padding-left: 0.8rem; margin-top: 0.5rem; }

    .rated-card {
        background: #12122a; border: 1px solid #2d2d4e; border-radius: 10px;
        padding: 0.6rem 0.9rem; margin-bottom: 0.4rem; font-size: 0.85rem; color: #ccc;
    }
    .rated-card span { color: #ffd700; font-weight: 600; margin-right: 0.4rem; }

    .stat-box { background: #1a1a2e; border: 1px solid #2d2d4e;
        border-radius: 12px; padding: 1rem; text-align: center; }
    .stat-val { font-size: 1.8rem; font-weight: 700; color: #667eea; }
    .stat-lbl { font-size: 0.8rem; color: #888; }

    .model-pill {
        display: inline-block; background: #1e1e3a; border: 1px solid #667eea;
        border-radius: 20px; padding: 0.2rem 0.7rem; font-size: 0.78rem;
        color: #667eea; margin-top: 0.3rem;
    }
    .warning-box { background: #2a1a0a; border: 1px solid #8B6914;
        border-radius: 10px; padding: 0.8rem 1rem;
        color: #f0c040; font-size: 0.88rem; margin-bottom: 1rem; }
    .section-label { font-size: 0.7rem; color: #555; text-transform: uppercase;
        letter-spacing: 1.5px; margin-bottom: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Model descriptions (visible, not just in expander) ─────────────────────────
MODEL_INFO = {
    "svd": {
        "label": "SVD — Matrix Factorization",
        "oneliner": "Best overall RMSE. Finds hidden taste patterns (e.g. 'RPG lover').",
        "detail": "Decomposes the user-item matrix into 100 hidden 'taste profiles'. Fast, accurate, slight black-box.",
    },
    "knn": {
        "label": "KNN — Item Similarity",
        "oneliner": "Most explainable. Recommends based on similar games you already loved.",
        "detail": "Finds the 40 most similar items using cosine similarity. Explanations are traceable and honest.",
    },
    "nmf": {
        "label": "NMF — Topic Decomposition",
        "oneliner": "Best ranking quality (NDCG). Factors represent additive 'game themes'.",
        "detail": "Like SVD but all factors ≥ 0, acting as genre/topic strengths. Best Precision@10 and Recall@10.",
    },
}


@st.cache_data(show_spinner=False)
def load_datasets():
    return load_data()


@st.cache_data(show_spinner=False)
def get_sample_users(_ratings_df, n=20):
    counts = _ratings_df["user_id"].value_counts()
    return counts[counts >= 15].head(n).index.tolist()


def render_stars(rating):
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


def show_user_rated_games(user_id, ratings_df, meta_df, max_show=5):
    user_data = (
        ratings_df[ratings_df["user_id"] == user_id]
        .sort_values("rating", ascending=False)
        .head(max_show)
    )
    if user_data.empty:
        return
    st.markdown('<div class="section-label">🎮 Games this user has already rated</div>', unsafe_allow_html=True)
    for row in user_data.itertuples():
        name  = get_item_name(row.item_id, meta_df)
        stars = render_stars(float(row.rating))
        st.markdown(
            f'<div class="rated-card"><span>{stars}</span>{name}</div>',
            unsafe_allow_html=True,
        )
    total = len(ratings_df[ratings_df["user_id"] == user_id])
    if total > max_show:
        st.caption(f"…and {total - max_show} more rated games")
    st.markdown("<br>", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎮 Settings")

    model_choice = st.radio(
        "Recommendation Algorithm",
        options=["nmf", "svd", "knn"],
        format_func=lambda x: MODEL_INFO[x]["label"],
    )
    # Visible one-liner under the radio — no need to open expander
    st.markdown(
        f'<div class="model-pill">💡 {MODEL_INFO[model_choice]["oneliner"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📖 How does each model work?"):
        for key, info in MODEL_INFO.items():
            st.markdown(f"**{info['label']}**  \n{info['detail']}\n")

    top_n = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

    st.markdown("---")
    with st.expander("📊 Project Info"):
        st.markdown("""
- **Dataset**: Amazon Video Games Reviews 2023
- **Models**: SVD · KNN · NMF (Surprise library)
- **Metrics**: RMSE · Precision@10 · NDCG@10
- **Built by**: Tisha Varma
- **Email**: varmatisha0@gmail.com
- **[GitHub](https://github.com/tisha-varma/Amazon-Recommendation-System)**
        """)
    with st.expander("⚠️ Known Limitations"):
        st.markdown("""
- Predicted ratings cluster near 5.0 — models are optimistic due to the dataset being 64% five-star reviews.
- Cold-start: new users (not in training data) get popular-item fallback, not personalised picks.
- Only covers items with ≥20 ratings — very niche games may not appear.
        """)

# ── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎮 Game Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Amazon Video Games &nbsp;·&nbsp; Collaborative Filtering '
    '&nbsp;·&nbsp; Built by <a href="mailto:varmatisha0@gmail.com" '
    'style="color:#667eea;text-decoration:none;">Tisha Varma</a></div>',
    unsafe_allow_html=True,
)

with st.spinner("Loading dataset…"):
    ratings_df, meta_df = load_datasets()

sample_users = get_sample_users(ratings_df)

col_input, col_sample = st.columns([3, 2])
with col_input:
    user_id_input = st.text_input(
        "User ID",
        placeholder="Paste a user ID or click 'Pick a Random User' →",
        label_visibility="collapsed",
    )
with col_sample:
    if st.button("🎲 Pick a Random User", use_container_width=True):
        st.session_state["selected_user"] = random.choice(sample_users)

if "selected_user" in st.session_state and not user_id_input:
    user_id_input = st.session_state["selected_user"]
    st.info(f"Using sample user: `{user_id_input}`")

run_btn = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

if run_btn and user_id_input.strip():
    user_id = user_id_input.strip()
    stats   = get_user_stats(user_id, ratings_df)

    # ── User Stats ─────────────────────────────────────────────────────────────
    if stats["found"]:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{stats["n_ratings"]}</div>'
                        f'<div class="stat-lbl">Games Rated</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{stats["avg_rating"]}</div>'
                        f'<div class="stat-lbl">Avg Rating</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{stats["min_rating"]}★</div>'
                        f'<div class="stat-lbl">Lowest Given</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{stats["max_rating"]}★</div>'
                        f'<div class="stat-lbl">Highest Given</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Show games already rated ────────────────────────────────────────────
        show_user_rated_games(user_id, ratings_df, meta_df)

    # ── Get Recommendations ────────────────────────────────────────────────────
    with st.spinner(f"Generating {top_n} recommendations with {model_choice.upper()}…"):
        recs = get_recommendations(
            user_id=user_id,
            model_name=model_choice,
            top_n=top_n,
            ratings_df=ratings_df,
            meta_df=meta_df,
        )

    if not recs:
        st.error("Could not generate recommendations for this user.")
    else:
        if "warning" in recs[0]:
            st.markdown(f'<div class="warning-box">⚠️ {recs[0]["warning"]}</div>',
                        unsafe_allow_html=True)

        st.markdown(f"### Top {len(recs)} Recommendations &nbsp; `{model_choice.upper()}`")

        for rec in recs:
            stars = render_stars(rec["predicted_rating"])
            st.markdown(f"""
<div class="card">
    <div class="card-rank">#{rec['rank']}</div>
    <div class="card-title">{rec['item_name']}</div>
    <div class="card-rating">{stars} &nbsp; {rec['predicted_rating']} predicted</div>
    <div class="card-reason">💡 {rec['explanation']}</div>
</div>
""", unsafe_allow_html=True)

elif run_btn:
    st.warning("Please enter a User ID first.")
