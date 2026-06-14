import os
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommend import get_recommendations, get_user_stats, load_data

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
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d4e;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        border-color: #667eea;
    }
    .card-rank {
        font-size: 0.75rem;
        color: #667eea;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e8e8ff;
        margin: 0.3rem 0;
    }
    .card-rating {
        color: #ffd700;
        font-size: 1rem;
        margin-bottom: 0.6rem;
    }
    .card-reason {
        color: #aaa;
        font-size: 0.88rem;
        line-height: 1.5;
        border-left: 3px solid #667eea;
        padding-left: 0.8rem;
        margin-top: 0.5rem;
    }
    .stat-box {
        background: #1a1a2e;
        border: 1px solid #2d2d4e;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-lbl {
        font-size: 0.8rem;
        color: #888;
    }
    .warning-box {
        background: #2a1a0a;
        border: 1px solid #8B6914;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #f0c040;
        font-size: 0.88rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_datasets():
    return load_data()


@st.cache_data(show_spinner=False)
def get_sample_users(_ratings_df, n=10):
    counts = _ratings_df["user_id"].value_counts()
    return counts[counts >= 20].head(n).index.tolist()


def render_stars(rating):
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎮 Settings")

    model_choice = st.radio(
        "Recommendation Algorithm",
        options=["svd", "knn", "nmf"],
        format_func=lambda x: {
            "svd": "SVD — Matrix Factorization",
            "knn": "KNN — Item Similarity",
            "nmf": "NMF — Topic Decomposition",
        }[x],
    )

    top_n = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

    with st.expander("ℹ️ How does each model work?"):
        st.markdown("""
**SVD** decomposes the user-item rating matrix into hidden
"taste profiles". It discovers patterns like *"RPG lover"* or
*"budget gamer"* without you telling it those categories exist.

**KNN** finds items similar to ones you already loved.
It says: *"You liked Dark Souls — here are 40 games that Dark
Souls fans also highly rated."*

**NMF** is like SVD but all factors must be positive, making
them loosely interpretable as topics like *"action"* or
*"indie puzzle"*.
        """)

    with st.expander("📊 Project Info"):
        st.markdown("""
- **Dataset**: Amazon Video Games Reviews 2023
- **Models**: SVD · KNN · NMF (Surprise library)
- **Metrics**: RMSE · Precision@10 · NDCG@10
- **Built by**: Tisha Varma
- **Email**: varmatisha0@gmail.com
- **[GitHub Repo](https://github.com/tisha-varma/Amazon-Recommendation-System)**
        """)

# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎮 Game Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Amazon Video Games &nbsp;·&nbsp; Powered by Collaborative Filtering &nbsp;·&nbsp; Built by <a href="mailto:varmatisha0@gmail.com" style="color:#667eea;text-decoration:none;">Tisha Varma</a></div>',
            unsafe_allow_html=True)

with st.spinner("Loading dataset…"):
    ratings_df, meta_df = load_datasets()

sample_users = get_sample_users(ratings_df)

col_input, col_sample = st.columns([3, 2])
with col_input:
    user_id_input = st.text_input(
        "Enter a User ID",
        placeholder="e.g. AHZPLBH7YWFXBH4ZXLKLBMTJQ4A",
        label_visibility="collapsed",
    )
with col_sample:
    if st.button("🎲 Pick a Random User", use_container_width=True):
        import random
        st.session_state["selected_user"] = random.choice(sample_users)

if "selected_user" in st.session_state and not user_id_input:
    user_id_input = st.session_state["selected_user"]
    st.info(f"Using sample user: `{user_id_input}`")

run_btn = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

if run_btn and user_id_input.strip():
    user_id = user_id_input.strip()

    # ── User Stats ────────────────────────────────────────────────────────────
    stats = get_user_stats(user_id, ratings_df)

    if stats["found"]:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{stats["n_ratings"]}</div>'
                        f'<div class="stat-lbl">Games Rated</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{stats["avg_rating"]}</div>'
                        f'<div class="stat-lbl">Avg Rating Given</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{stats["min_rating"]}★</div>'
                        f'<div class="stat-lbl">Lowest Rating</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{stats["max_rating"]}★</div>'
                        f'<div class="stat-lbl">Highest Rating</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Get Recommendations ───────────────────────────────────────────────────
    with st.spinner(f"Generating recommendations with {model_choice.upper()}…"):
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
        # Show warning if present
        if "warning" in recs[0]:
            st.markdown(f'<div class="warning-box">⚠️ {recs[0]["warning"]}</div>',
                        unsafe_allow_html=True)

        st.markdown(f"### Top {len(recs)} Recommendations  ·  `{model_choice.upper()}`")

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
