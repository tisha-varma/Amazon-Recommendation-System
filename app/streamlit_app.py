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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1E1B4B; }
    
    /* Global Background */
    .stApp { background-color: #F8F6FF; }
    .stSidebar { background-color: #FFFFFF; border-right: 1px solid #E5E7EB; }

    /* Streamlit Button Overrides */
    .stButton>button { 
        background: linear-gradient(135deg, #7C3AED, #06B6D4);
        color: white; border: none; border-radius: 12px;
        padding: 0.75rem 2rem; font-weight: 600; font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(124,58,237,0.3); color: white;}
    
    /* Random button override (we target it by kind secondary) */
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: #FFFFFF;
        color: #7C3AED;
        border: 1px solid #7C3AED;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background: #F3F0FF;
        color: #7C3AED;
        box-shadow: 0 4px 12px rgba(124,58,237,0.1);
    }
    
    /* Hero Section */
    .hero-title {
        font-size: 3.2rem; font-weight: 700;
        color: #7C3AED;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .hero-badges {
        display: flex; gap: 0.8rem; margin-bottom: 1.5rem;
    }
    .hero-badge {
        background: #EDE9FE; color: #7C3AED; font-size: 0.9rem; font-weight: 600;
        padding: 0.4rem 1rem; border-radius: 20px;
    }
    hr.hero-divider { border-top: 1px solid #DDD6FE; margin-bottom: 2rem; }

    /* Input Card */
    .input-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(124,58,237,0.06);
        margin-bottom: 1.5rem;
    }
    .input-label { font-size: 0.85rem; color: #6B7280; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem; }
    
    div[data-testid="stTextInput"] input {
        border: 1px solid #DDD6FE;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        color: #1E1B4B;
    }
    div[data-testid="stTextInput"] input:focus { border-color: #7C3AED; box-shadow: 0 0 0 1px #7C3AED; }

    /* Recommendation Cards */
    .rec-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1.5rem; margin-bottom: 1.2rem;
        box-shadow: 0 2px 12px rgba(124,58,237,0.05);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .rec-card:hover { 
        transform: translateY(-4px); 
        box-shadow: 0 12px 24px rgba(124,58,237,0.12);
        border-color: #DDD6FE;
    }
    
    .rank-badge { 
        display: inline-flex; align-items: center; justify-content: center;
        width: 32px; height: 32px; border-radius: 50%;
        font-size: 0.9rem; font-weight: 700; color: white;
        margin-bottom: 0.8rem;
    }
    .rank-1 { background: linear-gradient(135deg, #F59E0B, #FBBF24); }
    .rank-2 { background: linear-gradient(135deg, #9CA3AF, #D1D5DB); }
    .rank-3 { background: linear-gradient(135deg, #B45309, #D97706); }
    .rank-other { background: #7C3AED; }

    .card-title { font-size: 1.2rem; font-weight: 700; color: #1E1B4B; margin: 0 0 0.5rem 0; }
    .card-rating { color: #F59E0B; font-size: 1.1rem; margin-bottom: 0.8rem; font-weight: 600; }
    
    .card-reason { 
        color: #6B7280; font-size: 0.9rem; line-height: 1.5; font-style: italic;
        background: #F3F0FF; border-left: 3px solid #7C3AED; 
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin-top: 0.5rem;
    }

    /* Rated Cards (Chips) */
    .rated-chip {
        display: inline-flex; align-items: center;
        background: #FFFFFF;
        border: 1px solid #E5E7EB; border-left: 4px solid #7C3AED;
        border-radius: 8px; padding: 0.5rem 0.8rem; margin: 0 0.5rem 0.5rem 0;
        font-size: 0.85rem; color: #1E1B4B; font-weight: 500;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .rated-chip span { color: #F59E0B; font-weight: 700; margin-right: 0.4rem; }
    .rated-section-title { color: #7C3AED; font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem; margin-top: 1.5rem; }

    /* Stat Boxes */
    .stat-box { 
        background: #FFFFFF; 
        border: 1px solid #E5E7EB;
        border-radius: 16px; padding: 1.2rem; text-align: center;
        box-shadow: 0 2px 12px rgba(124,58,237,0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .stat-box::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    }
    .stat-box.stat-games::before { background: #7C3AED; }
    .stat-box.stat-avg::before { background: #06B6D4; }
    .stat-box.stat-low::before { background: #F59E0B; }
    .stat-box.stat-high::before { background: #10B981; }

    .stat-box:hover { transform: translateY(-2px); box-shadow: 0 8px 16px rgba(124,58,237,0.12); }
    .stat-val { font-size: 2rem; font-weight: 700; color: #7C3AED; }
    .stat-lbl { font-size: 0.75rem; color: #6B7280; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }

    /* Sidebar Model Selector custom styling */
    div[role="radiogroup"] > label {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        cursor: pointer;
        transition: all 0.2s ease;
    }
    div[role="radiogroup"] > label:hover {
        border-color: #DDD6FE;
        background: #F8F6FF;
    }

    .model-pill {
        display: inline-block; background: #ECFEFF; border: 1px solid #67E8F9;
        border-radius: 20px; padding: 0.3rem 0.8rem; font-size: 0.75rem;
        color: #0891B2; margin-top: 0.5rem; font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    .warning-box { 
        background: #FFFBEB; border: 1px solid #FCD34D;
        border-radius: 12px; padding: 1.2rem 1.5rem;
        color: #92400E; font-size: 0.95rem; margin-top: 2rem; 
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ── Model descriptions ─────────────────────────────────────────────────────────
MODEL_INFO = {
    "svd": {
        "label": "SVD — Matrix Factorization",
        "oneliner": "Best overall RMSE. Finds hidden taste patterns.",
        "detail": "Decomposes the user-item matrix into 100 hidden 'taste profiles'. Fast, accurate, slight black-box.",
    },
    "knn": {
        "label": "KNN — Item Similarity",
        "oneliner": "Most explainable. Recommends similar games.",
        "detail": "Finds the 40 most similar items using cosine similarity. Explanations are traceable and honest.",
    },
    "nmf": {
        "label": "NMF — Topic Decomposition",
        "oneliner": "Best ranking quality (NDCG). Additive themes.",
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
    )
    if user_data.empty:
        return
    
    st.markdown('<div class="rated-section-title">🎮 Games Already Rated</div>', unsafe_allow_html=True)
    
    top_games = user_data.head(max_show)
    chips_html = ""
    for row in top_games.itertuples():
        name  = get_item_name(row.item_id, meta_df)
        if len(name) > 35: name = name[:32] + "..."
        stars = render_stars(float(row.rating))
        chips_html += f'<div class="rated-chip"><span>{stars}</span>{name}</div>'
    
    st.markdown(chips_html, unsafe_allow_html=True)
    
    if len(user_data) > max_show:
        with st.expander(f"Show all {len(user_data)} games"):
            all_chips = ""
            for row in user_data.iloc[max_show:].itertuples():
                name  = get_item_name(row.item_id, meta_df)
                if len(name) > 35: name = name[:32] + "..."
                stars = render_stars(float(row.rating))
                all_chips += f'<div class="rated-chip"><span>{stars}</span>{name}</div>'
            st.markdown(all_chips, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎮 Settings")

    model_choice = st.radio(
        "Recommendation Algorithm",
        options=["nmf", "svd", "knn"],
        format_func=lambda x: MODEL_INFO[x]["label"],
    )
    
    st.markdown(
        f'<div class="model-pill">💡 {MODEL_INFO[model_choice]["oneliner"]}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("📖 How does each model work?"):
        for key, info in MODEL_INFO.items():
            st.markdown(f"**{info['label']}**  \n{info['detail']}\n")

    top_n = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

    st.markdown("---")
    st.markdown("### 📊 Project Info")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("**Dataset**")
        st.write("Amazon 2023")
        st.caption("**Built by**")
        st.write("Tisha Varma")
    with col2:
        st.caption("**Metrics**")
        st.write("RMSE, NDCG@10")
        st.caption("**Models**")
        st.write("SVD, KNN, NMF")
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.link_button("GitHub", "https://github.com/tisha-varma/Amazon-Recommendation-System", use_container_width=True)
    with c2: st.link_button("Email", "mailto:varmatisha0@gmail.com", use_container_width=True)


# ── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎮 Game Recommender</div>', unsafe_allow_html=True)
st.markdown("""
<div class="hero-badges">
    <div class="hero-badge">14,839 users</div>
    <div class="hero-badge">2,206 games</div>
    <div class="hero-badge">128,687 interactions</div>
</div>
<hr class="hero-divider">
""", unsafe_allow_html=True)

with st.spinner("Loading dataset…"):
    ratings_df, meta_df = load_datasets()

sample_users = get_sample_users(ratings_df)

def pick_random_user():
    st.session_state["user_input_key"] = random.choice(sample_users)

st.markdown('<div class="input-label">Enter a User ID</div>', unsafe_allow_html=True)
col_input, col_sample = st.columns([3, 2])
with col_input:
    user_id_input = st.text_input(
        "User ID",
        placeholder="Paste a user ID...",
        label_visibility="collapsed",
        key="user_input_key"
    )
with col_sample:
    st.button("🎲 Pick a Random User", use_container_width=True, type="secondary", on_click=pick_random_user)

st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

if run_btn and user_id_input.strip():
    user_id = user_id_input.strip()
    stats   = get_user_stats(user_id, ratings_df)

    # ── User Stats ─────────────────────────────────────────────────────────────
    if stats["found"]:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-box stat-games"><div class="stat-val">{stats["n_ratings"]}</div>'
                        f'<div class="stat-lbl">Games Rated</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box stat-avg"><div class="stat-val">{stats["avg_rating"]}</div>'
                        f'<div class="stat-lbl">Avg Rating</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box stat-low"><div class="stat-val">{stats["min_rating"]}★</div>'
                        f'<div class="stat-lbl">Lowest Given</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-box stat-high"><div class="stat-val">{stats["max_rating"]}★</div>'
                        f'<div class="stat-lbl">Highest Given</div></div>', unsafe_allow_html=True)
        
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
        st.markdown(f"### Top {len(recs)} Recommendations")

        for rec in recs:
            stars = render_stars(rec["predicted_rating"])
            rank = rec['rank']
            rank_class = f"rank-{rank}" if rank <= 3 else "rank-other"
            
            st.markdown(f"""
<div class="rec-card">
    <div class="rank-badge {rank_class}">#{rank}</div>
    <div class="card-title">{rec['item_name']}</div>
    <div class="card-rating">{stars} &nbsp; <span style="color:#6B7280; font-size:0.9rem;">{rec['predicted_rating']} predicted</span></div>
    <div class="card-reason">"{rec['explanation']}"</div>
</div>
""", unsafe_allow_html=True)

elif run_btn:
    st.warning("Please enter a User ID first.")

# ── Known Limitations ──────────────────────────────────────────────────────────
with st.expander("⚠️ Known Limitations"):
    st.markdown("""
<div class="warning-box">
    <strong>Note on Dataset & Predictions:</strong><br><br>
    • Predicted ratings cluster near 5.0 — models are optimistic due to the dataset being 64% five-star reviews.<br>
    • Cold-start: new users (not in training data) get popular-item fallback, not personalised picks.<br>
    • Only covers items with ≥20 ratings — very niche games may not appear.
</div>
    """, unsafe_allow_html=True)
