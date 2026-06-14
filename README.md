# 🎮 Amazon Video Games Recommendation System

> Comparing SVD, KNN, and NMF collaborative filtering algorithms with explainable recommendations — built for the Amazon ML School application.

## 🔴 Live Demo
[▶ Open on Streamlit Cloud](https://your-app-name.streamlit.app) ← *update after deploying*

---

## 📊 Key Results

| Model | RMSE | Precision@5 | Precision@10 | Recall@10 | NDCG@10 |
|-------|------|-------------|--------------|-----------|---------|
| SVD   | **1.1199** | 0.3511 | 0.1928 | 0.7465 | 0.8256 |
| KNN   | 1.1838 | 0.3613 | 0.2054 | 0.7632 | 0.8251 |
| **NMF** | 1.1872 | **0.3679** | **0.2091** | **0.7837** | **0.8260** |

> SVD wins on RMSE. NMF wins on Precision, Recall, and NDCG@10. NMF is set as the default model in the app.

---

## 📁 Dataset

- **Source**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — Video Games category
- **Sampled**: 1,000,000 reviews streamed from 2.8M total (no full download needed)
- **After filtering** (min 5 ratings/user, min 50 ratings/item):
  - Users: **2,966**
  - Items: **240**
  - Interactions: **22,841**
  - Sparsity: **96.79%**
  - Avg ratings/user: 7.7
  - Avg ratings/item: 95.2

---

## 💡 How Explainability Works

Most recommendation systems are black boxes — they show you a product without explaining why. This project generates human-readable reasons for every recommendation.

**KNN (true explanation):**
> "Users who liked *Elden Ring* and *Dark Souls III* also highly rated this product."

This works because KNN recommends items similar to ones the user already loved. We trace back exactly *which* of the user's liked items are neighbours of the recommended item.

**SVD/NMF (approximate explanation):**
> "Users who enjoyed *The Witcher 3* and *Cyberpunk 2077* also rated this highly."

SVD uses abstract latent factors that aren't human-readable. We approximate by finding co-purchase patterns among users who liked the recommended item.

---

## 🔍 Key EDA Findings

1. **Rating bias**: The dataset is heavily skewed toward 5-star ratings (~40% of all reviews), making RMSE a misleading sole metric
2. **Power law user distribution**: Top 10% of users contribute ~60% of all ratings
3. **Long tail items**: ~70% of games have fewer than 50 ratings — classic cold-start territory
4. **High sparsity (~99.7%)**: The user-item matrix is almost entirely empty — this is exactly why we need matrix factorization
5. **Temporal growth**: Review volume grew 10× between 2015–2020, with a slight decline in average rating over time

---

## 🧠 Algorithms Explained (Simply)

**Collaborative Filtering** — the core idea: we don't look at what's *in* the game (genre, graphics). We look at *who else* bought it and what they liked.

**SVD (Singular Value Decomposition)** — compresses the giant user-item matrix into hidden "taste profiles." Discovers patterns like "RPG lover" or "budget-conscious buyer" that we never explicitly defined.

**KNN (K-Nearest Neighbours)** — finds games similar to ones you already loved, based on shared ratings patterns. No compression — pure similarity lookup.

**NMF (Non-Negative Matrix Factorization)** — like SVD but all hidden factors must be positive, making them loosely interpretable as topics like "action" or "indie puzzle."

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| Surprise 1.1.5 | SVD, KNN, NMF algorithms |
| Pandas 2.x | Data manipulation |
| Scikit-learn | LabelEncoder, utilities |
| Matplotlib / Seaborn | EDA plots |
| Streamlit | Web application |
| Hugging Face datasets | Dataset download |

---

## 📂 Project Structure

```
recsys/
├── data/
│   ├── raw/               ← original downloaded data (never modified)
│   └── processed/         ← cleaned, encoded, split CSVs
├── notebooks/
│   └── eda.ipynb          ← 5 exploratory analysis plots
├── src/
│   ├── preprocess.py      ← download, filter, encode, split
│   ├── train.py           ← train SVD, KNN, NMF
│   ├── evaluate.py        ← RMSE, Precision@K, Recall@K, NDCG@K
│   ├── explain.py         ← explainability logic
│   └── recommend.py       ← final unified recommendation function
├── app/
│   └── streamlit_app.py   ← web application
├── models/                ← saved trained models (.pkl files)
├── outputs/               ← evaluation charts (model_comparison.png)
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/amazon-recsys.git
cd amazon-recsys/recsys

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data and preprocess  (~20 mins, downloads ~400MB)
python src/preprocess.py

# 4. Train all 3 models  (~10–30 mins)
python src/train.py

# 5. Evaluate and compare models
python src/evaluate.py

# 6. Launch the web app
streamlit run app/streamlit_app.py
```

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub (make sure it's public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → connect your GitHub repo
4. Set **Main file path** to: `recsys/app/streamlit_app.py`
5. Click Deploy → your app gets a public URL in ~2 minutes

> **Note**: Streamlit Cloud has a 1GB RAM limit. The trained model pkl files must be committed to the repo (don't gitignore the `models/` folder).

---

## 📌 Resume Bullet Point

> *"Built a collaborative filtering recommendation system comparing SVD, KNN, and NMF on Amazon Video Games dataset (XX,XXX users, X,XXX items); achieved Precision@10 of X.XX with explainable recommendations showing co-purchase reasoning; deployed live on Streamlit Cloud."*

---

## 📜 Citation

Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). Bridging Language and Items for Retrieval and Recommendation. *arXiv:2403.03952*
