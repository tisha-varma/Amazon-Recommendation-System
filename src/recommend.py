import os
import pickle
import numpy as np
import pandas as pd

from src.explain import get_explanation

PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR         = "models"

FILTERED_PATH     = os.path.join(PROCESSED_DATA_DIR, "ratings_filtered.csv")
META_PATH         = os.path.join("data", "raw", "video_games_meta.csv")


def load_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data():
    ratings_df = pd.read_csv(FILTERED_PATH)
    meta_df    = pd.read_csv(META_PATH)
    return ratings_df, meta_df


def get_item_name(item_id, meta_df):
    row = meta_df[meta_df["item_id"] == item_id]
    if row.empty or pd.isna(row.iloc[0]["title"]):
        return item_id
    return str(row.iloc[0]["title"])


def get_popular_fallback(ratings_df, meta_df, top_n=10, exclude_items=None):
    exclude_items = exclude_items or set()
    popular = (
        ratings_df[~ratings_df["item_id"].isin(exclude_items)]
        .groupby("item_id")["rating"]
        .agg(["mean", "count"])
        .query("count >= 50")
        .sort_values("mean", ascending=False)
        .head(top_n)
        .reset_index()
    )
    results = []
    for rank, row in enumerate(popular.itertuples(), 1):
        results.append({
            "rank":             rank,
            "item_id":          row.item_id,
            "item_name":        get_item_name(row.item_id, meta_df),
            "predicted_rating": round(row.mean, 2),
            "explanation":      "Highly rated by many users in the community.",
            "warning":          "Cold-start: user not found in training data. Showing popular items instead."
        })
    return results


def get_recommendations(user_id, model_name="svd", top_n=10,
                        ratings_df=None, meta_df=None):
    if ratings_df is None or meta_df is None:
        ratings_df, meta_df = load_data()

    model    = load_model(model_name)
    trainset = model.trainset

    if model_name == "knn":
        knn_model = model
    else:
        knn_model = None

    # Cold start check
    try:
        trainset.to_inner_uid(user_id)
    except ValueError:
        print(f"  Warning: user '{user_id}' not found in training data (cold start).")
        already_rated = set(ratings_df[ratings_df["user_id"] == user_id]["item_id"])
        return get_popular_fallback(ratings_df, meta_df, top_n, already_rated)

    already_rated = set(
        ratings_df[ratings_df["user_id"] == user_id]["item_id"]
    )

    # Sparse user warning
    n_rated = len(already_rated)
    warning = None
    if n_rated < 5:
        warning = f"This user has only rated {n_rated} item(s). Recommendations may be less accurate."

    # Get all items the model knows about
    all_items = set(trainset.all_items())
    all_raw_items = {trainset.to_raw_iid(inner) for inner in all_items}

    # Filter out already-rated items
    candidate_items = all_raw_items - already_rated

    # Predict rating for every unseen item
    predictions = []
    for item_id in candidate_items:
        pred = model.predict(uid=user_id, iid=item_id)
        predictions.append((item_id, pred.est))

    # Sort by predicted rating descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]

    results = []
    for rank, (item_id, predicted_rating) in enumerate(top_predictions, 1):
        explanation = get_explanation(
            user_id=user_id,
            recommended_item_id=item_id,
            model_name=model_name,
            ratings_df=ratings_df,
            meta_df=meta_df,
            knn_model=knn_model,
        )
        result = {
            "rank":             rank,
            "item_id":          item_id,
            "item_name":        get_item_name(item_id, meta_df),
            "predicted_rating": round(predicted_rating, 2),
            "explanation":      explanation,
        }
        if warning:
            result["warning"] = warning
        results.append(result)

    return results


def get_user_stats(user_id, ratings_df):
    user_data = ratings_df[ratings_df["user_id"] == user_id]
    if user_data.empty:
        return {"found": False}
    return {
        "found":       True,
        "n_ratings":   len(user_data),
        "avg_rating":  round(user_data["rating"].mean(), 2),
        "min_rating":  int(user_data["rating"].min()),
        "max_rating":  int(user_data["rating"].max()),
    }
