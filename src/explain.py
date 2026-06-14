import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR         = "models"

LIKE_THRESHOLD  = 4.0
TOP_N_REASONS   = 3


def load_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_user_liked_items(user_id, ratings_df, threshold=LIKE_THRESHOLD):
    user_data = ratings_df[ratings_df["user_id"] == user_id]
    liked = user_data[user_data["rating"] >= threshold]["item_id"].tolist()
    return liked


def explain_knn_recommendation(user_id, recommended_item_id,
                                ratings_df, meta_df, knn_model,
                                top_n=TOP_N_REASONS):
    trainset = knn_model.trainset

    try:
        rec_inner_id = trainset.to_inner_iid(recommended_item_id)
    except ValueError:
        return "Recommended based on overall popularity trends."

    neighbors_inner = knn_model.get_neighbors(rec_inner_id, k=knn_model.k)
    neighbor_raw_ids = set(
        trainset.to_raw_iid(inner) for inner in neighbors_inner
    )

    liked_items  = set(get_user_liked_items(user_id, ratings_df))
    reasons      = liked_items & neighbor_raw_ids

    if not reasons:
        return "Recommended because users with similar taste highly rated this."

    reason_items = list(reasons)[:top_n]

    def get_name(item_id):
        row = meta_df[meta_df["item_id"] == item_id]
        if row.empty or pd.isna(row.iloc[0]["title"]):
            return item_id
        return str(row.iloc[0]["title"])[:50]

    reason_names = [get_name(iid) for iid in reason_items]

    if len(reason_names) == 1:
        reason_str = reason_names[0]
    elif len(reason_names) == 2:
        reason_str = f"{reason_names[0]} and {reason_names[1]}"
    else:
        reason_str = f"{reason_names[0]}, {reason_names[1]} and {reason_names[2]}"

    rec_name = get_name(recommended_item_id)
    return (f"Users who liked {reason_str} also highly rated "
            f"products like this.")


def explain_svd_recommendation(user_id, recommended_item_id,
                                ratings_df, meta_df, top_n=TOP_N_REASONS):
    item_raters = ratings_df[
        (ratings_df["item_id"] == recommended_item_id) &
        (ratings_df["rating"] >= LIKE_THRESHOLD)
    ]["user_id"].tolist()

    if not item_raters:
        return "Recommended based on your rating patterns and hidden preferences."

    co_liked = (
        ratings_df[
            (ratings_df["user_id"].isin(item_raters)) &
            (ratings_df["rating"] >= LIKE_THRESHOLD) &
            (ratings_df["item_id"] != recommended_item_id)
        ]
        .groupby("item_id")
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    if not co_liked:
        return "Recommended based on your rating patterns and hidden preferences."

    def get_name(item_id):
        row = meta_df[meta_df["item_id"] == item_id]
        if row.empty or pd.isna(row.iloc[0]["title"]):
            return item_id
        return str(row.iloc[0]["title"])[:50]

    names = [get_name(iid) for iid in co_liked]

    if len(names) == 1:
        name_str = names[0]
    elif len(names) == 2:
        name_str = f"{names[0]} and {names[1]}"
    else:
        name_str = f"{names[0]}, {names[1]} and {names[2]}"

    return f"Users who enjoyed {name_str} also rated this highly."


def get_explanation(user_id, recommended_item_id, model_name,
                    ratings_df, meta_df, knn_model=None):
    if model_name == "knn" and knn_model is not None:
        return explain_knn_recommendation(
            user_id, recommended_item_id, ratings_df, meta_df, knn_model
        )
    else:
        return explain_svd_recommendation(
            user_id, recommended_item_id, ratings_df, meta_df
        )
