import os
import time
import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

MIN_USER_RATINGS   = 5    # minimum ratings a user must have
MIN_ITEM_RATINGS   = 50   # raised to reduce item count and keep KNN model under 100MB
TRAIN_RATIO        = 0.80
VAL_RATIO          = 0.10

SAMPLE_SIZE        = 1_000_000  # stream 1M of 2.8M records — good balance of speed and coverage

RAW_DATA_DIR       = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

REVIEWS_RAW_PATH  = os.path.join(RAW_DATA_DIR,       "video_games_reviews.csv")
META_RAW_PATH     = os.path.join(RAW_DATA_DIR,       "video_games_meta.csv")
FILTERED_PATH     = os.path.join(PROCESSED_DATA_DIR, "ratings_filtered.csv")
TRAIN_PATH        = os.path.join(PROCESSED_DATA_DIR, "train.csv")
VAL_PATH          = os.path.join(PROCESSED_DATA_DIR, "val.csv")
TEST_PATH         = os.path.join(PROCESSED_DATA_DIR, "test.csv")
USER_ENCODER_PATH = os.path.join(PROCESSED_DATA_DIR, "user_encoder_classes.npy")
ITEM_ENCODER_PATH = os.path.join(PROCESSED_DATA_DIR, "item_encoder_classes.npy")


def download_dataset():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    if os.path.exists(REVIEWS_RAW_PATH):
        print("Reviews already downloaded. Loading from cache...")
        reviews_df = pd.read_csv(REVIEWS_RAW_PATH)
    else:
        print(f"Streaming first {SAMPLE_SIZE:,} Video Games reviews from Hugging Face...")
        print("(streaming mode — no need to download the full 2.68GB file)")
        t0 = time.time()

        stream = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_review_Video_Games",
            split="full",
            streaming=True,
            trust_remote_code=True,
        )

        records = []
        for record in itertools.islice(stream, SAMPLE_SIZE):
            records.append({
                "user_id":   record["user_id"],
                "item_id":   record["parent_asin"],
                "rating":    record["rating"],
                "timestamp": record["timestamp"],
            })

        reviews_df = pd.DataFrame(records)
        print(f"Done in {time.time() - t0:.1f}s. Shape: {reviews_df.shape}")
        reviews_df.to_csv(REVIEWS_RAW_PATH, index=False)
        print(f"Saved to {REVIEWS_RAW_PATH}")

    if os.path.exists(META_RAW_PATH):
        print("Metadata already downloaded. Loading from cache...")
        meta_df = pd.read_csv(META_RAW_PATH)
    else:
        print("Streaming Video Games metadata (product names)...")
        t0 = time.time()

        meta_stream = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_meta_Video_Games",
            split="full",
            streaming=True,
            trust_remote_code=True,
        )

        meta_records = []
        for record in itertools.islice(meta_stream, 50_000):
            meta_records.append({
                "item_id":        record.get("parent_asin", ""),
                "title":          record.get("title", ""),
                "average_rating": record.get("average_rating", None),
                "price":          record.get("price", None),
            })

        meta_df = pd.DataFrame(meta_records).drop_duplicates(subset="item_id")
        print(f"Done in {time.time() - t0:.1f}s. Shape: {meta_df.shape}")
        meta_df.to_csv(META_RAW_PATH, index=False)
        print(f"Saved to {META_RAW_PATH}")

    return reviews_df, meta_df


def filter_sparse_interactions(df):
    print(f"\nFiltering: keeping users >= {MIN_USER_RATINGS} ratings, items >= {MIN_ITEM_RATINGS} ratings")
    prev_size = len(df) + 1
    iteration = 0

    while prev_size != len(df):
        prev_size = len(df)
        iteration += 1

        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= MIN_USER_RATINGS].index)]

        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= MIN_ITEM_RATINGS].index)]

        print(f"  Iteration {iteration}: {len(df):,} interactions | "
              f"{df['user_id'].nunique():,} users | {df['item_id'].nunique():,} items")

    print(f"Filtering done after {iteration} iteration(s)")
    return df


def encode_ids(df):
    print("\nEncoding user_id and item_id to integers...")
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df = df.copy()
    df["user_id_enc"] = user_encoder.fit_transform(df["user_id"])
    df["item_id_enc"] = item_encoder.fit_transform(df["item_id"])

    print(f"  Users: 0 to {df['user_id_enc'].max()}")
    print(f"  Items: 0 to {df['item_id_enc'].max()}")
    return df, user_encoder, item_encoder


def temporal_split(df):
    print("\nTemporal split (80/10/10 by timestamp)...")
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n         = len(df_sorted)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df   = df_sorted.iloc[train_end:val_end].copy()
    test_df  = df_sorted.iloc[val_end:].copy()

    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    return train_df, val_df, test_df


def print_summary(df, train_df, val_df, test_df):
    n_users        = df["user_id"].nunique()
    n_items        = df["item_id"].nunique()
    n_interactions = len(df)
    sparsity       = 1 - (n_interactions / (n_users * n_items))

    print("\n" + "=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total users:          {n_users:>10,}")
    print(f"  Total items:          {n_items:>10,}")
    print(f"  Total interactions:   {n_interactions:>10,}")
    print(f"  Sparsity:             {sparsity*100:>9.2f}%")
    print(f"  Avg ratings/user:     {n_interactions/n_users:>9.1f}")
    print(f"  Avg ratings/item:     {n_interactions/n_items:>9.1f}")
    print(f"  Train:                {len(train_df):>10,}")
    print(f"  Validation:           {len(val_df):>10,}")
    print(f"  Test:                 {len(test_df):>10,}")
    print("=" * 55)
    print("\n  Copy these numbers into your README!")


def main():
    print("=" * 55)
    print("  Amazon Video Games — Preprocessing Pipeline")
    print(f"  Sample size: {SAMPLE_SIZE:,} reviews")
    print("=" * 55)

    os.makedirs(RAW_DATA_DIR,       exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    reviews_df, meta_df = download_dataset()

    print(f"\nRaw sample shape: {reviews_df.shape}")
    print("Rating distribution:")
    print(reviews_df["rating"].value_counts().sort_index().to_string())

    filtered_df              = filter_sparse_interactions(reviews_df)
    encoded_df, user_enc, item_enc = encode_ids(filtered_df)

    np.save(USER_ENCODER_PATH, user_enc.classes_)
    np.save(ITEM_ENCODER_PATH, item_enc.classes_)
    filtered_df.to_csv(FILTERED_PATH, index=False)

    train_df, val_df, test_df = temporal_split(encoded_df)

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH,     index=False)
    test_df.to_csv(TEST_PATH,   index=False)

    print_summary(filtered_df, train_df, val_df, test_df)
    print("\nPreprocessing complete! Run: python src/train.py")


if __name__ == "__main__":
    main()
