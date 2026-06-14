import os
import time
import pickle
import pandas as pd
from surprise import SVD, KNNWithMeans, NMF, Reader, Dataset
from surprise.model_selection import cross_validate

PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR         = "models"
TRAIN_PATH         = os.path.join(PROCESSED_DATA_DIR, "train.csv")
VAL_PATH           = os.path.join(PROCESSED_DATA_DIR, "val.csv")

RATING_SCALE = (1, 5)


def load_surprise_data(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    reader = Reader(rating_scale=RATING_SCALE)

    train_data = Dataset.load_from_df(
        train_df[["user_id", "item_id", "rating"]], reader
    )
    val_data = Dataset.load_from_df(
        val_df[["user_id", "item_id", "rating"]], reader
    )

    trainset  = train_data.build_full_trainset()
    valset    = val_data.build_full_trainset().build_testset()

    print(f"Trainset: {trainset.n_ratings:,} ratings, "
          f"{trainset.n_users:,} users, {trainset.n_items:,} items")
    return trainset, valset


def train_svd(trainset):
    print("\nTraining SVD (Singular Value Decomposition)...")
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    t0 = time.time()
    model.fit(trainset)
    print(f"SVD training done in {time.time() - t0:.1f}s")
    return model


def train_knn(trainset):
    print("\nTraining KNN (Item-Based Collaborative Filtering)...")
    sim_options = {
        "name": "cosine",
        "user_based": False,
    }
    model = KNNWithMeans(k=40, sim_options=sim_options, verbose=False)
    t0 = time.time()
    model.fit(trainset)
    print(f"KNN training done in {time.time() - t0:.1f}s")
    return model


def train_nmf(trainset):
    print("\nTraining NMF (Non-Negative Matrix Factorization)...")
    model = NMF(n_factors=50, n_epochs=50, random_state=42)
    t0 = time.time()
    model.fit(trainset)
    print(f"NMF training done in {time.time() - t0:.1f}s")
    return model


def save_model(model, name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved {name} -> {path}")


def main():
    print("=" * 55)
    print("  Training Recommendation Models")
    print("=" * 55)

    trainset, valset = load_surprise_data(TRAIN_PATH, VAL_PATH)

    model_svd = train_svd(trainset)
    save_model(model_svd, "svd")

    model_knn = train_knn(trainset)
    save_model(model_knn, "knn")

    model_nmf = train_nmf(trainset)
    save_model(model_nmf, "nmf")

    print("\nAll 3 models saved to models/")
    print("Next step: run python src/evaluate.py")


if __name__ == "__main__":
    main()
