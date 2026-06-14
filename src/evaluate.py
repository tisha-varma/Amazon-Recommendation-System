import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from collections import defaultdict
from surprise import Reader, Dataset

PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR         = "models"
OUTPUTS_DIR        = "outputs"

TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(PROCESSED_DATA_DIR, "test.csv")

RATING_SCALE   = (1, 5)
LIKE_THRESHOLD = 4.0
K_VALUES       = [5, 10]


def load_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_testset(test_path, train_path):
    test_df  = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    reader     = Reader(rating_scale=RATING_SCALE)
    train_data = Dataset.load_from_df(train_df[["user_id", "item_id", "rating"]], reader)
    trainset   = train_data.build_full_trainset()

    testset = [(row.user_id, row.item_id, row.rating)
               for row in test_df.itertuples()]
    return trainset, testset, test_df


def compute_rmse(predictions):
    errors = [(pred.r_ui - pred.est) ** 2 for pred in predictions]
    return np.sqrt(np.mean(errors))


def precision_recall_at_k(predictions, k, threshold=LIKE_THRESHOLD):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel          = sum(1 for (_, true_r) in user_ratings if true_r >= threshold)
        n_rec_k        = sum(1 for (est, _)    in user_ratings[:k] if est >= threshold)
        n_rel_and_rec_k = sum(1 for (est, true_r) in user_ratings[:k]
                              if true_r >= threshold and est >= threshold)

        precisions[uid] = n_rel_and_rec_k / k if k != 0 else 0
        recalls[uid]    = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    avg_precision = np.mean(list(precisions.values()))
    avg_recall    = np.mean(list(recalls.values()))
    return avg_precision, avg_recall


def ndcg_at_k(predictions, k, threshold=LIKE_THRESHOLD):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    ndcg_scores = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        dcg = sum(
            (1 if true_r >= threshold else 0) / np.log2(rank + 2)
            for rank, (_, true_r) in enumerate(top_k)
        )

        ideal = sorted([1 if tr >= threshold else 0 for _, tr in user_ratings],
                       reverse=True)[:k]
        idcg = sum(
            val / np.log2(rank + 2)
            for rank, val in enumerate(ideal)
        )

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(ndcg_scores)


def evaluate_model(model, testset, model_name):
    print(f"\nEvaluating {model_name}...")
    predictions = model.test(testset)

    rmse   = compute_rmse(predictions)
    p5, r5 = precision_recall_at_k(predictions, k=5)
    p10, r10 = precision_recall_at_k(predictions, k=10)
    ndcg10 = ndcg_at_k(predictions, k=10)

    results = {
        "Model":   model_name,
        "RMSE":    round(rmse, 4),
        "P@5":     round(p5, 4),
        "P@10":    round(p10, 4),
        "R@10":    round(r10, 4),
        "NDCG@10": round(ndcg10, 4),
    }
    print(f"  RMSE={rmse:.4f}  P@5={p5:.4f}  P@10={p10:.4f}  "
          f"R@10={r10:.4f}  NDCG@10={ndcg10:.4f}")
    return results


def plot_comparison(results_list):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = pd.DataFrame(results_list).set_index("Model")
    metrics = ["RMSE", "P@5", "P@10", "R@10", "NDCG@10"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(df.index, df[metric], color=colors, edgecolor="white", width=0.5)
        ax.set_title(metric, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylim(0, df[metric].max() * 1.25)
        ax.tick_params(axis="x", labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Comparison — Amazon Video Games Recommendation System",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nSaved comparison chart -> {out_path}")


def print_results_table(results_list):
    df = pd.DataFrame(results_list)
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    best_rmse  = df.loc[df["RMSE"].idxmin(),    "Model"]
    best_ndcg  = df.loc[df["NDCG@10"].idxmax(), "Model"]
    best_prec  = df.loc[df["P@10"].idxmax(),    "Model"]

    print(f"\n  Best RMSE:    {best_rmse}")
    print(f"  Best NDCG@10: {best_ndcg}")
    print(f"  Best P@10:    {best_prec}")
    print("\n  (Copy these numbers into your README!)")


def main():
    print("=" * 55)
    print("  Evaluating All Models")
    print("=" * 55)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    trainset, testset, _ = build_testset(TEST_PATH, TRAIN_PATH)

    results_list = []
    for name in ["svd", "knn", "nmf"]:
        model   = load_model(name)
        results = evaluate_model(model, testset, name.upper())
        results_list.append(results)

    print_results_table(results_list)
    plot_comparison(results_list)
    print("\nEvaluation complete! Next: run python src/recommend.py")


if __name__ == "__main__":
    main()
