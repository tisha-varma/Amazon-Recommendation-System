"""
One-time script to fetch metadata for exactly the items in our filtered dataset.
Streams the full metadata file but stops the moment we've found all needed items.
Run this once: python src/fetch_metadata.py
"""
import os
import itertools
import pandas as pd
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

FILTERED_PATH = os.path.join("data", "processed", "ratings_filtered.csv")
META_PATH     = os.path.join("data", "raw", "video_games_meta.csv")


def main():
    # Step 1: Find which item IDs we actually need
    ratings_df   = pd.read_csv(FILTERED_PATH)
    needed_ids   = set(ratings_df["item_id"].unique())
    print(f"Items in filtered dataset: {len(needed_ids)}")

    # Step 2: Check what we already have
    existing_meta = pd.read_csv(META_PATH)
    already_have  = set(existing_meta["item_id"].dropna().unique())
    missing_ids   = needed_ids - already_have
    print(f"Already in metadata CSV:   {len(needed_ids - missing_ids)}")
    print(f"Still missing:             {len(missing_ids)}")

    if not missing_ids:
        print("All items already have metadata! Nothing to do.")
        return

    # Step 3: Stream full metadata to find missing items
    print(f"\nStreaming metadata to find {len(missing_ids)} missing items...")
    stream = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Video_Games",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    found = []
    checked = 0
    for record in stream:
        checked += 1
        pid = record.get("parent_asin", "")
        if pid in missing_ids:
            found.append({
                "item_id":        pid,
                "title":          record.get("title", ""),
                "average_rating": record.get("average_rating", None),
                "price":          record.get("price", None),
            })
            missing_ids.discard(pid)
            print(f"  Found {len(found)} | Remaining: {len(missing_ids)} | Scanned: {checked:,}", end="\r")

        if not missing_ids:
            break

    print(f"\nFound {len(found)} new items after scanning {checked:,} metadata records.")

    if found:
        new_rows = pd.DataFrame(found)
        updated  = pd.concat([existing_meta, new_rows], ignore_index=True)
        updated  = updated.drop_duplicates(subset="item_id")
        updated.to_csv(META_PATH, index=False)
        print(f"Updated metadata CSV: {len(updated)} total items.")
    
    # Show coverage
    final_meta   = pd.read_csv(META_PATH)
    all_needed   = set(pd.read_csv(FILTERED_PATH)["item_id"].unique())
    covered      = all_needed & set(final_meta["item_id"])
    named        = final_meta[final_meta["item_id"].isin(covered) & final_meta["title"].notna() & (final_meta["title"].str.strip() != "")]
    print(f"\nFinal coverage: {len(named)}/{len(all_needed)} items have real names ({len(named)/len(all_needed)*100:.0f}%)")


if __name__ == "__main__":
    main()
