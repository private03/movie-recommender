import pandas as pd
from lightfm.data import Dataset

PROCESSED_DIR = "data/processed"

def load_processed():
    interactions = pd.read_csv(f"{PROCESSED_DIR}/interactions.csv")
    items = pd.read_csv(f"{PROCESSED_DIR}/items.csv")
    return interactions, items

def make_item_feature_tuples(items: pd.DataFrame):
    """
    Returns an iterator of (item_id, [feature1, feature2, ...])
    using simple categorical features like genres and release year.
    """
    def row_to_features(row):
        feats = []

        # genres column may be a string representation of list: "['Drama', 'Crime']"
        # We'll parse it safely.
        g = row.get("genres", "[]")
        if isinstance(g, str):
            try:
                genres_list = eval(g, {"__builtins__": {}})  # safe-ish minimal eval
            except Exception:
                genres_list = []
        else:
            genres_list = g if g is not None else []

        for genre in genres_list:
            feats.append(f"genre:{genre}")

        # release_date -> year bucket
        rd = str(row.get("release_date", "") or "")
        year = rd[:4] if len(rd) >= 4 and rd[:4].isdigit() else None
        if year:
            feats.append(f"year:{year}")

        return feats

    for _, row in items.iterrows():
        item_id = int(row["tmdbId"])
        yield (item_id, row_to_features(row))

def main():
    interactions_df, items_df = load_processed()

    # Ensure correct dtypes
    interactions_df["userId"] = interactions_df["userId"].astype(int)
    interactions_df["tmdbId"] = interactions_df["tmdbId"].astype(int)

    items_df["tmdbId"] = items_df["tmdbId"].astype(int)

    # 1) Create Dataset object
    dataset = Dataset()

    # 2) Fit mappings (users, items, and item features vocabulary)
    users = interactions_df["userId"].unique()
    item_ids = items_df["tmdbId"].unique()

    # Collect feature vocab (so Dataset knows them ahead of time)
    # This is optional (LightFM can infer during build_item_features),
    # but explicit is clearer.
    feature_vocab = set()
    for _, feats in make_item_feature_tuples(items_df):
        feature_vocab.update(feats)

    dataset.fit(
        users=users,
        items=item_ids,
        item_features=list(feature_vocab),
    )

    # 3) Build interaction matrix
    # If you want implicit feedback, use (user, item) only.
    # For explicit ratings, LightFM is usually used as implicit (recommended).
    # We'll use implicit by default: interaction present = 1.
    interactions, weights = dataset.build_interactions(
        ((row.userId, row.tmdbId) for row in interactions_df.itertuples(index=False))
    )

    # 4) Build item feature matrix
    item_features = dataset.build_item_features(make_item_feature_tuples(items_df))

    print("users:", interactions.shape[0])
    print("items:", interactions.shape[1])
    print("interactions nnz:", interactions.nnz)
    print("item_features shape:", item_features.shape)
    print("item_features nnz:", item_features.nnz)
    
    print("interaction matrix shape:", interactions.shape)
    print("weights shape:", weights.shape)
    print("sample item ids:", items_df["tmdbId"].head().tolist())
    print("sample interaction rows:")
    print(interactions_df.head())

if __name__ == "__main__":
    main()