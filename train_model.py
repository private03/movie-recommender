import pandas as pd
import os
import joblib
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, recall_at_k

PROCESSED_DIR = "data/processed"


def load_processed():
    interactions = pd.read_csv(f"{PROCESSED_DIR}/interactions.csv")
    items = pd.read_csv(f"{PROCESSED_DIR}/items.csv")
    return interactions, items


def parse_genres(genres_str):
    import ast

    if pd.isna(genres_str):
        return []

    try:
        parsed = ast.literal_eval(genres_str)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


def make_item_feature_tuples(items_df: pd.DataFrame):
    for _, row in items_df.iterrows():
        item_id = int(row["tmdbId"])
        features = []

        genres_list = parse_genres(row.get("genres", "[]"))
        for genre in genres_list:
            features.append(f"genre:{genre}")

        release_date = str(row.get("release_date", "") or "")
        if len(release_date) >= 4 and release_date[:4].isdigit():
            year = release_date[:4]
            features.append(f"year:{year}")

        yield (item_id, features)


def build_dataset_and_matrices(interactions_df: pd.DataFrame, items_df: pd.DataFrame):
    interactions_df["userId"] = interactions_df["userId"].astype(int)
    interactions_df["tmdbId"] = interactions_df["tmdbId"].astype(int)
    items_df["tmdbId"] = items_df["tmdbId"].astype(int)

    users = interactions_df["userId"].unique()
    item_ids = items_df["tmdbId"].unique()

    feature_vocab = set()
    for _, features in make_item_feature_tuples(items_df):
        feature_vocab.update(features)

    dataset = Dataset()
    dataset.fit(
        users=users,
        items=item_ids,
        item_features=list(feature_vocab),
    )

    interactions_matrix, weights = dataset.build_interactions(
        (row.userId, row.tmdbId) for row in interactions_df.itertuples(index=False)
    )

    item_features = dataset.build_item_features(
        make_item_feature_tuples(items_df)
    )

    return dataset, interactions_matrix, weights, item_features


def main():
    interactions_df, items_df = load_processed()

    dataset, interactions_matrix, weights, item_features = build_dataset_and_matrices(
        interactions_df, items_df
    )

    print("Interaction matrix shape:", interactions_matrix.shape)
    print("Interaction nnz:", interactions_matrix.nnz)
    print("Item feature matrix shape:", item_features.shape)
    print("Item feature nnz:", item_features.nnz)

    # Split interaction data into train and test
    train, test = random_train_test_split(
        interactions_matrix,
        test_percentage=0.2,
        random_state=42
    )

    print("Train nnz:", train.nnz)
    print("Test nnz:", test.nnz)

    # Create model
    model = LightFM(
        loss="warp",
        no_components=32,
        learning_rate=0.05,
        random_state=42
    )

    # Train model
    model.fit(
        train,
        item_features=item_features,
        epochs=5,
        num_threads=1
    )

    # Evaluate
    train_precision = precision_at_k(
        model,
        train,
        item_features=item_features,
        k=10
    ).mean()

    test_precision = precision_at_k(
        model,
        test,
        train_interactions=train,
        item_features=item_features,
        k=10
    ).mean()

    train_recall = recall_at_k(
        model,
        train,
        item_features=item_features,
        k=10
    ).mean()

    test_recall = recall_at_k(
        model,
        test,
        train_interactions=train,
        item_features=item_features,
        k=10
    ).mean()

    print(f"Train Precision@10: {train_precision:.4f}")
    print(f"Test Precision@10:  {test_precision:.4f}")
    print(f"Train Recall@10:    {train_recall:.4f}")
    print(f"Test Recall@10:     {test_recall:.4f}")

    os.makedirs("artifacts", exist_ok=True)

    interactions_df.to_csv("artifacts/interactions_clean.csv", index=False)

    joblib.dump(model, "artifacts/lightfm_model.pkl")
    joblib.dump(dataset, "artifacts/dataset.pkl")
    items_df.to_csv("artifacts/items_clean.csv", index=False)

    print("Saved model and dataset artifacts.")


if __name__ == "__main__":
    main()