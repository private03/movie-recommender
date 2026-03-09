import pandas as pd
import numpy as np
import ast
import requests

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def parse_genres(genres_str):
    if pd.isna(genres_str):
        return []
    try:
        parsed = ast.literal_eval(genres_str)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


def get_internal_mappings(dataset):
    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
    return user_id_map, item_id_map


def recommend_for_user(model, dataset, items_df, interactions_df, user_id, item_features, top_n=10):
    user_id_map, item_id_map = get_internal_mappings(dataset)

    if user_id not in user_id_map:
        return None, f"User ID {user_id} not found."

    internal_user_id = user_id_map[user_id]

    reverse_item_map = {internal: original for original, internal in item_id_map.items()}

    seen_tmdb_ids = set(
        interactions_df.loc[interactions_df["userId"] == user_id, "tmdbId"].astype(int).tolist()
    )

    all_internal_item_ids = np.arange(len(reverse_item_map))
    scores = model.predict(
        internal_user_id,
        all_internal_item_ids,
        item_features=item_features
    )

    ranked_internal_ids = np.argsort(-scores)

    recs = []
    for internal_item_id in ranked_internal_ids:
        tmdb_id = reverse_item_map[internal_item_id]

        if tmdb_id in seen_tmdb_ids:
            continue

        movie_rows = items_df[items_df["tmdbId"] == tmdb_id]
        if movie_rows.empty:
            continue

        movie = movie_rows.iloc[0]
        recs.append({
            "tmdbId": int(tmdb_id),
            "title": movie.get("title", "Unknown"),
            "release_date": movie.get("release_date", ""),
            "genres": movie.get("genres", "[]"),
            "overview": movie.get("overview", ""),
            "score": float(scores[internal_item_id]),
        })

        if len(recs) >= top_n:
            break

    return recs, None


def fetch_poster_url(tmdb_id, tmdb_api_key):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {"api_key": tmdb_api_key}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"{TMDB_IMAGE_BASE}{poster_path}"
    except Exception:
        return None

    return None