import pandas as pd
import ast

RAW_DIR = "data/raw"

def parse_genres(genre_str):
    try:
        genres_list = ast.literal_eval(genre_str)
        return [g["name"] for g in genres_list]
    except Exception:
        return []
    
# use this function to be able to find unique tmdbId interest then extract the row with column contents (tmdbId/title/genres/overview) 
def create_items(merged: pd.DataFrame) -> pd.DataFrame:
    # Keep only relevant columns
    movie_df = merged[
        ["tmdbId", "title", "genres", "overview", "release_date"]
    ].copy()

    # Ensure one row per movie
    movie_df = movie_df.drop_duplicates(subset=["tmdbId"])

    # Reset index for cleanliness
    movie_df = movie_df.reset_index(drop=True)

    return movie_df

"""
def create_interactions_table(merged):
    return merged[["userId", "tmdbId", "rating"]]
"""

def create_interactions_table(merged):

    interactions = merged[["userId", "tmdbId", "rating", "timestamp"]].copy()

    # keep only latest rating per user-movie
    interactions = interactions.sort_values("timestamp")
    interactions = interactions.drop_duplicates(
        subset=["userId", "tmdbId"],
        keep="last"
    )

    return interactions[["userId", "tmdbId", "rating"]]

def load_raw():
    # Use ratings first for speed; switch to ratings.csv later
    ratings = pd.read_csv(f"{RAW_DIR}/ratings.csv")
    movies = pd.read_csv(f"{RAW_DIR}/movies_metadata.csv", low_memory=False)

    # Basic columns we care about
    movies = movies[["id", "title", "genres", "release_date", "overview"]].copy()
    return ratings, movies

def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    # movies_metadata "id" is TMDB id, but it contains some bad rows / non-numeric ids
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["id", "title"])
    movies["id"] = movies["id"].astype(int)

    # Fill missing text fields
    movies["overview"] = movies["overview"].fillna("")
    movies["genres"] = movies["genres"].fillna("[]")
    movies["release_date"] = movies["release_date"].fillna("")

    # Drop duplicate TMDB ids if any
    movies = movies.drop_duplicates(subset=["id"])
    return movies

def merge_ratings_with_titles(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    # ratings uses MovieLens movieId, not TMDB id
    # Need links.csv to map MovieLens movieId -> tmdbId -> movies_metadata.id
    links = pd.read_csv(f"{RAW_DIR}/links.csv")
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    # Merge: ratings.movieId -> links.movieId -> tmdbId
    r = ratings.merge(links[["movieId", "tmdbId"]], on="movieId", how="inner")
    

    # Merge: tmdbId -> movies_metadata.id
    merged = r.merge(movies, left_on="tmdbId", right_on="id", how="inner")
    

    cols = ["userId", "movieId", "tmdbId", "title", "rating", "timestamp", "genres", "release_date", "overview"]
    merged = merged[cols].copy()
    merged["overview"] = merged["overview"].fillna("")
    merged["genres"] = merged["genres"].fillna("[]")
    # merged["genres_list"] = merged["genres"].apply(parse_genres)
    merged["genres"] = merged["genres"].apply(parse_genres)
    merged["release_date"] = merged["release_date"].fillna("")
    return merged

def main():
    ratings, movies = load_raw()

    print("Raw ratings shape:", ratings.shape)
    print("Raw movies shape:", movies.shape)

    movies = clean_movies(movies)
    print("Clean movies shape:", movies.shape)

    merged = merge_ratings_with_titles(ratings, movies)
    items = create_items(merged)
    intercation = create_interactions_table(merged)

    print("Merged shape:", merged.shape)
    print(merged.head(10))

    print("Movie table shape:", items.shape)
    print(items.head(10))

    print("Interactions table shape", intercation.shape)
    print(intercation.head(10))

    import os
    OUT_PATH = "data/processed"
    os.makedirs(OUT_PATH, exist_ok=True)

    merged.to_csv(f"{OUT_PATH}/ratings_with_titles.csv", index=False)
    items.to_csv(f"{OUT_PATH}/items.csv", index=False)
    intercation.to_csv(f"{OUT_PATH}/interactions.csv", index=False)

    # print("Saved:", out_path)

if __name__ == "__main__":
    main()