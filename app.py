import streamlit as st
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from lightfm.data import Dataset
from recommender_utils import recommend_for_user, fetch_poster_url
from train_model import build_dataset_and_matrices

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/lightfm_model.pkl")
    dataset = joblib.load("artifacts/dataset.pkl")
    items_df = pd.read_csv("artifacts/items_clean.csv")
    interactions_df = pd.read_csv("artifacts/interactions_clean.csv")

    _, _, _, item_features = build_dataset_and_matrices(interactions_df, items_df)
    return model, dataset, items_df, interactions_df, item_features

model, dataset, items_df, interactions_df, item_features = load_artifacts()

st.title("Hybrid Movie Recommender")
st.write("Enter a User ID to see recommended movies.")

user_id_input = st.text_input("User ID", value="1")
load_dotenv()
tmdb_api_key = os.getenv("TMDB_API_KEY")
manual_key = st.text_input("TMDB API Key (optional)", type="password")
if manual_key:
    tmdb_api_key = manual_key


if st.button("Get Recommendations"):
    try:
        user_id = int(user_id_input)
        recs, error = recommend_for_user(
            model,
            dataset,
            items_df,
            interactions_df,
            user_id,
            item_features,
            top_n=10
        )

        if error:
            st.error(error)
        else:
            st.success(f"Top recommendations for User {user_id}")

            cols = st.columns(5)
            for idx, rec in enumerate(recs):
                with cols[idx % 5]:
                    poster_url = None
                    if tmdb_api_key:
                        poster_url = fetch_poster_url(rec["tmdbId"], tmdb_api_key)

                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.write("No poster found")

                    st.markdown(f"**{rec['title']}**")
                    st.caption(rec["release_date"])
                    st.write(f"Score: {rec['score']:.3f}")

    except ValueError:
        st.error("Please enter a valid integer User ID.")