# Hybrid Movie Recommender

A hybrid movie recommendation system using:

- LightFM
- Movie metadata features
- Streamlit UI
- TMDB poster API

## Setup

Clone the repository:

git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender

Create the environment:

conda env create -f environment.yml
conda activate movierec

Add your TMDB API key:

Create a `.env` file:

TMDB_API_KEY=your_tmdb_api_key_here

Download the dataset:

kaggle datasets download -d rounakbanik/the-movies-dataset -p data/raw --unzip

Run preprocessing:

python preprocess.py

Train the model:

python train_model.py

Run the app:

streamlit run app.py
