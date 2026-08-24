"""
Film Öneri Sistemi - Hibrit Model
Content-based (TF-IDF) + Collaborative Filtering (item-based, mean-centered) + max-birleştirme.
"""

import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.dirname(__file__)


def download_data():
    """MovieLens 100K veri setini indirir (yoksa)."""
    ml_dir = os.path.join(DATA_DIR, 'ml-100k')
    if os.path.exists(ml_dir):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = os.path.join(DATA_DIR, 'ml-100k.zip')
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)


def load_data():
    """Ratings ve movies verisini yükler, tür bilgilerini okunabilir hale getirir."""
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, 'ml-100k', 'u.data'),
        sep='\t', names=['userId', 'movieId', 'rating', 'timestamp']
    )

    movie_cols = ['movieId', 'title', 'release_date', 'video_release_date', 'imdb_url'] + \
                 [f'genre_{i}' for i in range(19)]
    movies = pd.read_csv(
        os.path.join(DATA_DIR, 'ml-100k', 'u.item'),
        sep='|', names=movie_cols, encoding='latin-1'
    )

    genre_names = pd.read_csv(
        os.path.join(DATA_DIR, 'ml-100k', 'u.genre'),
        sep='|', header=None, names=['genre', 'genre_id']
    )
    genre_list = genre_names['genre'].tolist()
    genre_cols = [f'genre_{i}' for i in range(19)]

    def get_genres(row):
        return '|'.join([genre_list[i] for i in range(19) if row[genre_cols[i]] == 1])

    movies['genres'] = movies.apply(get_genres, axis=1)
    movies['genres_str'] = movies['genres'].str.replace('|', ' ', regex=False)

    return ratings, movies


def build_content_similarity(movies):
    """TF-IDF + cosine similarity ile content-based benzerlik matrisi."""
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies['genres_str'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def build_collaborative_similarity(ratings):
    """Mean-centered item-based collaborative filtering benzerlik matrisi."""
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
    user_means = user_movie_matrix.mean(axis=1)
    centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)
    movie_user_matrix = centered.T
    sim = cosine_similarity(movie_user_matrix)
    return pd.DataFrame(sim, index=movie_user_matrix.index, columns=movie_user_matrix.index)


def get_hybrid_recommendations(movie_id, movies, cosine_sim, item_similarity_df, n=5):
    """Max-based birleştirme ile hibrit öneri üretir."""
    movie_idx = movies[movies['movieId'] == movie_id].index[0]
    content_scores = pd.Series(cosine_sim[movie_idx], index=movies['movieId'].values)
    collab_scores = item_similarity_df[movie_id]

    content_rank = content_scores.rank(pct=True)
    collab_rank = collab_scores.rank(pct=True)

    hybrid_scores = pd.concat([content_rank, collab_rank], axis=1).max(axis=1)
    hybrid_scores = hybrid_scores.drop(movie_id)
    hybrid_scores = hybrid_scores.sort_values(ascending=False)

    top_movie_ids = hybrid_scores.head(n).index
    result = movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title', 'genres']]
    result = result.set_index('movieId').loc[top_movie_ids].reset_index()
    result['hybrid_score'] = hybrid_scores.head(n).values
    return result


def main():
    print("Veri indiriliyor (varsa atlanacak)...")
    download_data()

    print("Veri yükleniyor...")
    ratings, movies = load_data()

    print("Content-based benzerlik matrisi hesaplanıyor...")
    cosine_sim = build_content_similarity(movies)

    print("Collaborative filtering benzerlik matrisi hesaplanıyor...")
    item_similarity_df = build_collaborative_similarity(ratings)

    print("Modeller kaydediliyor...")
    joblib.dump(movies, os.path.join(MODEL_DIR, 'movies.pkl'))
    joblib.dump(cosine_sim, os.path.join(MODEL_DIR, 'cosine_sim.pkl'))
    joblib.dump(item_similarity_df, os.path.join(MODEL_DIR, 'item_similarity.pkl'))

    print("\nTest: Toy Story (movieId=1) için öneriler:")
    print(get_hybrid_recommendations(1, movies, cosine_sim, item_similarity_df))


if __name__ == "__main__":
    main()