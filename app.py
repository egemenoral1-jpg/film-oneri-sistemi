import streamlit as st
import pandas as pd
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'src')

st.set_page_config(page_title="Film Öneri Sistemi", page_icon="🎬", layout="centered")

st.title("🎬 Hibrit Film Öneri Sistemi")
st.write("Sevdiğin bir filmi seç, hem tür hem de izleyici davranışına dayalı öneriler alalım.")

@st.cache_resource
def load_models():
    movies = joblib.load(os.path.join(MODEL_DIR, 'movies.pkl'))
    cosine_sim = joblib.load(os.path.join(MODEL_DIR, 'cosine_sim.pkl'))
    item_similarity_df = joblib.load(os.path.join(MODEL_DIR, 'item_similarity.pkl'))
    return movies, cosine_sim, item_similarity_df

movies, cosine_sim, item_similarity_df = load_models()

def get_hybrid_recommendations(movie_id, n=5):
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
    return result

movie_titles = movies['title'].sort_values().tolist()
selected_title = st.selectbox("Bir film seç:", movie_titles)

n_recs = st.slider("Kaç öneri gösterilsin?", min_value=3, max_value=10, value=5)

if st.button("Önerileri Göster"):
    movie_id = movies[movies['title'] == selected_title]['movieId'].values[0]
    recs = get_hybrid_recommendations(movie_id, n=n_recs)

    st.subheader(f"'{selected_title}' filmini sevdiysen:")
    for _, row in recs.iterrows():
        st.markdown(f"**{row['title']}**")
        st.caption(row['genres'])
        st.divider()

st.caption("Hibrit sistem: TF-IDF içerik benzerliği + item-based collaborative filtering (max-birleştirme)")