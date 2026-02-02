import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="KinoGo ML")
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# --- 1. CHARGEMENT ET PRÉPARATION ML ---
@st.cache_resource
def prepare_ml(df):
    # Nettoyage pour éviter l'erreur 'float'
    df['genres_text'] = df['genres_text'].fillna('').astype(str)
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['genres_text'])
    knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
    knn.fit(tfidf_matrix)
    return tfidf_matrix, knn

@st.cache_data
def load_data():
    path = "data/df_ml_ready.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_data()

if df is not None:
    tfidf_matrix, knn_model = prepare_ml(df)
else:
    st.error("Fichier de données manquant !")
    st.stop()

# --- 2. FONCTION DE RECOMMANDATION ---
def get_recommendations(title, df, model, tfidf_mat, n=5):
    # On cherche l'index exact dans le dataframe original
    if title not in df['title'].values:
        return None
    idx = df[df['title'] == title].index[0]
    distances, indices = model.kneighbors(tfidf_mat[idx], n_neighbors=n+1)
    similar_indices = indices[0][1:]
    return df.iloc[similar_indices]

# --- 3. STYLE CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .movie-card {
        background-color: #1f2630; border-radius: 8px; margin-bottom: 10px;
        transition: transform 0.3s; border: 1px solid #333;
        height: 420px; overflow: hidden; text-align: center;
    }
    .movie-card:hover { transform: scale(1.02); border-color: #ff9d00; }
    .movie-img { width: 100%; height: 300px; object-fit: cover; }
    .movie-title { color: white; font-weight: bold; font-size: 13px; padding: 5px; height: 40px; overflow: hidden; }
    .movie-rating { background-color: #ff9d00; color: black; padding: 1px 5px; border-radius: 4px; font-weight: bold; font-size: 11px; }
</style>
""", unsafe_allow_html=True)

if 'opened_movie_id' not in st.session_state:
    st.session_state.opened_movie_id = None

# --- 4. BARRE LATÉRALE ---
with st.sidebar:
    st.markdown("<h2 style='color:#ff9d00;'>KinoGo ML</h2>", unsafe_allow_html=True)
    search = st.text_input("🔍 Rechercher un titre...")
    
    years = sorted(df['year'].dropna().unique().astype(int), reverse=True)
    year_filter = st.multiselect("Année", years)
    
    all_genres = set()
    df['genres_text'].dropna().str.split(' ').apply(lambda x: [all_genres.add(g.strip()) for g in x if g])
    genre_filter = st.selectbox("Genre", ["Tous"] + sorted(list(all_genres)))

# --- 5. LOGIQUE DE FILTRAGE ---
filtered_df = df.copy()
if search:
    filtered_df = filtered_df[filtered_df['title'].str.contains(search, case=False, na=False)]
if year_filter:
    filtered_df = filtered_df[filtered_df['year'].isin(year_filter)]
if genre_filter != "Tous":
    filtered_df = filtered_df[filtered_df['genres_text'].str.contains(genre_filter.lower(), na=False)]

# --- 6. AFFICHAGE ---

# CAS A : L'utilisateur a cherché un film spécifique (On affiche le film + Recos)
if search and len(filtered_df) > 0:
    main_movie = filtered_df.iloc[0]
    st.title(f"🎬 Résultat pour : {main_movie['title']}")
    
    # Affichage du film principal
    col1, col2 = st.columns([1, 3])
    with col1:
        img = TMDB_IMAGE_BASE_URL + str(main_movie['poster_path']) if pd.notna(main_movie['poster_path']) else "https://via.placeholder.com/300x450"
        st.image(img)
    with col2:
        st.write(f"**Année :** {int(main_movie['year'])}")
        st.write(f"**Note :** ⭐ {main_movie['rating']}")
        st.write(f"**Synopsis :** {main_movie['overview']}")
    
    st.markdown("---")
    st.subheader("🔥 Parce que vous avez regardé ce film...")
    
    # Appel du ML pour les recommandations
    recom_df = get_recommendations(main_movie['title'], df, knn_model, tfidf_matrix)
    
    if recom_df is not None:
        rec_cols = st.columns(5)
        for idx, (_, row) in enumerate(recom_df.iterrows()):
            with rec_cols[idx]:
                p_url = TMDB_IMAGE_BASE_URL + str(row['poster_path']) if pd.notna(row['poster_path']) else "https://via.placeholder.com/300x450"
                st.markdown(f"""
                    <div class="movie-card">
                        <img class="movie-img" src="{p_url}">
                        <div class="movie-title">{row['title']}</div>
                        <div style="margin-top:5px;">
                            <span style="color:#888; font-size:11px;">{int(row['year'])}</span>
                            <span class="movie-rating">★ {row['rating']}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# CAS B : Pas de recherche (On affiche la grille normale des nouveautés)
else:
    st.title("🎬 Nouveautés")
    cols_per_row = 5
    display_df = filtered_df.head(40) 

    for i in range(0, len(display_df), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(display_df):
                row = display_df.iloc[i + j]
                movie_id = row['imdb_id']
                p_url = TMDB_IMAGE_BASE_URL + str(row['poster_path']) if pd.notna(row['poster_path']) else "https://via.placeholder.com/300x450"
                
                with cols[j]:
                    st.markdown(f"""
                        <div class="movie-card">
                            <img class="movie-img" src="{p_url}">
                            <div class="movie-title">{row['title']}</div>
                            <div style="margin-top:5px;">
                                <span style="color:#888; font-size:11px;">{int(row['year'])}</span>
                                <span class="movie-rating">★ {row['rating']}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("Infos", key=f"btn_{movie_id}"):
                        st.session_state.opened_movie_id = None if st.session_state.opened_movie_id == movie_id else movie_id
                        st.rerun()

                if st.session_state.opened_movie_id == movie_id:
                    st.info(f"**Synopsis:** {row['overview']}")