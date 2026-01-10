import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.markdown("""
<style>
.main .block-container {background-color: #fff8e1;padding-top: 2rem;}
[data-testid="stSidebar"] {background: linear-gradient(to bottom, #ffecb3, #ffe0b2) !important;padding: 1rem !important;color: #333 !important;}
.stTitle {color: #d81b60 !important;font-size: 3rem;}
h1, h2, h3 {color: #1976d2 !important;}
.stSuccess {background-color: #e8f5e8;color: #2e7d32;}
.stTabs [data-baseweb="tab-list"] {background: #ffccbc;}
.stTabs [data-baseweb="tab"] {color: #bf360c;}
.stSelectbox label {color: #1a1a1a !important;font-weight: bold;font-size: 1.2rem;}
.stSelectbox div div {color: #000000 !important;background-color: #ffffff !important;}
</style>
""", unsafe_allow_html=True)



st.title("🎵🎬 CINETUNE")
st.markdown("**Content-Based (TF-IDF + Cosine) & Popularity Filtering**")

# Dummy data (replace with pd.read_csv('songs.csv') / 'movies.csv' if available)
songs_df = pd.DataFrame({
    'title': ['Bohemian Rhapsody', 'Hotel California', 'Stairway to Heaven', 'Imagine', 'Sweet Child O Mine'],
    'artist': ['Queen', 'Eagles', 'Led Zeppelin', 'John Lennon', 'Guns N Roses'],
    'genre': ['Rock', 'Rock', 'Rock', 'Pop', 'Rock'],
    'mood': ['Energetic', 'Calm', 'Energetic', 'Inspirational', 'Energetic'],
    'rating': [9.5, 9.2, 9.4, 9.0, 9.3]
})

movies_df = pd.DataFrame({
    'title': ['Inception', 'The Godfather', 'Pulp Fiction', 'Forrest Gump', 'The Shawshank Redemption'],
    'director': ['Christopher Nolan', 'Francis Ford Coppola', 'Quentin Tarantino', 'Robert Zemeckis', 'Frank Darabont'],
    'genre': ['Sci-Fi', 'Crime', 'Crime', 'Drama', 'Drama'],
    'imdb': [8.8, 9.2, 8.9, 8.8, 9.3]
})

# Sidebar for category choice
category = st.sidebar.selectbox("Choose category:", ["Music", "Movies"])

if category == "Music":
    df = songs_df.copy()
    col1, col2 = 'artist', 'genre'
else:
    df = movies_df.copy()
    col1, col2 = 'director', 'genre'

st.subheader(f"**{category} Recommendations**")

tab1, tab2 = st.tabs(["📊 Content-Based", "⭐ Popularity-Based"])

with tab1:
    input_title = st.text_input("Enter a song/movie title:", value="Bohemian Rhapsody" if category=="Music" else "Inception")
    
    if st.button("Get Recommendations", key="content"):
        if input_title.lower() not in df['title'].str.lower().values:
            st.error("Title not found! Try one from the list.")
        else:
            # Fill NaNs
            df[col1] = df[col1].fillna('')
            df[col2] = df[col2].fillna('')
            
            # TF-IDF
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df[col1] + ' ' + df[col2])
            
            # Cosine similarity
            idx = df[df['title'].str.lower() == input_title.lower()].index[0]
            sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
            
            n = st.slider("Number of recommendations:", 1, 5, 3)
            top_items = sim_scores[:n]
            
            st.success(f"**Top {n} similar {category.lower()}s:**")
            for i, (i_idx, score) in enumerate(top_items):
                title = df['title'].iloc[i_idx]
                st.write(f"{i+1}. **{title}** (Similarity: {score:.2f})")

with tab2:
    filter_col = 'mood' if category=="Music" else 'genre'
    selected_filter = st.selectbox("Select mood/genre:", df[filter_col].unique())
    n_pop = st.slider("Top N popular:", 1, 5, 3)
    
    if st.button("Get Popular Recommendations", key="pop"):
        filtered = df[df[filter_col] == selected_filter].sort_values('rating' if category=="Music" else 'imdb', ascending=False)
        top_pop = filtered.head(n_pop)
        
        st.success(f"**Top {n_pop} {selected_filter} {category.lower()}s:**")
        for _, row in top_pop.iterrows():
            rating_col = 'rating' if category=="Music" else 'imdb'
            st.write(f"- **{row['title']}** ({row[rating_col]})")

st.markdown("---")
st.caption("Built with Streamlit + scikit-learn. Load real CSV for production!")

