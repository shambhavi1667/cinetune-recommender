
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------- STYLING --------------------
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

# -------------------- TITLE --------------------
st.title("🎵🎬 CINETUNE")
st.markdown("**Content-Based, Popularity Filtering & Personalized User Profiling**")

# -------------------- USER PROFILE MEMORY --------------------
if "profile" not in st.session_state:
    st.session_state.profile = {
        "genres": {},
        "artists": {},
        "directors": {},
        "history": []
    }

def update_profile(genre, person, title, category):
    profile = st.session_state.profile

    profile["genres"][genre] = profile["genres"].get(genre, 0) + 1

    if category == "Music":
        profile["artists"][person] = profile["artists"].get(person, 0) + 1
    else:
        profile["directors"][person] = profile["directors"].get(person, 0) + 1

    profile["history"].append(f"{category}: {title}")

def get_top_pref(pref_dict):
    return max(pref_dict, key=pref_dict.get) if pref_dict else None

# -------------------- DATA --------------------
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

# -------------------- SIDEBAR --------------------
category = st.sidebar.selectbox("Choose category:", ["Music", "Movies"])

if category == "Music":
    df = songs_df.copy()
    col1, col2 = 'artist', 'genre'
    person_label = "Artist"
else:
    df = movies_df.copy()
    col1, col2 = 'director', 'genre'
    person_label = "Director"

st.subheader(f"**{category} Recommendations**")

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["📊 Content-Based", "⭐ Popularity-Based", "👤 Personalized (User Profiling)"])

# -------------------- CONTENT-BASED --------------------
with tab1:
    input_title = st.text_input(
        "Enter a song/movie title:",
        value="Bohemian Rhapsody" if category == "Music" else "Inception"
    )

    if st.button("Get Recommendations", key="content"):
        if input_title.lower() not in df['title'].str.lower().values:
            st.error("Title not found! Try one from the list.")
        else:
            df[col1] = df[col1].fillna('')
            df[col2] = df[col2].fillna('')

            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df[col1] + ' ' + df[col2])

            idx = df[df['title'].str.lower() == input_title.lower()].index[0]
            sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

            n = st.slider("Number of recommendations:", 1, 5, 3)
            top_items = sim_scores[:n]

            st.success(f"**Top {n} similar {category.lower()}s:**")
            for i, (i_idx, score) in enumerate(top_items):
                title = df['title'].iloc[i_idx]
                st.write(f"{i+1}. **{title}** (Similarity: {score:.2f})")

# -------------------- POPULARITY-BASED --------------------
with tab2:
    filter_col = 'mood' if category == "Music" else 'genre'
    selected_filter = st.selectbox("Select mood/genre:", df[filter_col].unique())
    n_pop = st.slider("Top N popular:", 1, 5, 3)

    if st.button("Get Popular Recommendations", key="pop"):
        filtered = df[df[filter_col] == selected_filter].sort_values(
            'rating' if category == "Music" else 'imdb',
            ascending=False
        )

        st.success(f"**Top {n_pop} {selected_filter} {category.lower()}s:**")
        for _, row in filtered.head(n_pop).iterrows():
            rating_col = 'rating' if category == "Music" else 'imdb'
            st.write(f"- **{row['title']}** ({row[rating_col]})")

# -------------------- USER PROFILING --------------------
with tab3:
    st.markdown("### 👍 Like a Song/Movie (Build Your Profile)")

    selected_title = st.selectbox("Choose title:", df["title"].unique())
    selected_row = df[df["title"] == selected_title].iloc[0]

    if category == "Music":
        person = selected_row["artist"]
    else:
        person = selected_row["director"]

    genre = selected_row["genre"]

    if st.button("Like This"):
        update_profile(genre, person, selected_title, category)
        st.success("Profile updated! Preferences learned.")

    # --------- USER PROFILE UI ---------
    st.markdown("### 👤 User Profile")

    profile = st.session_state.profile

    colA, colB, colC = st.columns(3)

    # Favorite Genre
    with colA:
        top_genre = get_top_pref(profile["genres"])
        st.metric("⭐ Favorite Genre", top_genre if top_genre else "None")

    # Favorite Artist / Director
    with colB:
        if category == "Music":
            top_person = get_top_pref(profile["artists"])
            st.metric("🎤 Favorite Artist", top_person if top_person else "None")
        else:
            top_person = get_top_pref(profile["directors"])
            st.metric("🎬 Favorite Director", top_person if top_person else "None")

    # Total Likes
    with colC:
        st.metric("❤️ Total Likes", len(profile["history"]))

    # Recently liked list
    st.markdown("### 🕒 Recently Liked")
    if profile["history"]:
        for item in profile["history"][-5:][::-1]:
            st.write("•", item)
    else:
        st.info("No activity yet. Like something to build your profile!")

    # --------- PERSONALIZED RECOMMENDATIONS ---------
    st.markdown("### 🎯 Personalized Recommendations")

    top_genre = get_top_pref(profile["genres"])

    if category == "Music":
        top_person = get_top_pref(profile["artists"])
    else:
        top_person = get_top_pref(profile["directors"])

    if not top_genre and not top_person:
        st.info("Like some items to get personalized recommendations!")
        st.dataframe(df.head(3))
    else:
        genre_match = df["genre"].str.contains(str(top_genre), case=False, na=False)
        person_match = df[col1].str.contains(str(top_person), case=False, na=False)

        results = df[genre_match | person_match].head(5)
        st.dataframe(results)

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Built with Streamlit + scikit-learn + User Profiling AI")
