import streamlit as st
import pandas as pd

# Load dataset
songs = pd.read_csv("songs1.csv")

# Initialize user profile in session
if "profile" not in st.session_state:
    st.session_state.profile = {
        "genres": {},
        "artists": {},
        "songs_liked": []
    }

st.title("🎵 Cinetune - User Profiling Recommender")

# User inputs
genre = st.selectbox("Select Genre", sorted(songs["genre"].unique()))
artist = st.selectbox("Select Artist", sorted(songs["artist"].unique()))
song_name = st.text_input("Enter Song Name (optional)")

# Like button
if st.button("Like Song"):
    profile = st.session_state.profile

    profile["genres"][genre] = profile["genres"].get(genre, 0) + 1
    profile["artists"][artist] = profile["artists"].get(artist, 0) + 1
    profile["songs_liked"].append(song_name if song_name else f"{artist} - {genre}")

    st.success("User profile updated!")

# Show profile
st.subheader("👤 User Profile")
st.json(st.session_state.profile)

# Recommendation logic
def get_top_pref(pref_dict):
    return max(pref_dict, key=pref_dict.get) if pref_dict else None

def recommend():
    profile = st.session_state.profile
    top_genre = get_top_pref(profile["genres"])
    top_artist = get_top_pref(profile["artists"])

    if not top_genre and not top_artist:
        return songs.head(5)

    genre_match = songs["genre"].str.contains(str(top_genre), case=False, na=False)
    artist_match = songs["artist"].str.contains(str(top_artist), case=False, na=False)

    return songs[genre_match | artist_match].head(5)

# Show recommendations
st.subheader("🎯 Personalized Recommendations")
st.dataframe(recommend())
