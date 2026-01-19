import pandas as pd

def create_user_profile():
    return {
        "genres": {},
        "artists": {},
        "movies_watched": [],
        "songs_liked": []
    }

def update_profile(profile, genre, artist, name, item_type):
    if genre in profile["genres"]:
        profile["genres"][genre] += 1
    else:
        profile["genres"][genre] = 1

    if artist in profile["artists"]:
        profile["artists"][artist] += 1
    else:
        profile["artists"][artist] = 1

    if item_type == "movie":
        profile["movies_watched"].append(name)
    else:
        profile["songs_liked"].append(name)

def get_top_preferences(profile):
    top_genre = max(profile["genres"], key=profile["genres"].get, default=None)
    top_artist = max(profile["artists"], key=profile["artists"].get, default=None)
    return top_genre, top_artist

def recommend_from_profile(data, profile):
    genre, artist = get_top_preferences(profile)

    if genre is None and artist is None:
        return data.head(5)

    # Case-insensitive + partial matching
    genre_match = data["genre"].str.contains(str(genre), case=False, na=False)
    artist_match = data["artist"].str.contains(str(artist), case=False, na=False)

    return data[genre_match | artist_match].head(5)


# DEMO RUN
if __name__ == "__main__":
    print("Loading songs dataset...")
    songs = pd.read_csv("songs1.csv")

    print("\nAvailable Genres:")
    print(songs["genre"].unique())

    print("\nAvailable Artists:")
    print(songs["artist"].unique())

    user = create_user_profile()

    print("\nSimulating user liking songs...")
    update_profile(user, "pop", "adele", "Someone Like You", "song")
    update_profile(user, "pop", "the weeknd", "Blinding Lights", "song")


    print("\nUser Profile:")
    print(user)

    print("\nRecommended Songs:")
    print(recommend_from_profile(songs, user))

