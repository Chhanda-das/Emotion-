import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="06664b927668458780591da91df0fe19",
    client_secret="3cb0eeb1b7984ce3ba49eb132e2209d0"
))

results = sp.search(q="happy", type="track", limit=3)
for track in results['tracks']['items']:
    print(track['name'], "by", track['artists'][0]['name'])
