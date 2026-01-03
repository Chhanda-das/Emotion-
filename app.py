import os
import json
import base64
import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_from_directory, jsonify
)

from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
PROFILE_FOLDER = BASE_DIR / "static" / "profile"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
PROFILE_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE_DIR / "emotion_model.h5"

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "super-secret-emo-key"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# Jamendo API
JAMENDO_CLIENT_ID = "c04788ea"

# Emotion labels
EMOTION_LABELS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

# Emotion → Genre
EMOTION_TO_GENRE = {
    "angry": "rock",
    "disgust": "experimental",
    "fearful": "ambient",
    "happy": "pop",
    "neutral": "indie",
    "sad": "acoustic",
    "surprised": "dance"
}

ALLOWED = {"png", "jpg", "jpeg"}

# ======================= LOAD MODEL =======================

if not MODEL_PATH.exists():
    raise FileNotFoundError("emotion_model.h5 missing!")

model = load_model(str(MODEL_PATH))

# ===================== UTIL FUNCTIONS =====================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def preprocess_image(pil_img, size=(224, 224)):
    img = pil_img.convert("RGB").resize(size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, 0)


def predict_emotion(path):
    img = Image.open(path)
    X = preprocess_image(img)
    preds = model.predict(X)[0]

    idx = int(np.argmax(preds))
    label = EMOTION_LABELS[idx]
    conf = float(preds[idx])
    return label, conf


def fetch_jamendo_tracks(genre, limit=12):
    try:
        url = "https://api.jamendo.com/v3.0/tracks"
        params = {
            "client_id": JAMENDO_CLIENT_ID,
            "format": "json",
            "fuzzytags": genre,
            "limit": limit,
            "include": "musicinfo+stats",
            "audioformat": "mp31"
        }

        r = requests.get(url, params=params, timeout=8).json()

        tracks = []
        for t in r.get("results", []):
            tracks.append({
                "name": t.get("name"),
                "artist": t.get("artist_name"),
                "image": t.get("album_image"),
                "audio": t.get("audio")
            })
        return tracks

    except Exception as e:
        print("Jamendo Error:", e)
        return []


def run_full_prediction(path):
    emotion, conf = predict_emotion(str(path))
    genre = EMOTION_TO_GENRE.get(emotion, "pop")
    tracks = fetch_jamendo_tracks(genre)
    return emotion, conf, genre, tracks

# ===================== USER AUTH SYSTEM =====================

USERS_FILE = BASE_DIR / "users.json"

DEFAULT_USER = {
    "daschhanda059@gmail.com": {
        "password": "chhanda@123",
        "full_name": "User",
        "age": "",
        "gender": "",
        "profile_pic": "/static/default-avatar.png",
        "join_date": "2025-12-01"
    }
}

# ensure login exists
if not USERS_FILE.exists():
    with open(USERS_FILE, "w") as f:
        json.dump(DEFAULT_USER, f, indent=4)
else:
    users = json.load(open(USERS_FILE))
    users.update(DEFAULT_USER)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)


def load_users():
    return json.load(open(USERS_FILE))


def save_users(data):
    json.dump(data, open(USERS_FILE, "w"), indent=4)


def get_user_from_db(email):
    users = load_users()

    if email not in users:
        return None

    user = users[email]

    # ensure all fields exist
    user.setdefault("full_name", "User")
    user.setdefault("age", "")
    user.setdefault("gender", "")
    user.setdefault("profile_pic", "/static/default-avatar.png")
    user.setdefault("join_date", "2025-12-01")
    user["email"] = email
    return user


def update_user(email, full_name=None, age=None, gender=None, profile_pic=None):
    users = load_users()

    if email not in users:
        return

    if full_name:
        users[email]["full_name"] = full_name

    if age:
        users[email]["age"] = age

    if gender:
        users[email]["gender"] = gender

    if profile_pic:
        users[email]["profile_pic"] = "/" + profile_pic

    save_users(users)

# ================= LOGIN =================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == "daschhanda059@gmail.com" and password == "chhanda@123":
            session["user"] = username
            return redirect("/")

        return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")

# ================= REGISTER =================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        confirm  = request.form.get("confirm").strip()

        if password != confirm:
            return render_template("register.html", error="Passwords do not match")

        users = load_users()

        if username in users:
            return render_template("register.html", error="User already exists")

        users[username] = {
            "password": password,
            "full_name": "User",
            "age": "",
            "gender": "",
            "profile_pic": "/static/default-avatar.png",
            "join_date": datetime.date.today().isoformat()
        }

        save_users(users)
        return redirect("/login")

    return render_template("register.html")

# ================= PROFILE =================

@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect("/login")

    email = session["user"]
    user = get_user_from_db(email)
    return render_template("profile.html", user=user)


@app.route("/update/<email>")
def edit_profile(email):
    user = get_user_from_db(email)
    return render_template("edit_profile.html", user=user)


@app.route("/update_profile", methods=["POST"])
def update_profile():

    email = session["user"]

    full_name = request.form.get("full_name")
    age = request.form.get("age")
    gender = request.form.get("gender")

    profile_pic = None

    pic = request.files.get("profile_pic")
    if pic and pic.filename:
        filename = secure_filename(pic.filename)
        save_path = "static/profile/" + filename
        pic.save(save_path)
        profile_pic = save_path

    update_user(email, full_name, age, gender, profile_pic)

    return redirect("/profile")


# ================= LOGOUT =================

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ================= HOME PAGE =================

@app.route("/")
def home():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html")

# ================= IMAGE UPLOAD =================

@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return redirect("/")

    file = request.files["file"]
    if not file or not allowed_file(file.filename):
        return redirect("/")

    filename = secure_filename(file.filename)
    path = UPLOAD_FOLDER / filename
    file.save(path)

    emotion, conf, genre, tracks = run_full_prediction(path)

    return jsonify({
        "predicted": {
            "label": emotion,
            "confidence": conf,
            "genre": genre
        },
        "tracks": tracks,
        "image_url": url_for("uploaded_file", filename=filename)
    })

# ================= CAMERA BASE64 =================

@app.route("/api/upload", methods=["POST"])
def api_upload():

    data = request.json.get("image")
    if not data:
        return jsonify({"error": "No image data"}), 400

    header, encoded = data.split(",", 1)
    ext = "png" if "png" in header else "jpg"

    filename = f"cam_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.{ext}"
    path = UPLOAD_FOLDER / filename

    with open(path, "wb") as f:
        f.write(base64.b64decode(encoded))

    emotion, conf, genre, tracks = run_full_prediction(path)

    return jsonify({
        "predicted": {
            "label": emotion,
            "confidence": conf,
            "genre": genre
        },
        "tracks": tracks,
        "image_url": url_for("uploaded_file", filename=filename)
    })

# ================= SERVE UPLOADED FILES =================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# =================== RUN ===================

if __name__ == "__main__":
    app.run(debug=True)
