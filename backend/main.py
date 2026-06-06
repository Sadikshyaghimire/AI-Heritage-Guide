import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from backend.database import init_db, get_connection, now


app = FastAPI(title="Heritage Landmark Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

MODEL_PATH = Path("backend/models/heritage_efficientnet.keras")
CLASS_NAMES_PATH = Path("backend/models/class_names.json")
HERITAGE_INFO_PATH = Path("backend/data/heritage_info.json")

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

with open(HERITAGE_INFO_PATH, "r") as f:
    heritage_info = json.load(f)


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    landmark: str
    question: str
    user_email: str | None = None


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.get("/")
def root():
    return {"message": "Heritage Landmark Recognition API is running"}


@app.post("/signup")
def signup(request: SignupRequest):
    name = request.name.strip()
    email = request.email.strip().lower()
    password = request.password.strip()

    if not name or not email or not password:
        raise HTTPException(
            status_code=400,
            detail="Name, email and password are required"
        )

    password_hash = hash_password(password)

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users (name, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (name, email, password_hash, now())
        )
        conn.commit()

    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")

    conn.close()

    return {
        "message": "Signup successful",
        "user": {
            "name": name,
            "email": email
        }
    }


@app.post("/login")
def login(request: LoginRequest):
    email = request.email.strip().lower()
    password = request.password.strip()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name, email, password_hash FROM users WHERE email = ?",
        (email,)
    )

    user = cursor.fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    name, stored_email, stored_password_hash = user

    if hash_password(password) != stored_password_hash:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "message": "Login successful",
        "user": {
            "name": name,
            "email": stored_email
        }
    }


@app.get("/heritage-info")
def get_all_heritage_info():
    return heritage_info


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_email: str | None = Form(None)
):
    upload_dir = Path("backend/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)[0]

    top_index = int(np.argmax(predictions))
    confidence = float(predictions[top_index])

    top_3_indices = predictions.argsort()[-3:][::-1]

    top_3 = [
        {
            "landmark": class_names[int(i)],
            "confidence": float(predictions[int(i)])
        }
        for i in top_3_indices
    ]

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "predicted_landmark": "unknown",
            "confidence": confidence,
            "top_3_predictions": top_3,
            "info": {
                "name": "No Heritage Landmark Detected",
                "location": "Unknown",
                "type": "Unknown",
                "recognition_enabled": False,
                "description": (
                    "The uploaded image does not appear to clearly match any of "
                    "the trained Kathmandu Valley heritage landmarks. Please upload "
                    "a clear image of one of the supported heritage sites."
                ),
                "recommended": []
            }
        }

    landmark_key = class_names[top_index]
    info = heritage_info.get(landmark_key, {})

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO scan_history
        (user_email, predicted_landmark, confidence, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (user_email, landmark_key, confidence, now())
    )

    conn.commit()
    conn.close()

    return {
        "predicted_landmark": landmark_key,
        "confidence": confidence,
        "top_3_predictions": top_3,
        "info": info
    }


@app.post("/chatbot")
def chatbot(request: ChatRequest):
    landmark_key = request.landmark
    raw_question = request.question.strip()
    question = raw_question.lower()

    if landmark_key == "unknown":
        answer = (
            "I could not identify a supported heritage landmark from the uploaded "
            "image. Please upload a clear photo of one of the trained Kathmandu "
            "Valley heritage sites before asking landmark-specific questions."
        )

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO chat_history
            (user_email, landmark, question, answer, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (request.user_email, landmark_key, request.question, answer, now())
        )

        conn.commit()
        conn.close()

        return {
            "landmark": landmark_key,
            "question": request.question,
            "answer": answer
        }

    info = heritage_info.get(landmark_key, {})

    greetings = [
        "hi", "hello", "hey", "namaste", "good morning",
        "good afternoon", "good evening"
    ]

    thanks_words = ["thank", "thanks", "thank you", "dhanyabad"]

    nonsense_words = [
        "asdf", "qwerty", "blah", "random", "nonsense",
        "lol", "haha", "hehe"
    ]

    unrelated_topics = [
        "football", "cricket", "movie", "song", "game", "dating",
        "politics", "stock", "bitcoin", "crypto", "weather",
        "math", "homework", "exam answer", "cook", "recipe"
    ]

    rude_words = [
        "stupid", "idiot", "shut up", "hate you", "dumb"
    ]

    if not raw_question:
        answer = "Please ask a question about this heritage site."

    elif question in greetings:
        answer = (
            "Namaste! I am your AI heritage guide. You can ask me about this "
            "site's history, location, architecture, cultural importance, or "
            "nearby places to visit."
        )

    elif any(word in question for word in thanks_words):
        answer = (
            "You're welcome! You can ask me more about this heritage site or "
            "ask for nearby recommendations."
        )

    elif any(word in question for word in rude_words):
        answer = (
            "I am here to help with respectful heritage guidance. Please ask "
            "a question about the landmark, its history, architecture, or nearby sites."
        )

    elif any(word in question for word in nonsense_words) or len(question) < 3:
        answer = (
            "I could not understand that question clearly. Please ask something "
            "related to the heritage site, such as its history, location, architecture, "
            "or recommendations."
        )

    elif any(topic in question for topic in unrelated_topics):
        answer = (
            "That question seems unrelated to this heritage guide. I can help with "
            "questions about cultural heritage, history, architecture, location, "
            "recommendations, and visitor information."
        )

    elif "who are you" in question or "what can you do" in question:
        answer = (
            "I am your AI heritage guide. I can explain the recognized landmark, "
            "describe its cultural importance, provide location details, suggest "
            "nearby places, and help you understand Kathmandu Valley heritage sites."
        )

    elif not info:
        answer = (
            "I could not find information about this heritage site in my current "
            "knowledge base."
        )

    elif "where" in question or "location" in question or "located" in question:
        answer = f"{info['name']} is located in {info['location']}."

    elif "type" in question or "what kind" in question or "category" in question:
        answer = f"{info['name']} is categorized as a {info['type']}."

    elif (
        "recommend" in question
        or "nearby" in question
        or "visit next" in question
        or "where should i go" in question
        or "similar place" in question
    ):
        recommended = ", ".join([
            place.replace("_", " ").title()
            for place in info.get("recommended", [])
        ])

        if recommended:
            answer = (
                f"Based on {info['name']}, you may also like to visit: "
                f"{recommended}."
            )
        else:
            answer = (
                f"I do not have specific recommendations for {info['name']} yet, "
                "but nearby cultural sites in Kathmandu Valley may be worth exploring."
            )

    elif (
        "history" in question
        or "origin" in question
        or "old" in question
        or "when" in question
        or "who built" in question
        or "built" in question
    ):
        answer = (
            f"{info['name']} has historical importance within Kathmandu Valley's "
            f"cultural landscape. {info['description']}"
        )

    elif (
        "architecture" in question
        or "design" in question
        or "structure" in question
        or "style" in question
        or "features" in question
    ):
        answer = (
            f"Architecturally, {info['name']} reflects the character of a "
            f"{info['type']}. Its visible features, setting, and cultural use "
            f"help distinguish it from other heritage locations. {info['description']}"
        )

    elif (
        "significance" in question
        or "important" in question
        or "why is" in question
        or "why" in question
        or "famous" in question
    ):
        answer = (
            f"{info['name']} is significant because it represents religious, "
            f"historical, architectural, and cultural identity in Kathmandu Valley. "
            f"{info['description']}"
        )

    elif "unesco" in question or "world heritage" in question:
        if landmark_key in [
            "boudhanath",
            "swayambhunath",
            "pashupatinath",
            "kathmandu_durbar",
            "patan_durbar",
            "bhaktapur_durbar",
            "changunarayan"
        ]:
            answer = (
                f"{info['name']} is associated with the Kathmandu Valley UNESCO "
                f"World Heritage grouping. {info['description']}"
            )
        else:
            answer = (
                f"{info['name']} is culturally important, but it is not one of the "
                "main UNESCO-listed monument zones included in this prototype's "
                "heritage recognition classes."
            )

    elif (
        "ticket" in question
        or "entry fee" in question
        or "opening" in question
        or "time" in question
        or "hours" in question
    ):
        answer = (
            "Visitor hours and entry fees can change over time. For reliable travel "
            "planning, please check the latest official or local tourism information "
            "before visiting. This app currently focuses on recognition and heritage guidance."
        )

    elif (
        "about" in question
        or "describe" in question
        or "tell me" in question
        or "explain" in question
    ):
        answer = info["description"]

    else:
        answer = (
            f"I can answer heritage-related questions about {info['name']}. "
            "Try asking about its history, architecture, significance, location, "
            "UNESCO status, or nearby places to visit."
        )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO chat_history
        (user_email, landmark, question, answer, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (request.user_email, landmark_key, request.question, answer, now())
    )

    conn.commit()
    conn.close()

    return {
        "landmark": landmark_key,
        "question": request.question,
        "answer": answer
    }


@app.get("/scan-history/{user_email}")
def get_scan_history(user_email: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT predicted_landmark, confidence, created_at
        FROM scan_history
        WHERE user_email = ?
        ORDER BY created_at DESC
        """,
        (user_email,)
    )

    rows = cursor.fetchall()
    conn.close()

    return {
        "history": [
            {
                "predicted_landmark": row[0],
                "confidence": row[1],
                "created_at": row[2]
            }
            for row in rows
        ]
    }


@app.get("/chat-history/{user_email}")
def get_chat_history(user_email: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT landmark, question, answer, created_at
        FROM chat_history
        WHERE user_email = ?
        ORDER BY created_at DESC
        """,
        (user_email,)
    )

    rows = cursor.fetchall()
    conn.close()

    return {
        "history": [
            {
                "landmark": row[0],
                "question": row[1],
                "answer": row[2],
                "created_at": row[3]
            }
            for row in rows
        ]
    }