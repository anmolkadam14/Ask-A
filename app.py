from flask import Flask, request, jsonify, send_from_directory
import os
import speech_recognition as sr

from backend import generate_response
from memory import add_to_memory, get_memory_context

app = Flask(__name__)


@app.route("/")
def serve_ui():
    return send_from_directory(os.path.dirname(__file__), "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data.get("prompt", "").strip()

    if not user_text:
        return jsonify({"reply": "Please type or say something."})

    add_to_memory("User", user_text)

    system_prompt = (
        "You are Ask-A, an AI Voice Recognition Chatbot built for education "
        "and general assistance. Respond clearly and politely."
    )

    full_prompt = f"""
{system_prompt}

Conversation so far:
{get_memory_context()}

User: {user_text}
Ask-A:
"""

    reply = generate_response(full_prompt)
    add_to_memory("Ask-A", reply)

    return jsonify({"reply": reply})


@app.route("/voice", methods=["POST"])
def voice():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        text = recognizer.recognize_google(audio)
        return jsonify({"text": text})

    except Exception:
        return jsonify({"text": ""})


if __name__ == "__main__":
    app.run(debug=True)
