import os
import base64
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, jsonify, request
# Remove PyAudio and Microphone usage:
# import speech_recognition as sr
from elevenlabs.client import ElevenLabs
import google.generativeai as genai
from asgiref.sync import async_to_sync

# Load API keys from environment variables
GEMINI_API_KEY = "AIzaSyCnn1aiF0yMQnkE2DqDFFwUvefOrk-buIE"
ELEVENLABS_API_KEY = "sk_a60d92521f60bc72b7a4cea7fec39527978c5a9ca11db413"

# Configure AI models
genai.configure(api_key=GEMINI_API_KEY)
elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

app = Flask(__name__, template_folder="templates")
conversation_history = []
executor = ThreadPoolExecutor(max_workers=4)

def limit_text(text, max_chars=250):
    """Limit text length to reduce API costs and speed up processing."""
    return text[:max_chars].rsplit(" ", 1)[0] + "..." if len(text) > max_chars else text

def get_audio_bytes(audio_generator, timeout=5):
    """Fetch audio bytes with a timeout for better API response handling."""
    result_queue = queue.Queue()

    def worker():
        try:
            result_queue.put(b"".join(audio_generator))
        except Exception as e:
            result_queue.put(e)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout)

    return result_queue.get() if not thread.is_alive() else None

def generate_response(user_input):
    """Get AI response from Gemini."""
    try:
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(user_input)
        return response.text.strip() if response and hasattr(response, "text") else "I didn't understand."
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Error processing your request."

def text_to_speech(text):
    """Convert text to speech using ElevenLabs API."""
    try:
        audio_generator = elevenlabs.text_to_speech.convert(
            text=limit_text(text),
            voice_id="Xb7hH8MSUJpSbSDYk0k2",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        audio_bytes = get_audio_bytes(audio_generator, timeout=5)
        if audio_bytes is None:
            print("TTS conversion timed out or failed.")
        return base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/voice", methods=["POST"])
def voice_assistant():
    """
    Handles voice interaction, but now relies on text input from the request
    rather than using a microphone.
    """
    global conversation_history

    # Get text from the POST form (or JSON) instead of using a microphone
    user_input = request.form.get("user_input") or request.json.get("user_input") if request.is_json else None

    if not user_input:
        # If no user_input was provided, respond with a default message
        response_text = "How can I assist you today?"
        conversation_history.append({"role": "agent", "text": response_text})
    else:
        # Add user's text to the conversation history
        conversation_history.append({"role": "user", "text": user_input})
        # Generate AI response
        response_text = generate_response(user_input)
        conversation_history.append({"role": "agent", "text": response_text})

    # Convert the response text to speech
    audio_base64 = text_to_speech(response_text)

    return jsonify({
        "response": response_text,
        "audio": f"data:audio/mp3;base64,{audio_base64}" if audio_base64 else None,
        "conversation": conversation_history
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
