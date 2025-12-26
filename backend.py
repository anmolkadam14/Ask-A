import os
import subprocess
from google import genai

# Gemini client (optional)
GEMINI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
client = None

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)


def ollama_fallback(prompt: str) -> str:
    """
    Local LLM fallback using Ollama
    Make sure: ollama pull llama3
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=60
        )
        return result.stdout.strip()
    except Exception:
        return "I'm currently running in offline mode. Please try again later."


def generate_response(prompt: str) -> str:
    """
    Try Gemini first → fallback to Ollama
    """
    if client:
        try:
            response = client.models.generate_content(
                model="models/gemini-2.0-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception:
            # Do NOT spam errors
            pass

    return ollama_fallback(prompt)
