from google import genai
from fastapi import FastAPI
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")

# Initialize FastAPI
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/test-gemini")
def gemini_endpoint():
    return test_gemini()


def test_gemini():
    # Create Gemini client
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Hardcoded prompt (no user input yet)
    prompt = "Explain what a large language model is in one paragraph."

    # Call Gemini (USING YOUR AVAILABLE MODEL)
    response = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents=prompt
    )

    # Return API response safely
    return {
        "prompt": prompt,
        "response": response.text
    }