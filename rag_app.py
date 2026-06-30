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

    try:
        # STEP 1: Generate an outline
        outline_response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents="Create a 3-point outline explaining what a large language model is."
        )

        outline = outline_response.text

        # Optional: print the outline for debugging
        print("Generated outline:")
        print(outline)

        # STEP 2: Expand the outline into a paragraph
        final_response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=f"""
Use the following outline to write one clear paragraph.

Outline:
{outline}
"""
        )

        # Return only the final response
        return {
            "response": final_response.text
        }

    except Exception as e:
        return {
            "error": str(e)
        }