from google import genai
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
from pydantic import BaseModel



# Load .env file
load_dotenv()



# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")



#create gemini client
client = genai.Client(api_key=GEMINI_API_KEY)



# Initialize FastAPI
app = FastAPI()

print("LOADED THE CORRECT rag_app.py")

print("Registered routes:")
for route in app.routes:
    print(route.path, route.methods)



class QueryRequest(BaseModel):
    question: str



def validate_user_input(text: str):
    if text is None or text.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(text) < 5:
        raise HTTPException(status_code=400, detail="Question is too short")

    if len(text) > 500:
        raise HTTPException(status_code=400, detail="Question is too long")

def validate_model_output(text: str):
    if text is None or text.strip() == "":
        raise HTTPException(
            status_code=500,
            detail="AI returned an empty response"
        )

    if len(text) < 10:
        raise HTTPException(
            status_code=500,
            detail="AI response is too short"
        )



#health check of the ssh
@app.get("/health")
def health():
    return {"status": "ok"}



#checks that the ai works
@app.get("/test-gemini")
def gemini_endpoint():
    return test_gemini()



#tests the ai
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



def review_model_output(original_answer: str):
    review_prompt = f"""
You are reviewing an AI-generated response.

Your job:
- If the response is unclear, incomplete, or poorly written, improve it.
- If the response is already good, return it unchanged.

AI response to review:

{original_answer}
"""

    review_response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=review_prompt
    )

    return review_response.text



@app.post("/query")
def query_ai(request: QueryRequest):
    validate_user_input(request.question)

    primary_response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=request.question
    )

    raw_answer = primary_response.text

    validate_model_output(raw_answer)

    reviewed_answer = review_model_output(raw_answer)

    return {
        "question": request.question,
        "answer": reviewed_answer
    }



print("Registered routes:")
for route in app.routes:
    print(route.path, route.methods)
