# monitoring.py
# -------------
# This file monitors the quality of our RAG app's responses.
#
# What is hallucination?
# Even when we give an LLM context documents, it sometimes generates
# information that isn't actually in those documents. It "fills in the gaps"
# with plausible-sounding but unverified facts. This is called hallucination.
#
# How do we detect it?
# We use a technique called "LLM-as-judge": we send the answer AND the
# source documents back to Gemini and ask it to evaluate whether the answer
# is actually supported by the context. This is a common pattern in
# production RAG systems.

from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL
import csv
from datetime import datetime

_client = genai.Client(api_key=GEMINI_API_KEY)



def log_hallucination_check(answer, verdict, confidence):
    """Save hallucination monitoring results to a CSV log."""
    log_file = "hallucination_log.csv"

    file_exists = False
    try:
        with open(log_file, "r", encoding="utf-8"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(log_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "verdict",
                "confidence",
                "answer"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            verdict,
            confidence,
            answer
        ])



def check_hallucination(answer, context_docs):
    """
    Ask Gemini to evaluate whether the generated answer is grounded in
    the source documents that were retrieved.

    Args:
        answer: The answer our app generated.
        context_docs: The documents we retrieved and used as context.

    Returns:
        A dictionary with:
          - "verdict": "GROUNDED", "PARTIAL", or "HALLUCINATED"
          - "is_grounded": True if verdict is GROUNDED, False otherwise
          - "warning": A warning string to show the user (empty if grounded)
    """
    try:
        context = "\n\n".join(
            [f"Document {i+1}: {doc}" for i, doc in enumerate(context_docs)]
        )

        prompt = f"""
You are evaluating whether an AI answer is supported by the source documents.

Source Documents:
{context}

Generated Answer:
{answer}

Classify the answer using exactly ONE of these words:

GROUNDED
PARTIAL
HALLUCINATED

Respond with ONLY one word.
"""

        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            ),
        )

        verdict = response.text.strip().upper()


        if verdict not in ["GROUNDED", "PARTIAL", "HALLUCINATED"]:
            verdict = "PARTIAL"

        if verdict == "GROUNDED":
            warning = ""
        elif verdict == "PARTIAL":
            warning = (
                "Note: This answer may include some information beyond the provided sources."
            )
        else:
            warning = (
                "Warning: This answer may contain information not found in the source documents."
            )

        return {
            "verdict": verdict,
            "is_grounded": verdict == "GROUNDED",
            "warning": warning,
        }

    except Exception:
        return {
            "verdict": "UNKNOWN",
            "is_grounded": True,
            "warning": "",
        }

def calculate_confidence(distances):
    """
    Convert ChromaDB similarity distances into a 0–1 confidence score.

    Args:
        distances: A list of L2 distance values from ChromaDB.
                   0.0 = identical vectors, 2.0 = completely different.

    Returns:
        A float between 0.0 (not confident) and 1.0 (very confident).
    """
    # TODO (Week 13): Implement the confidence score calculation.
    #
    # --- The RAG concept ---
    # When ChromaDB retrieves documents, it returns a "distance" for each one.
    # Distance measures how far apart two vectors are in embedding space.
    # A low distance means the document is very similar to the query —
    # which means we can be more confident the answer will be relevant.
    #
    # The formula to convert distance to confidence:
    #   confidence = max(0.0, 1.0 - (avg_distance / 2.0))
    #
    # Why divide by 2? ChromaDB L2 distances range from 0 to 2, so
    # dividing by 2 scales the result to a 0–1 range.
    #
    # Steps:
    #   1. If distances is empty, return 0.0
    #   2. Compute the average: avg_distance = sum(distances) / len(distances)
    #   3. Apply the formula above
    #   4. Return the result rounded to 2 decimal places: round(confidence, 2)
    #
   
    if not distances:
        return 0.0

    avg_distance = sum(distances) / len(distances)

    confidence = max(0.0, 1.0 - (avg_distance / 2.0))

    return round(confidence, 2)



