# workflow.py
# -----------
# This file improves retrieval quality using multi-step AI workflows.
#
# The retrieval quality problem:
# The quality of a RAG answer depends heavily on what gets retrieved.
# And what gets retrieved depends on how similar the query embedding is
# to the document embeddings. If the user's query is vague or uses
# different vocabulary than the documents, retrieval suffers.
#
# Two solutions:
#
# 1. Query rewriting: Use an LLM to rewrite the user's question into a
#    version that will produce a better embedding for semantic search.
#    "tell me about that database thing" → "How do relational databases
#    store and query structured data using SQL?"
#
# 2. Query decomposition: Some questions are actually multiple questions.
#    Split them up and retrieve separately, then combine the results.
#    This is called "multi-hop retrieval."

from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL
from embeddings import embed_text
from vector_store import query_similar

_client = genai.Client(api_key=GEMINI_API_KEY)



def rewrite_query(original_query, conversation_context=""):
    """
    Use Gemini to rewrite the user's query for better semantic search.

    Args:
        original_query:      The user's original question.
        conversation_context: Recent conversation history (helps resolve
                              pronouns like "it" or "that").

    Returns:
        A rewritten query string, or the original if rewriting fails.
    """
    try:
        context_section = ""
        if conversation_context:
            context_section = f"""
Recent conversation context:
{conversation_context}

Use this context to resolve references such as "it", "that", or "they".
"""

        prompt = f"""
Rewrite the user's question into a clear, specific, technical query
suitable for semantic search in a knowledge base.

Keep the original meaning. If the question refers to something from the
conversation, use the conversation context to make the reference explicit.

Do not answer the question. Only return the rewritten query.

{context_section}

User's original question:
{original_query}
"""

        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        )

        rewritten_query = response.text.strip()

        if rewritten_query and len(rewritten_query) < 500:
            return rewritten_query

    except Exception:
        pass

    return original_query




def decompose_query(query):
    """
    Break a complex multi-part question into simpler sub-questions.

    Args:
        query: A question that may contain multiple distinct topics.

    Returns:
        A list of sub-question strings (up to 3), or [query] if it's
        already simple or if decomposition fails.
    """
    try:
        prompt = f"""
Determine whether the following question contains multiple distinct
questions or topics.

If it does, split it into 2-3 simpler, self-contained questions.
If it does not, return the original question unchanged.

Each question must be on its own line.
Do not number the questions.
Do not provide explanations or answers.

Question:
{query}
"""

        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        )

        sub_queries = [
            line.strip()
            for line in response.text.strip().splitlines()
            if line.strip() and len(line.strip()) > 5
        ]

        if sub_queries:
            return sub_queries[:3]

    except Exception:
        pass

    return [query]


    # TODO (Week 15): Implement query decomposition using Gemini.
    #
    # --- The RAG concept ---
    # Some questions have multiple parts, each requiring different documents.
    # "How does Python connect to databases, and what's the difference between
    # SQL and NoSQL?" needs documents about Python AND about SQL/NoSQL separately.
    # By splitting the question and searching for each part independently,
    # we get much better document coverage for complex questions.
    #
    # Steps:
    #   1. Build a prompt asking Gemini: if this question covers multiple topics,
    #      split it into 2-3 simpler sub-questions; otherwise return it as-is
    #   2. Call _client.models.generate_content() with temperature=0.1
    #   3. Split response.text on newlines, strip each line, drop empty/short lines
    #   4. Return at most 3 sub-questions
    #   5. Wrap in try/except — if anything fails, return [query]
    #
    return [query]  # placeholder — query is not decomposed


def multi_hop_retrieve(query, n_per_hop=2):
    """
    Retrieve documents for each sub-question and combine the results.

    Steps:
      1. Decompose the query into sub-questions
      2. Embed and search for each sub-question independently
      3. Combine results, removing duplicates

    Args:
        query:     The original complex query.
        n_per_hop: Documents to retrieve per sub-question.

    Returns:
        A deduplicated list of relevant document strings.
    """
    sub_queries = decompose_query(query)

    all_documents = []
    seen_documents = set()

    for sub_query in sub_queries:
        embedding = embed_text(sub_query)
        results = query_similar(embedding, n_per_hop)

        for doc in results["documents"][0]:
            if doc not in seen_documents:
                seen_documents.add(doc)
                all_documents.append(doc)

    return all_documents
