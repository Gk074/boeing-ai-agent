import os
from dotenv import load_dotenv
import cohere

# Force load .env from project root
load_dotenv()

api_key = os.getenv("CO_API_KEY")

if not api_key:
    raise RuntimeError("CO_API_KEY not found. Add it to your .env file.")

_co = cohere.Client(api_key=api_key)

def embed_texts(texts: list[str]) -> list[list[float]]:
    # For codebases, embed-english-v3.0 is fine; if you want multilingual use embed-multilingual-v3.0
    resp = _co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
    )
    return resp.embeddings
