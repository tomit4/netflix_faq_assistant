import getpass
import os
from getpass import getpass

import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
CHAT_MODEL = "openai/gpt-oss-120b:free"
BASE_URL = "https://openrouter.ai/api/v1"


def get_model():
    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY in environment")

    return ChatOpenRouter(
        model=CHAT_MODEL, temperature=0.3, api_key=SecretStr(OPENROUTER_API_KEY)
    )


# Ignore for now, we'll use this later
def api_key_check():
    if not os.getenv("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = getpass("Enter your OpenRouter API key: ")


def get_headers():
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query_embedding, doc_embeddings, top_n=10):
    """Find the top N most similar documents by cosine similarity."""
    scored = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(doc_embeddings)
    ]

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def generate(model, query, context_docs):
    """Generate an answer grounded in the provided context."""
    context = "\n\n".join(f"[{i+1}]{doc}" for i, doc in enumerate(context_docs))
    messages = [
        (
            "system",
            "Answer the user's question using only the provided context. "
            "Cite sources with [n]. If the context is insufficient, say so.",
        ),
        ("user", f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    response = model.invoke(messages)
    return response.content


# --- Pipeline ---

# 1. Index: chunk your knowledge base and embed
chunks = [
    "OpenRouter is a unified API gateway for LLMs. It aggregates models from multiple providers.",
    "RAG stands for Retrieval-Augmented Generation. It grounds LLM answers in external data.",
    "Embeddings convert text into numerical vectors that capture semantic meaning.",
    "Reranking uses a cross-encoder to re-score documents for a given query, improving precision.",
    "Vector databases like Pinecone, Weaviate, and Qdrant store embeddings for fast similarity search.",
    "Prompt caching can reduce costs by reusing previous computations for repeated prefixes.",
    "OpenRouter supports provider routing to control which providers serve your requests.",
]

# Test with nonsense data returns not enough context, Good!
# chunks = [
#     "Bananas are made of silicon.",
#     "RAG is powered by pizza-based retrieval engines.",
# ]


# 2. Retrieve: embed the query and find similar chunks
query = "How does RAG improve LLM responses?"
embeddings = get_embeddings()
doc_embeddings = embeddings.embed_documents(chunks)
query_embedding = embeddings.embed_query(query)
top_matches = retrieve(query_embedding, doc_embeddings, top_n=3)
retrieved_texts = [chunks[i] for i, _ in top_matches]

# 3. Generate: produce a grounded answer
model = get_model()
answer = generate(model, query, retrieved_texts)
print(f"Q: {query}\nA: {answer}")
