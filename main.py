import json
import os

import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
CHAT_MODEL = "openai/gpt-oss-120b:free"
BASE_URL = "https://openrouter.ai/api/v1"


def get_model():
    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY in environment")

    return ChatOpenRouter(
        model=CHAT_MODEL, temperature=0.3, api_key=SecretStr(OPENROUTER_API_KEY)
    )


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_data(docs):
    documents = []

    for file in docs:
        with open(file, "r") as f:
            data = json.loads(f.read())
            for qa_pair in data:
                if "?" in qa_pair:
                    question, answer = qa_pair.split("?", 1)
                    documents.append(
                        f"Question: {question.strip()}? Answer: {answer.strip()}"
                    )
                else:
                    documents.append(
                        f"Question: No question found. Answer: {qa_pair.strip()}"
                    )

    # TODO: add check if empty here
    return documents


def retrieve(query_embedding, doc_embeddings, top_n=10):
    """Find the top N most similar documents by cosine similarity."""
    scored = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(doc_embeddings)
    ]

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def generate_stream(model, query, context_docs):
    context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(context_docs))

    messages = [
        (
            "system",
            "Answer the user's question using only the provided context. "
            "Cite sources with [n]. If the context is insufficient, say so.",
        ),
        ("user", f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    try:
        print("\nBot: ", end="", flush=True)

        streamed = False
        for chunk in model.stream(messages):
            content = getattr(chunk, "content", None)
            if content:
                streamed = True
                print(content, end="", flush=True)

        if streamed:
            print("\n")
            return

        raise RuntimeError("No streamed content received")

    except Exception:
        response = model.invoke(messages)
        print(f"\nBot: {response.content}\n")


def main():
    model = get_model()
    embeddings = get_embeddings()

    faq_files = ["data.json"]
    documents = retrieve_data(faq_files)
    doc_embeddings = embeddings.embed_documents(documents)

    print("FAQ Chatbot ready. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            break

        query_embedding = embeddings.embed_query(query)
        top_matches = retrieve(query_embedding, doc_embeddings, top_n=3)
        retrieved_texts = [documents[i] for i, _ in top_matches]

        generate_stream(model, query, retrieved_texts)


if __name__ == "__main__":
    main()
