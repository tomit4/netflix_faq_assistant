import json
import os
import sys

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


def print_help():
    """
    Display usage instructions for the chatbot.
    """
    print("""
Available commands:
    - Type any question about Netflix to get an answer
    - help : Show this message
    - exit : Quit the chatbot

Notes:
    - Answers are based only on the provided FAQ dataset
    - If no relevant information is found, the bot will say so
""")


def get_model():
    """
    Initialize and return the LLM client using OpenRouter.

    Returns:
        ChatOpenRouter: Configured language model instance.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY in environment")

    return ChatOpenRouter(
        model=CHAT_MODEL, temperature=0.3, api_key=SecretStr(OPENROUTER_API_KEY)
    )


def get_embeddings():
    """
    Load the HuggingFace embedding model for text vectorization.

    Returns:
        HuggingFaceEmbeddings: Embedding model instance.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        float: Similarity score between -1 and 1.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_data(docs):
    """
    Load and preprocess FAQ data from JSON files.

    Each entry is converted into a structured "Question/Answer" string.

    Args:
        docs (list[str]): List of JSON file paths.

    Returns:
        list[str]: Processed document strings.
    """
    documents = []

    try:
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
    except FileNotFoundError:
        print("Error: dataset file not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: invalid JSON format in dataset")
        sys.exit(1)

    return documents


def retrieve(query_embedding, doc_embeddings, top_n=10):
    """
    Retrieve top-N most similar documents using cosine similarity.

    Args:
        query_embedding (np.ndarray): Embedding of the user query.
        doc_embeddings (list[np.ndarray]): Precomputed document embeddings.
        top_n (int): Number of results to return.

    Returns:
        list[tuple[int, float]]: Ranked list of (doc_index, similarity_score).
    """
    scored = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(doc_embeddings)
    ]

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def generate_stream(model, query, context_docs):
    """
    Generate a streamed LLM response using retrieved context.

    Args:
        model: LLM instance.
        query (str): User question.
        context_docs (list[str]): Retrieved relevant documents.
    """
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

    except Exception as e:
        print(f"\n[Some error occurred during streaming: {e}]")


def main():
    model = get_model()
    embeddings = get_embeddings()

    faq_files = ["data.json"]

    documents = retrieve_data(faq_files)
    if not documents:
        raise ValueError("No documents loaded from dataset.")

    doc_embeddings = embeddings.embed_documents(documents)

    print("Netflix FAQ Chatbot ready. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            print("\nExiting...")
            break

        if query.lower() in ["exit", "quit", "q"]:
            break

        if query.lower() == "help":
            print_help()
            continue

        query_embedding = embeddings.embed_query(query)
        top_matches = retrieve(query_embedding, doc_embeddings, top_n=3)
        retrieved_texts = [documents[i] for i, _ in top_matches]

        generate_stream(model, query, retrieved_texts)


if __name__ == "__main__":
    main()
    sys.exit()
