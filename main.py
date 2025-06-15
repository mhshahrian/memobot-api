from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI client (v1+)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Persistent ChromaDB client with DuckDB+Parquet
db_client = chromadb.PersistentClient(path="chroma_db")

collection = db_client.get_or_create_collection(
    name="messages",
    embedding_function=OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class Message(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    memory_used: bool    
    related_memories: list[str]

@app.post("/chat")
async def chat(msg: Message):
    emb = collection._embedding_function([msg.message])[0]

    # Query before adding new message to avoid self-recall
    results = collection.query(query_embeddings=[emb], n_results=10)
    docs, dists = results["documents"][0], results["distances"][0]
    relevant = [d for d, dist in zip(docs, dists) if 1 - dist >= 0.7 and d.strip() != msg.message.strip()]
    memory_used = len(relevant) > 0
    context = "\n".join(relevant)

    # Add new message after querying
    uid = f"{msg.user_id}_{len(collection.get()['ids'])}"
    collection.add(documents=[msg.message], embeddings=[emb], ids=[uid])

    chat_messages = []
    if context:
        chat_messages.append({"role": "system", "content": f"Relevant memory:\n{context}"})
    else:
        chat_messages.append({"role": "system", "content": "You're asking about your previous messages. Unfortunately, no relevant memory was found based on similarity. Please try rephrasing or being more specific."})
    chat_messages.append({"role": "user", "content": msg.message})

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=chat_messages
    )
    return ChatResponse(
        response=resp.choices[0].message.content,
        memory_used=memory_used,
        related_memories=relevant
    )
