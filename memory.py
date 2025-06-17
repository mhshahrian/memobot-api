import uuid
from utils import get_embedding
from chromadb import Client

db = Client().create_collection("chat_memory")

def store_message(user_id, message):
    embedding = get_embedding(message)
    message_id = str(uuid.uuid4())  # Generate a unique ID for the message
    db.add(
        ids=[message_id],
        documents=[message],
        embeddings=[embedding],
        metadatas=[{"user_id": user_id}]
    )

def fetch_memories(user_id, query, top_k=10, return_embeddings=False):
    query_emb = get_embedding(query)
    
    results = db.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where={"user_id": user_id}
    )

    # Extract documents and/or embeddings based on the request
    documents = results.get("documents", [[]])[0] if query else []
    # Handle None case for embeddings
    embeddings_list = results.get("embeddings", [[]])
    embeddings = embeddings_list[0] if embeddings_list and return_embeddings else []

    if return_embeddings:
        return documents, embeddings
    return documents if query else embeddings