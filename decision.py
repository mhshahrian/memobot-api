from utils import get_embedding, cosine_similarity
from memory import fetch_memories
from ai_agent import should_use_memory_gpt

def load_keywords(file_path="keywords.txt"):
    with open(file_path, "r") as file:
        return {line.strip().lower() for line in file if line.strip()}

KEYWORDS = load_keywords()

def should_use_memory_keywords(user_input):
    # Check for keyword presence
    if any(keyword in user_input.lower() for keyword in KEYWORDS):
        return True


def should_use_memory_similarity(user_id, user_input, threshold=0.7):
    # Calculate embedding for the current input
    current_embedding = get_embedding(user_input)

    # Fetch past embeddings for the user
    _, past_embeddings = fetch_memories(user_id, query=user_input, return_embeddings=True)

    # Check for embedding similarity
    for past_embedding in past_embeddings:
        similarity = cosine_similarity(current_embedding, past_embedding)
        if similarity > threshold:
            return True

    # Default to not using memory
    return False


def get_context_if_needed(user_id, user_input):
    if should_use_memory_keywords(user_input):
        return fetch_memories(user_id, query=user_input)
    if should_use_memory_similarity(user_id, user_input):
        return fetch_memories(user_id, query=user_input)
    if should_use_memory_gpt(user_input):
        return fetch_memories(user_id, query=user_input)
    return []
