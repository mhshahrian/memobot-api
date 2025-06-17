from fastapi import FastAPI
from models import ChatRequest, ChatResponse
from memory import store_message
from decision import get_context_if_needed
from ai_agent import generate_response

app = FastAPI()

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    context = get_context_if_needed(req.user_id, req.message)
    store_message(req.user_id, req.message)
    memory_used = len(context) > 0
    reply = generate_response(req.message, context)

    return ChatResponse(reply=reply, memory_used=memory_used, related_memories=context)