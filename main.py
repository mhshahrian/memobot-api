from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Message(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat(msg: Message):
    return {"response": f"You said: {msg.message}"}


