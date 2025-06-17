import os
from openai import OpenAI
from openai.types.chat import ChatCompletion

# Get and validate API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not set in environment.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def generate_response(user_input: str, context: list[str] = [], memory: list[str] = []) -> str:
    """
    Generates a response from the OpenAI chat model using optional context and memory.
    """
    system_prompt = (
        "You are a helpful and context-aware assistant. "
        "You remember previous conversations when relevant. "
        "If memory has been retrieved, use it thoughtfully and explain how it connects. "
        "If no memory is retrieved, answer normally based only on the latest user message."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if memory:
        memory_block = "\n".join(f"- {m}" for m in memory)
        messages.append({
            "role": "system",
            "content": f"The following relevant memories were retrieved and may be useful:\n{memory_block}"
        })

    for ctx in context:
        messages.append({"role": "user", "content": ctx})

    messages.append({"role": "user", "content": user_input})

    try:
        response: ChatCompletion = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.4,
            max_tokens=100,
            n=1
        )
        if response.choices:
            return response.choices[0].message.content.strip()
        else:
            return "No response generated."
    except Exception as e:
        return f"Error generating response: {e}"


def should_use_memory_gpt(user_message: str) -> bool:
    system_prompt = "You're an assistant deciding whether memory is needed to respond to a message. Answer 'YES' or 'NO'."
    user_prompt = f"User message: \"{user_message}\". Is memory required to reply helpfully?"

    response: ChatCompletion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = response.choices[0].message.content.strip().upper()
    return "YES" in answer
