import os
from openai import OpenAI
from openai.types.chat import ChatCompletion

# Get and validate API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not set in environment.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def generate_response(user_input: str, context: list[str] = []) -> str:
    """
    Generates a response from the OpenAI chat model using optional context.
    
    Parameters:
        user_input (str): The latest message from the user.
        context (list[str], optional): List of previous user inputs for context.
        
    Returns:
        str: The assistant's response.
    """
    system_prompt = "You are a helpful, concise, and safe assistant."

    # Build the conversation history with role-based messages
    messages = [{"role": "system", "content": system_prompt}]
    
    for ctx in context:
        messages.append({"role": "user", "content": ctx})
    
    messages.append({"role": "user", "content": user_input})

    try:
        response: ChatCompletion = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=messages,
            temperature=0.7,
            max_tokens=50,
            n=1
        )
        if response.choices:
            return response.choices[0].message.content.strip()
        else:
            return "No response generated."
    except Exception as e:
        return f"Error generating response: {e}"
