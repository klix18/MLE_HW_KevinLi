import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the .env file BEFORE calling os.getenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

client = OpenAI(api_key=api_key)


# store alternating user/assistant messages
conversation_history: list[dict] = []

def generate_response(user_text: str) -> str:
    global conversation_history

    # add latest user turn
    conversation_history.append({"role": "user", "content": user_text})

    # keep only last 5 turns (10 messages)
    if len(conversation_history) > 10:
        conversation_history[:] = conversation_history[-10:]

    messages = [{"role": "system", "content": "You are a concise, helpful assistant."}] + conversation_history

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=150,
    )
    reply_text = resp.choices[0].message.content.strip()

    # add assistant turn
    conversation_history.append({"role": "assistant", "content": reply_text})
    return reply_text

def stream_response(user_text: str):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_text})
    messages = [{"role":"system","content":"You are a concise, helpful assistant."}] + conversation_history
    with client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.7, stream=True) as stream:
        full = []
        for event in stream:
            delta = event.choices[0].delta.content or ""
            if delta:
                full.append(delta)
                yield delta  # stream to client
        reply = "".join(full).strip()
        conversation_history.append({"role":"assistant","content": reply})

# chat history ui
def get_history():
    # returns the full list, newest last
    return conversation_history
