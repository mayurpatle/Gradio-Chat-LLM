"""
app.py
-----------
A minimal chat UI built with Gradio whose backend simply forwards the
conversation to an LLM "chat completion" endpoint and streams the reply
back to the browser.

High-level flow:
    Browser (Gradio UI)  ->  respond()  ->  OpenAI-compatible API  ->  stream tokens back

Run it:
    pip install -r requirements.txt
    export OPENAI_API_KEY="sk-..."        # your key
    python chat_app.py                    # opens at http://127.0.0.1:7860
"""

import os
from openai import OpenAI  # The OpenAI SDK speaks the standard "chat completions" protocol.

# ---------------------------------------------------------------------------
# 1. Client setup
# ---------------------------------------------------------------------------
# The API key is read from an environment variable so it never lives in source
# control. `base_url` is left as the default (OpenAI), but you can point this at
# ANY OpenAI-compatible server — local Ollama, vLLM, Together, Groq, etc. — by
# setting OPENAI_BASE_URL. That's the only change needed to swap providers.

# Open source  Model    

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama', 
)

# Use 'llama3.2' as the model name
MODEL = 'qwen2.5'

# Open AI API   
# from dotenv import load_dotenv
# load_dotenv(override=True)

# api_key = os.getenv('OPENAI_API_KEY')



# if api_key and api_key.startswith('sk-proj-') and len(api_key)>10:

#     print("API key looks good so far")

# else:

#     print("There might be a problem with your API key? Please visit the troubleshooting notebook!")

    

# MODEL = 'gpt-5-nano'

# openai = OpenAI()


# A system prompt sets the assistant's persona/behavior for the whole session.
SYSTEM_PROMPT = "You are a concise, helpful assistant."


# ---------------------------------------------------------------------------
# 2. The backend function
# ---------------------------------------------------------------------------
# Gradio calls this every time the user sends a message.
#   - `message` is the newest user turn (a string).
#   - `history` is the prior conversation as a list of {"role", "content"} dicts,
#     because we set type="messages" on the ChatInterface below. This is the
#     same shape the chat completions API expects, so almost no transformation
#     is needed.
def respond(message, history):
    # Older Gradio passes history as [[user, assistant], ...] pairs,
    # so we unpack each pair back into the role/content shape the API needs.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
    )

    partial = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        partial += token
        yield partial


# ---------------------------------------------------------------------------
# 3. The Gradio interface
# ---------------------------------------------------------------------------
# gr.ChatInterface wires the `respond` function to a full chat UI for free:
# message box, send button, scrolling history, and the streaming display.
import gradio as gr

demo = gr.ChatInterface(
    fn=respond,                 # the backend function defined above
    # type="messages",            # use the OpenAI-style {"role","content"} history format
    title="Chat Completion Demo",
    description="A thin Gradio front-end over a chat completion API.",
    examples=[                  # clickable starter prompts shown on first load
        "Explain Kafka consumer groups in two sentences.",
        "Give me a one-line summary of CAP theorem.",
    ],
)

# ---------------------------------------------------------------------------
# 4. Launch
# ---------------------------------------------------------------------------
# `launch()` starts a local web server. Set share=True to get a temporary public
# URL, or server_name="0.0.0.0" to expose it on your LAN / inside a container.
if __name__ == "__main__":
    demo.launch()