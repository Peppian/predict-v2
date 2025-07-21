import requests
import streamlit as st

def ask_openrouter(prompt: str) -> str:
    api_key = st.secrets["openrouter"]["api_key"]
    model = st.secrets["openrouter"]["model"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    if response.status_code != 200:
        print(f"[OpenRouter ERROR {response.status_code}] {response.text}")
        return "❌ Gagal mendapatkan respons dari LLM."

    return response.json()["choices"][0]["message"]["content"]
