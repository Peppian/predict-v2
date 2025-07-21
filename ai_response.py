import requests
import os
from dotenv import load_dotenv

# Muat API key dari file .env
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

def ask_openrouter(prompt: str, model: str = "deepseek/deepseek-r1:free") -> str:
    if not API_KEY:
        return "[ERROR] API key tidak ditemukan. Pastikan OPENROUTER_API_KEY sudah di-set di .env."

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.RequestException as e:
        return f"[ERROR] Koneksi gagal: {str(e)}"
    except KeyError:
        return "[ERROR] Format respons API tidak sesuai. Cek JSON dari server."
