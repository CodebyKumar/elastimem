"""Elastimem with any OpenAI-compatible endpoint (OpenAI, Groq, Together, a
local vLLM or llama.cpp server) through a plain HTTP POST.

Requires:  pip install requests
Env:       OPENAI_API_KEY (or GROQ_API_KEY), OPENAI_BASE_URL, OPENAI_MODEL.
           Defaults target Groq's free tier; any compatible endpoint works.

The model is remote, so it never competes with Elastimem's background work
and no ``pause()`` is needed. No ``embedder=`` is passed: the built-in one
is used when ``elastimem[embed]`` is installed (first use downloads a small
model); pass ``embedder=None`` for keyword search only.
"""

import os

import elastimem

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.environ.get("OPENAI_MODEL", "llama-3.1-8b-instant")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")


def make_llm(api_key: str, base_url: str, model: str):
    import requests

    def llm(prompt: str, *, max_tokens: int, temperature: float) -> str:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    return llm


def main() -> None:
    if not API_KEY:
        print("Set OPENAI_API_KEY (or GROQ_API_KEY) to run this example.")
        return

    llm = make_llm(API_KEY, BASE_URL, MODEL)
    with elastimem.open("./openai_agent_memory.db", llm=llm) as mem:
        print(f"[tier {mem.status.tier.name}, model {MODEL}]")
        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user or user.lower() in ("quit", "exit"):
                break
            reply = llm(f"{mem.context(user)}\n\nUser: {user}\nAssistant:",
                        max_tokens=200, temperature=0.3)
            print(f"bot> {reply}")
            mem.record(user, reply)


if __name__ == "__main__":
    main()
