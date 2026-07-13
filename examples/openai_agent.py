"""Elastimem wired to any OpenAI-compatible chat completions endpoint
(OpenAI itself, Groq, Together, a local vLLM/llama.cpp server, etc.) via a
plain HTTP POST — no vendor SDK required, just requests.

Requires:  pip install requests
Env:       OPENAI_API_KEY (or GROQ_API_KEY etc.), OPENAI_BASE_URL,
           OPENAI_MODEL — defaults target Groq's endpoint since it's a free
           tier, but any OpenAI-compatible base_url/model works unchanged.

This is the same `llm=` shape as any other example — a plain
``(prompt, *, max_tokens, temperature) -> str`` callable — just backed by a
network call instead of an in-process model.

No `embedder=` is passed below, so Elastimem's built-in embedder
auto-activates (see docs/governor.md) once the `elastimem[embed]` extra is
installed — its first use downloads a small model from Hugging Face Hub in
the background. Pass `disable_builtin_embedder=True` to opt out and stay
FTS5-only.
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
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
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
    mem = elastimem.open("./openai_agent_memory.db", llm=llm)
    print(f"[tier: {mem.profile.tier.name}, model: {MODEL}]")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in ("quit", "exit"):
            break

        plan = mem.build_context(user)
        prompt = f"{plan.render()}\nUser: {user}\nAssistant:"
        reply = llm(prompt, max_tokens=200, temperature=0.3)

        print(f"bot> {reply}")
        mem.record_turn(user, reply)

    mem.end_session()
    mem.close()


if __name__ == "__main__":
    main()
