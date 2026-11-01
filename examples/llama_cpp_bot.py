"""Elastimem with llama-cpp-python, sharing one chat model with the host.

Requires:  pip install llama-cpp-python
Models:    a chat GGUF (CHAT_MODEL) and optionally a small embedding GGUF
           (EMBED_MODEL, e.g. all-MiniLM-L6-v2.Q8_0.gguf, ~25 MB).

Shows the parts a local host needs: ``pause()`` so background extraction
never runs while the host generates, ``keep_turns`` to trim the host's own
message list, and ``report_pressure()`` after a failed generation.
"""

import os

import elastimem

CHAT_MODEL = os.environ.get("CHAT_MODEL", "models/chat.gguf")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "")   # optional
N_CTX = 4096
PERSONA = "You are a helpful local assistant. Be concise."


def load_models():
    from llama_cpp import Llama

    chat = Llama(model_path=CHAT_MODEL, n_ctx=N_CTX, n_gpu_layers=-1, verbose=False)

    def llm(prompt, *, max_tokens, temperature):
        out = chat.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature)
        return out["choices"][0]["message"]["content"]

    embedder = None
    if EMBED_MODEL and os.path.exists(EMBED_MODEL):
        # A separate small CPU instance, so embedding never waits on the chat model.
        embed_model = Llama(model_path=EMBED_MODEL, embedding=True, n_ctx=512,
                            n_gpu_layers=0, verbose=False)

        def embedder(texts):
            return [embed_model.create_embedding(t)["data"][0]["embedding"] for t in texts]

    return chat, llm, embedder


def main() -> None:
    chat, llm, embedder = load_models()
    mem = elastimem.open(
        "./llama_bot_memory.db",
        llm=llm,
        embedder=embedder or "auto",
        context_tokens=N_CTX,
        reserved_tokens=512 + len(PERSONA) // 4,     # reply + persona
        blocked={"model", "agent_name"},
    )
    print(f"[tier {mem.status.tier.name}: {mem.status.reasons}]")

    history: list[dict] = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in ("quit", "exit"):
            break

        context = mem.context(user)
        history = history[-2 * context.keep_turns:]     # older turns are in context.summary
        messages = ([{"role": "system", "content": f"{PERSONA}\n\n{context}"}]
                    + history + [{"role": "user", "content": user}])
        try:
            with mem.pause():                            # one model: background work waits
                out = chat.create_chat_completion(messages=messages, max_tokens=512)
            reply = out["choices"][0]["message"]["content"]
        except RuntimeError:
            mem.report_pressure()
            reply = "(model error; memory stepped down a tier, try again)"

        print(f"bot> {reply}")
        history += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
        mem.record(user, reply)

    mem.close()


if __name__ == "__main__":
    main()
