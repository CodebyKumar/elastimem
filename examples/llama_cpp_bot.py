"""Full-capability example: Elastimem wired to llama-cpp-python.

Requires:  pip install llama-cpp-python
Models:    a chat GGUF (CHAT_MODEL) and optionally a small embedding GGUF
           (EMBED_MODEL, e.g. all-MiniLM-L6-v2.Q8_0.gguf, ~25 MB).

Demonstrates every integration point from docs/integration.md: the
foreground gate, tick/report_pressure, budgeted context, rolling summary via
report_evictions, and end_session.
"""

import os

import elastimem

CHAT_MODEL = os.environ.get("CHAT_MODEL", "models/chat.gguf")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "")   # optional
N_CTX = 4096


def load_models():
    from llama_cpp import Llama

    chat = Llama(model_path=CHAT_MODEL, n_ctx=N_CTX, n_gpu_layers=-1,
                 verbose=False)

    def llm(prompt, *, max_tokens, temperature):
        out = chat.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature)
        return out["choices"][0]["message"]["content"]

    embedder = None
    if EMBED_MODEL and os.path.exists(EMBED_MODEL):
        # Separate tiny instance on CPU: embedding must never contend with
        # chat generation for the GPU or the chat model's lock.
        embed_model = Llama(model_path=EMBED_MODEL, embedding=True, n_ctx=512,
                            n_gpu_layers=0, verbose=False)

        def embedder(texts):
            return [embed_model.create_embedding(t)["data"][0]["embedding"]
                    for t in texts]

    return chat, llm, embedder


def main() -> None:
    chat, llm, embedder = load_models()
    base_prompt = "You are a helpful local assistant. Be concise."

    mem = elastimem.open(
        "./llama_bot_memory.db",
        llm=llm,
        embedder=embedder,
        context_tokens=N_CTX,
        static_prompt_tokens=len(base_prompt) // 4,
        reserved_keys={"model", "agent_name"},
    )
    print(f"[tier: {mem.profile.tier.name}, budgets: {mem.profile.budgets}]")

    history: list[dict] = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in ("quit", "exit"):
            break

        mem.tick()
        plan = mem.build_context(user)
        system = base_prompt + "\n\n" + plan.render()
        if plan.rolling_summary:
            system += f"\n\nEARLIER IN THIS CONVERSATION: {plan.rolling_summary}"

        # trim the live window to the plan, reporting evictions to Elastimem
        max_msgs = plan.keep_last_n_turns * 2
        if len(history) > max_msgs:
            evicted = history[:-max_msgs]
            history = history[-max_msgs:]
            pairs = [(evicted[i]["content"], evicted[i + 1]["content"])
                     for i in range(0, len(evicted) - 1, 2)]
            mem.report_evictions(pairs)

        messages = ([{"role": "system", "content": system}]
                    + history + [{"role": "user", "content": user}])
        try:
            with mem.foreground():                 # hold background LLM jobs
                out = chat.create_chat_completion(messages=messages,
                                                  max_tokens=512)
            reply = out["choices"][0]["message"]["content"]
        except RuntimeError:
            mem.report_pressure()                  # decode failure -> shed load
            reply = "(model error — memory tier lowered, try again)"

        print(f"bot> {reply}")
        history += [{"role": "user", "content": user},
                    {"role": "assistant", "content": reply}]
        mem.record_turn(user, reply)

    mem.end_session()
    mem.close()


if __name__ == "__main__":
    main()
