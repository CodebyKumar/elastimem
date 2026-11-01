"""Retrieval quality: does search() find the planted memory for a question?

Builds a store of ~20 sessions of distractor chat with 20 planted memories,
then asks one question per memory. Most questions are paraphrased (they
share few or no words with the memory) and measure meaning-based recall;
three ask for exact identifiers, where keyword search has to carry it. No LLM is involved: this is the LLM-free floor.

Metrics, over the chunk hits of search(question, k=10):
  recall@5  share of questions whose memory ranks in the top 5
  MRR       mean of 1/rank (0 when not found in the top 10)

Run:  uv run python benchmarks/eval_recall.py [keyword|builtin] [tier]
"""

import os
import random
import sys
import tempfile

import elastimem
from elastimem import Tier

# (session, user text, assistant reply, question, substring identifying the memory)
PLANTED = [
    (1, "I adopted a puppy last weekend, his name is Bruno", "Congratulations, that is exciting!",
     "how is my dog doing?", "bruno"),
    (2, "I moved to Berlin for a new job at a robotics startup", "Good luck with the move.",
     None, None),
    (15, "Actually I relocated again, I now live in Lisbon", "Lisbon is lovely.",
     "which city am I living in now?", "lisbon"),
    (3, "My sister Priya is getting married in March", "How wonderful for her.",
     "when is the wedding in my family?", "priya"),
    (4, "I'm allergic to peanuts, so be careful with recipes", "Noted, I will avoid them.",
     "what food should I avoid?", "peanuts"),
    (5, "I've been learning to play the cello for two months", "That is a beautiful instrument.",
     "what instrument am I practicing?", "cello"),
    (6, "My dentist appointment is on Friday at 3pm", "I hope it goes smoothly.",
     "when do I see the dentist", "dentist"),
    (7, "I'm training for the marathon in September", "Keep up the long runs.",
     "what race am I preparing for?", "marathon"),
    (8, "We use Kubernetes on AWS at work for deployment", "A common setup.",
     "what is our infrastructure stack?", "kubernetes"),
    (9, "My favorite book is The Left Hand of Darkness", "A classic of science fiction.",
     "which novel do I love most?", "left hand"),
    (10, "I can't drink coffee after 2pm or I don't sleep", "Caffeine lingers for hours.",
     "why do I have trouble sleeping?", "coffee"),
    (11, "My laptop is a 2021 MacBook Air with 8GB of RAM", "That should handle most tasks.",
     "what computer do I use?", "macbook"),
    (12, "I'm saving up to buy a house in two years", "A solid goal.",
     "what is my long term financial goal?", "house"),
    (13, "My daughter Mia starts kindergarten in the fall", "A big milestone.",
     "when does my child start school?", "kindergarten"),
    (14, "I quit sugar three weeks ago and feel great", "Well done.",
     "what change did I make to my diet?", "sugar"),
    (16, "Our team standup is every day at 9:30", "Short meetings help.",
     "what time is the daily meeting?", "standup"),
    (17, "The wifi password at the cabin is pinecone42", "Saved.",
     "cabin wifi password", "pinecone42"),
    # Exact identifiers: keyword search should carry these, not meaning.
    (18, "My flight confirmation code is QX7Z2P", "Have a good trip.",
     "QX7Z2P", "qx7z2p"),
    (19, "The staging server is at 10.4.2.19 behind the VPN", "Got it.",
     "which machine is 10.4.2.19", "10.4.2.19"),
    (20, "Ticket ENG-4821 tracks the login timeout bug", "I will keep it in mind.",
     "status of ENG-4821", "eng-4821"),
]

# Distractors share topics and words with the planted memories on purpose.
FILLER = [
    ("My friend's cat keeps knocking things off the table", "Cats do that."),
    ("Can you recommend a good movie for tonight?", "Try a recent thriller."),
    ("How do I reverse a list in Python?", "Use reversed() or slicing."),
    ("I visited Paris last year and loved the museums", "The Louvre is huge."),
    ("What is a good recipe for pasta with tomatoes?", "Garlic, basil and olive oil."),
    ("I watched the football match yesterday", "Was it a close game?"),
    ("Explain how a hash map works", "Keys are hashed into buckets."),
    ("The weather has been rainy all week", "Hopefully it clears up."),
    ("I'm thinking about learning guitar someday", "Start with a few chords."),
    ("My coworker is planning a trip to Rome", "Rome has great food."),
    ("What's the difference between TCP and UDP?", "TCP is reliable, UDP is not."),
    ("I need ideas for a birthday gift for my dad", "Maybe a book or a watch."),
    ("Our office got new chairs this week", "Comfort matters."),
    ("I went hiking in the mountains on Sunday", "Sounds refreshing."),
    ("How do I center a div in CSS?", "Use flexbox."),
    ("I'm reading a lot about history lately", "Any particular era?"),
    ("My neighbor's dog barks all night", "That must be tiring."),
    ("Should I learn Rust or Go?", "Depends on your goals."),
    ("I made pancakes for breakfast", "Yum."),
    ("What does a load balancer do?", "Spreads traffic across servers."),
]


def build(mode: str, tier: Tier):
    path = os.path.join(tempfile.mkdtemp(), "eval.db")
    kwargs = {"tier_override": tier} if _old_api() else {"tier": tier}
    if mode == "keyword":
        kwargs.update({"disable_builtin_embedder": True} if _old_api() else {"embedder": None})
    mem = elastimem.open(path, **kwargs)
    record = getattr(mem, "record", None) or mem.record_turn
    rng = random.Random(7)
    planted = {p[0]: p for p in PLANTED}
    for session in range(1, 21):
        if session > 1:
            (mem.new_session if hasattr(mem, "new_session") else mem.begin_session)()
        turns = rng.sample(FILLER, 8)
        if session in planted:
            turns.insert(rng.randrange(len(turns) + 1), planted[session][1:3])
        for user, reply in turns:
            record(user, reply)
    (getattr(mem, "flush", None) or mem.drain)(120)
    return mem


def _old_api() -> bool:
    return hasattr(elastimem.Elastimem, "record_turn")


def evaluate(mem) -> tuple[float, float, list]:
    search = getattr(mem, "search", None) or mem.recall
    found, rr, misses = 0, 0.0, []
    questions = [(q, e) for *_, q, e in PLANTED if q]
    for question, expect in questions:
        hits = [h for h in search(question, k=10) if h.kind != "fact"]
        rank = next((i + 1 for i, h in enumerate(hits) if expect in h.text.lower()), None)
        if rank is not None and rank <= 5:
            found += 1
        rr += 1.0 / rank if rank else 0.0
        if rank != 1:
            misses.append((question, rank))
    n = len(questions)
    return found / n, rr / n, misses


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "keyword"
    tier = Tier[sys.argv[2].upper()] if len(sys.argv) > 2 else Tier.FULL
    mem = build(mode, tier)
    recall5, mrr, misses = evaluate(mem)
    print(f"{elastimem.__version__} mode={mode} tier={tier.name}: "
          f"recall@5={recall5:.2f} MRR={mrr:.2f}")
    for question, rank in misses:
        print(f"  rank {rank or '-':>2}  {question}")
    mem.close()
