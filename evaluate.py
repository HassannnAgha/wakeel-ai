"""Evaluate the live agent against the indexed corpus.

Measures three things that matter for a legal research tool:

  1. Routing accuracy - does the classifier pick the right branch?
  2. Grounding        - does the answer cite case numbers and citations that
                        actually exist in the corpus, or invent them?
  3. Latency          - median and worst case seconds per query.

Needs an API key for the configured provider (GOOGLE_API_KEY, or OPENAI_API_KEY with
WAKEEL_PROVIDER=openai) and an indexed corpus.

    .venv/bin/python evaluate.py [--limit N] [--model gemini-3.5-flash]

Each query is a real LLM call. The Gemini free tier allows roughly 20 per day,
so start with --limit 5.
"""
import argparse
import json
import re
import statistics
import time
from collections import Counter

import wakeel_ai as W

# (query, expected route). Expected routes are mechanical - which branch should
# handle the query - not judgements about legal correctness.
CASES = [
    ("Find bail cases under Section 497 CrPC", "GQ"),
    ("Which judgments cite 2016 SCMR 2073?", "GQ"),
    ("What were the penalties in the last corruption case?", "GQ"),
    ("Show me judgments about thumb impression comparison", "GQ"),
    ("Which cases discuss Section 302 PPC?", "GQ"),
    ("What does the court say about adjudication on merits?", "GQ"),
    ("Find writ petitions about civil judge orders", "GQ"),
    ("Summarize the Lahore High Court judgment on maintenance", "SQ"),
    ("Summarize the writ petition about thumb impression comparison", "SQ"),
    ("Give me a plain-language summary of a bail judgment", "SQ"),
    ("Compare the 2023 writ petition with the criminal appeal", "CQ"),
    ("Compare how two judgments treat procedural technicalities", "CQ"),
    ("What is the weather in Lahore?", "REJECT"),
    ("Write me a poem about mangoes", "REJECT"),
    ("Who won the cricket match yesterday?", "REJECT"),
]

CASE_NO_RE = re.compile(r'\b\d{1,5}\s*/\s*(?:19|20)\d{2}\b')
CITATION_RE = re.compile(r'\b(?:\d{4}\s+SCMR\s+\d+|PLD\s+\d{4}\s+[A-Z]+\s+\d+|'
                         r'\d{4}\s+[A-Z]{2,5}\s+\d+)\b')


def corpus_strings():
    """Every case number and citation present in the indexed chunks."""
    numbers, citations = set(), set()
    for meta_file in W.METADATA_DIR.glob("*.json"):
        for chunk in json.loads(meta_file.read_text(encoding="utf-8")):
            if chunk.get("case_no"):
                numbers.add(chunk["case_no"].replace(" ", ""))
            for c in chunk.get("citations", []):
                citations.add(re.sub(r"\s+", " ", c).strip().lower())
            for m in CASE_NO_RE.findall(chunk.get("raw_text", "")):
                numbers.add(m.replace(" ", ""))
            for m in CITATION_RE.findall(chunk.get("raw_text", "")):
                citations.add(re.sub(r"\s+", " ", m).strip().lower())
    return numbers, citations


def grounding(answer, query, numbers, citations):
    """Case numbers / citations in the answer that are not in the corpus.

    Anything the user supplied in the query is excluded: echoing back a citation
    the user asked about is not a hallucination.
    """
    def found(rx, text):
        return {re.sub(r"\s+", " ", m).strip().lower() for m in rx.findall(text)}

    from_query = found(CASE_NO_RE, query) | found(CITATION_RE, query)
    claimed = found(CASE_NO_RE, answer) | found(CITATION_RE, answer)
    claimed -= from_query

    known = {n.lower() for n in numbers} | citations
    bad = sorted(c for c in claimed if c.replace(" ", "") not in
                 {k.replace(" ", "") for k in known})
    return len(claimed), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(CASES))
    ap.add_argument("--model", default=W.LLM_MODEL)
    args = ap.parse_args()

    W.require_api_key()
    llm = W.make_llm(temperature=0.3, model=args.model, max_retries=3)
    agent = W.WakeelAgent(llm)

    numbers, citations = corpus_strings()
    print(f"Corpus: {len(numbers)} case numbers, {len(citations)} citations indexed")
    print(f"Model : {args.model}\n")

    selected = CASES[:args.limit]
    times, routed_ok, errors = [], 0, 0
    claimed_total = ungrounded_total = 0
    routes = Counter()
    rows = []

    print(f"{'#':>5}  {'query':<48}{'route':>8}{'secs':>7}  note")
    print("-" * 100)

    for i, (query, expected) in enumerate(selected, 1):
        # Printed before the call so a slow or hanging query is visible.
        print(f"{i:>2}/{len(selected)}  {query[:46]:<48}", end="", flush=True)
        t = time.time()
        try:
            res = agent.run_detailed(query, thread_id=f"eval-{abs(hash(query))}")
        except Exception as e:
            errors += 1
            print(f"{'EXC':>8}{time.time()-t:>6.1f}s  {type(e).__name__}: {str(e)[:40]}")
            rows.append((query, expected, "EXC", 0.0))
            continue
        elapsed = time.time() - t
        route, answer = res["route"], res["answer"]

        # An API failure is not a routing result. Reporting accuracy over these
        # would be meaningless, so stop rather than produce a misleading number.
        if route == "ERROR":
            print(f"{'ERROR':>8}{elapsed:>6.1f}s")
            print(f"\n  {answer}\n")
            print("Aborting: the model is not answering, so no accuracy can be measured.")
            return

        times.append(elapsed)
        routes[route] += 1
        hit = route == expected
        routed_ok += hit

        n_claimed, bad = grounding(answer, query, numbers, citations)
        claimed_total += n_claimed
        ungrounded_total += len(bad)

        note = "" if hit else f"want {expected}"
        if bad:
            note += (" | " if note else "") + f"UNGROUNDED: {', '.join(bad[:3])}"
        if not res.get("grounded", True):
            note += (" | " if note else "") + "not in corpus"
        print(f"{route:>8}{elapsed:>6.1f}s  {note}")
        rows.append((query, expected, route, elapsed))

    n = len(rows)
    print("\n" + "=" * 60)
    print(f"routing accuracy   : {routed_ok}/{n}  ({100.0*routed_ok/n:.0f}%)" if n else "no rows")
    if times:
        print(f"latency median/max : {statistics.median(times):.1f}s / {max(times):.1f}s")
    print(f"errors             : {errors}")
    print(f"citations claimed  : {claimed_total}")
    print(f"  ungrounded       : {ungrounded_total}"
          + (f"  ({100.0*ungrounded_total/claimed_total:.0f}% hallucinated)"
             if claimed_total else "  (none claimed)"))
    print(f"route distribution : {dict(routes)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
