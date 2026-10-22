"""Evaluation for Wakeel AI: how well does retrieval find the right judgment?

Reads the labelled questions in eval_queries.jsonl and reports, for each retrieval
setting, Recall@10 and MRR at the judgment level:

    dense    semantic search only (bge-large + ChromaDB), HYBRID_ALPHA = 1.0
    keyword  BM25 only
    hybrid   the weighted mix the app actually uses (HYBRID_ALPHA)

    .venv/Scripts/python.exe evaluate_retrieval.py                  # retrieval only, no LLM calls
    .venv/Scripts/python.exe evaluate_retrieval.py --reviewed-only  # only questions marked reviewed
    .venv/Scripts/python.exe evaluate_retrieval.py --answers 5      # also check citations in 5 real answers
    .venv/Scripts/python.exe evaluate_retrieval.py --out eval_results.md

Retrieval needs no API key and uses no quota. --answers N sends N questions through the
full agent, about two LLM requests each, so keep N small on the Gemini free tier.

A question counts a judgment as found if any of its chunks is in the top 10 chunks.
Judgments are ranked by where their first chunk appears, so MRR is 1 / that rank.
"""
import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, List

import wakeel_ai as W

QUERIES_FILE = Path("eval_queries.jsonl")
K = W.TOP_K_CHUNKS  # 10


def load_queries(path: Path, reviewed_only: bool) -> List[Dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                if not reviewed_only or q.get("reviewed"):
                    queries.append(q)
    return queries


def ranked_docs(doc_ids: List[str]) -> List[str]:
    """Collapse a ranked chunk list into a ranked judgment list (first appearance wins)."""
    seen = []
    for d in doc_ids:
        if d not in seen:
            seen.append(d)
    return seen


def retrieve(setting: str, query: str, bm25: W.BM25Index) -> List[str]:
    if setting == "keyword":
        hits = bm25.search(query, top_k=K)
        return ranked_docs([bm25.chunks_data[i]["doc_id"] for i, _ in hits])
    alpha = 1.0 if setting == "dense" else W.HYBRID_ALPHA
    hits = W.search_hybrid(query, top_k=K, alpha=alpha)
    return ranked_docs([h["meta"]["doc_id"] for h in hits])


def score(relevant: List[str], docs: List[str]) -> Dict:
    found = [d for d in relevant if d in docs]
    first = next((i for i, d in enumerate(docs, 1) if d in relevant), None)
    return {
        "recall": len(found) / len(relevant),
        "rr": 1.0 / first if first else 0.0,
        "rank": first,
    }


def run_retrieval(queries: List[Dict], settings: List[str]) -> Dict:
    bm25 = W.BM25Index(W.BM25_DIR)
    if not bm25.load():
        sys.exit("No BM25 index found. Index the PDFs first.")
    results = {s: [] for s in settings}
    for n, q in enumerate(queries, 1):
        print(f"  [{n}/{len(queries)}] {q['id']}", end="\r", flush=True)
        for s in settings:
            docs = retrieve(s, q["query"], bm25)
            results[s].append({"id": q["id"], "type": q["type"], "top": docs[:3],
                               **score(q["relevant"], docs)})
    print(" " * 40, end="\r")
    return results


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def retrieval_table(results: Dict, types: List[str]) -> str:
    rows = ["| Setting | Questions | Recall@10 | MRR |", "|---|---|---|---|"]
    for s, rs in results.items():
        rows.append(f"| {s} | {len(rs)} | {mean([r['recall'] for r in rs]):.1%} | "
                    f"{mean([r['rr'] for r in rs]):.3f} |")
    out = "\n".join(rows)

    # The same numbers split by question type: identity questions (case numbers, party
    # names) should favour keyword search, topic questions should favour dense.
    rows = ["| Type | n | " + " | ".join(f"{s} R@10 / MRR" for s in results) + " |",
            "|---|---|" + "---|" * len(results)]
    for t in types:
        cells = []
        for rs in results.values():
            sub = [r for r in rs if r["type"] == t]
            cells.append(f"{mean([r['recall'] for r in sub]):.0%} / {mean([r['rr'] for r in sub]):.2f}")
        n = sum(1 for r in next(iter(results.values())) if r["type"] == t)
        rows.append(f"| {t} | {n} | " + " | ".join(cells) + " |")
    return out + "\n\n" + "\n".join(rows)


def misses(results: Dict, queries: List[Dict]) -> str:
    """Questions the app's own setting got wrong, so they can be looked at one by one."""
    by_id = {q["id"]: q for q in queries}
    lines = []
    for r in results.get("hybrid", []):
        if r["recall"] < 1.0:
            q = by_id[r["id"]]
            lines.append(f"- {r['id']} ({q['type']}): {q['query']}\n"
                         f"  expected {', '.join(q['relevant'])}; got {', '.join(r['top']) or 'nothing'}")
    return "\n".join(lines) or "- none"


CITE_RE = re.compile(r"\[S(\d+)\]")


def run_answers(queries: List[Dict], n: int) -> str:
    """Send n questions through the full agent and check the answers cite real sources."""
    from ui import build_agent
    agent = build_agent()
    rows = []
    for q in queries[:n]:
        r = agent.run_detailed(q["query"], thread_id=f"eval-{uuid.uuid4()}", mode="lawyer")
        if r["route"] == "ERROR":
            # Almost always the free-tier quota; further calls would fail the same way.
            print(f"  stopped at {q['id']}: {r['answer'][:120]}")
            break
        source_ns = {s["n"] for s in r["sources"]}
        cited = {int(c) for c in CITE_RE.findall(r["answer"])}
        rows.append({
            "id": q["id"],
            "not_found": W.NOT_FOUND_PHRASE.lower() in r["answer"].lower(),
            "cited": bool(cited),
            "verifiable": bool(cited) and cited <= source_ns,
            "relevant_cited": any(s["doc_id"] in q["relevant"] for s in r["sources"]
                                  if s["n"] in cited),
        })
    if not rows:
        return "No answers completed."
    k = len(rows)
    answered = [r for r in rows if not r["not_found"]]
    return "\n".join([
        f"| Answers checked | {k} |", "|---|---|",
        f"| Answered (not 'not found') | {len(answered)}/{k} |",
        f"| Answers with a verifiable citation | {sum(r['verifiable'] for r in answered)}/{len(answered)} |",
        f"| ...citing a judgment labelled relevant | {sum(r['relevant_cited'] for r in answered)}/{len(answered)} |",
        f"| 'Not found in the indexed judgments' | {k - len(answered)}/{k} |",
    ])


def main():
    ap = argparse.ArgumentParser(description="Evaluate Wakeel AI retrieval")
    ap.add_argument("--queries", type=Path, default=QUERIES_FILE)
    ap.add_argument("--reviewed-only", action="store_true",
                    help="use only questions marked \"reviewed\": true")
    ap.add_argument("--answers", type=int, default=0, metavar="N",
                    help="also run N questions through the agent (uses ~2N LLM requests)")
    ap.add_argument("--out", type=Path, help="also write the report to this Markdown file")
    args = ap.parse_args()

    queries = load_queries(args.queries, args.reviewed_only)
    if not queries:
        sys.exit("No questions to evaluate.")
    reviewed = sum(1 for q in queries if q.get("reviewed"))
    types = sorted({q["type"] for q in queries})

    print(f"Evaluating {len(queries)} questions ({reviewed} reviewed)...")
    results = run_retrieval(queries, ["dense", "keyword", "hybrid"])

    report = [
        "# Wakeel AI retrieval evaluation", "",
        f"{len(queries)} labelled questions ({reviewed} reviewed), "
        f"top {K} chunks, hybrid alpha = {W.HYBRID_ALPHA}.", "",
        retrieval_table(results, types), "",
        "## Hybrid misses", "", misses(results, queries),
    ]
    if args.answers:
        print(f"Checking {args.answers} answers through the agent...")
        report += ["", "## Answer citations (lawyer mode)", "", run_answers(queries, args.answers)]

    text = "\n".join(report)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
