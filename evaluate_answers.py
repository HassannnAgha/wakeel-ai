"""Measure answer quality on the evaluation questions, through the running app.

    .venv/Scripts/python.exe ui.py --no-activity --port 8001      # in one terminal (any provider)
    .venv/Scripts/python.exe evaluate_answers.py --base http://127.0.0.1:8001 --n 30

Sends a spread of the questions in eval_queries.jsonl to /api/chat, saves each result in --out, then
reports, with no model-based grading:

  * answers written, "not found" answers, and errors;
  * answers whose [S#] citations all point at a passage that exists;
  * answers that cite a passage from a judgment the question was labelled with (the labels in
    eval_queries.jsonl are not yet reviewed by a lawyer, so read this as a rough guide);
  * the fact check (names, sections, numbers and quotations in a cited sentence must be in the
    cited passage), by running tests/factcheck_test.js over the saved answers, and every
    detail it flags, for you to read.

It costs one model call per question on whichever provider the app is using.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ask(base: str, query: str, mode: str, language: str) -> dict:
    body = {"message": query, "thread_id": f"eval-{uuid.uuid4()}", "mode": mode, "language": language}
    req = urllib.request.Request(base + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=400) as r:
        return json.loads(r.read())


def cited_numbers(answer: str):
    nums = set()
    for group in re.findall(r"\[(S\d+(?:\s*[,;]\s*S?\d+)*)\]", answer or ""):
        nums.update(int(n) for n in re.findall(r"\d+", group))
    return nums


def main():
    ap = argparse.ArgumentParser(description="Answer-quality evaluation through the running app")
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--n", type=int, default=30, help="how many questions (spread evenly over the 50)")
    ap.add_argument("--mode", default="lawyer", choices=["citizen", "lawyer"])
    ap.add_argument("--language", default="en", choices=["en", "ur", "roman_ur"])
    ap.add_argument("--out", default=str(ROOT / "logs" / "answer_eval"))
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / "eval_queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    n = max(1, min(args.n, len(rows)))
    picked = [rows[i * len(rows) // n] for i in range(n)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()

    stats = dict(written=0, not_found=0, errors=0, valid_cites=0, cites_relevant=0, labelled=0, cited_any=0)
    for row in picked:
        t = time.time()
        for attempt in range(1 + args.retries):
            try:
                r = ask(args.base, row["query"], args.mode, args.language)
            except Exception as e:
                r = {"route": "ERROR", "answer": f"{type(e).__name__}: {e}", "sources": []}
            if r.get("route") != "ERROR" or attempt == args.retries:
                break
            time.sleep(8)
        (out / f"{row['id']}.json").write_text(json.dumps({**r, "query": row["query"], "relevant": row["relevant"]},
                                                           ensure_ascii=False), encoding="utf-8")
        answer = r.get("answer") or ""
        sources = {s["n"]: s for s in r.get("sources", [])}
        nums = cited_numbers(answer)
        if r.get("route") == "ERROR":
            stats["errors"] += 1
            verdict = "ERROR"
        elif "not found in the indexed judgments" in answer.lower():
            stats["not_found"] += 1
            verdict = "not found"
        else:
            stats["written"] += 1
            valid = bool(nums) and nums <= set(sources)
            stats["valid_cites"] += valid
            stats["cited_any"] += bool(nums)
            hit = any(sources[k]["doc_id"] in row["relevant"] for k in nums if k in sources)
            stats["labelled"] += 1
            stats["cites_relevant"] += hit
            verdict = f"cites {len(nums)} passage(s), valid={valid}, from a labelled judgment={hit}"
        print(f"{row['id']} {time.time() - t:5.1f}s  {verdict}   {row['query'][:70]}")

    print("\nFact check (tests/factcheck_test.js over the saved answers):")
    node = shutil.which("node")
    flagged_lines, total = [], ""
    if node:
        res = subprocess.run([node, str(ROOT / "tests" / "factcheck_test.js"), str(out)], capture_output=True,
                             text=True, encoding="utf-8", cwd=ROOT)
        for line in res.stdout.splitlines():
            if line.startswith("total "):
                total = line
            m = re.match(r"(q\d+)\.json \d+ checked (\[.+\])$", line)
            if m and m.group(2) != "[]":
                flagged_lines.append(line)
    print("  " + (total or "Node.js not found: skipped"))
    for line in flagged_lines:
        print("  flagged:", line)

    w = max(1, stats["written"])
    print(f"\n{n} questions: {stats['written']} answered, {stats['not_found']} 'not found', {stats['errors']} errors")
    print(f"Answers whose every citation points at a real passage: {stats['valid_cites']}/{stats['written']}")
    print(f"Answers citing a passage from a judgment the question was labelled with: "
          f"{stats['cites_relevant']}/{stats['labelled']}  (labels unreviewed)")


if __name__ == "__main__":
    main()
