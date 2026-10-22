"""A/B benchmark: legacy chunking vs. short-paragraph merging.

The original chunker discarded any accumulated text that had not reached
min_chunk_size (400 chars) before the next numbered paragraph began. This
measures what that costs on real court judgments.

    .venv/bin/python benchmark_chunking.py pdfs/

Needs no API key, no Chroma and no embedding model: it exercises the chunker
and a BM25 index only.
"""
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi

import wakeel_ai as W

PARA_RE = re.compile(r'^(\d+)\.\s+', re.M)


def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip().lower()


def load_documents(source: Path) -> Dict[str, str]:
    docs = {}
    for path in sorted(source.iterdir()):
        if path.suffix.lower() == ".txt":
            docs[path.stem] = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".pdf":
            import pdfplumber
            try:
                with pdfplumber.open(path) as pdf:
                    docs[path.stem] = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
            except Exception as e:
                print(f"  [skip] {path.name}: {e}")
    return {k: v for k, v in docs.items() if v.strip()}


def source_paragraphs(text: str) -> Dict[str, str]:
    paras, matches = {}, list(PARA_RE.finditer(text))
    for n, m in enumerate(matches):
        end = matches[n + 1].start() if n + 1 < len(matches) else len(text)
        num = m.group(1)
        if num not in paras:
            paras[num] = text[m.start():end].strip()
    return paras


def probe_for(paragraph: str, words: int = 12) -> str:
    """A distinctive phrase from the middle of a paragraph, used as a query."""
    toks = normalize(paragraph).split()
    if len(toks) < words + 6:
        return " ".join(toks[:words])
    start = len(toks) // 3
    return " ".join(toks[start:start + words])


def evaluate(docs: Dict[str, str], merge: bool) -> Dict:
    all_chunks: List[Dict] = []
    found = lost = total = 0
    lost_examples = []

    for doc_id, text in docs.items():
        chunks = W.chunk_text(text, {"doc_id": doc_id, "case_no": "", "title": ""},
                              merge_short_paragraphs=merge)
        all_chunks.extend(chunks)
        blob = normalize(" ".join(c["raw_text"] for c in chunks))

        paras = source_paragraphs(text)
        total += len(paras)
        for num, body in paras.items():
            if probe_for(body) in blob:
                found += 1
            else:
                lost += 1
                if len(lost_examples) < 5:
                    lost_examples.append(f"{doc_id} para {num}: {normalize(body)[:70]}...")

    sizes = [len(c["raw_text"]) for c in all_chunks] or [0]
    return {
        "chunks": len(all_chunks),
        "chars_kept": sum(sizes),
        "chars_source": sum(len(t) for t in docs.values()),
        "avg_size": statistics.mean(sizes),
        "min_size": min(sizes), "max_size": max(sizes),
        "citations": sum(len(c["citations"]) for c in all_chunks),
        "provisions": sum(len(c["provisions"]) for c in all_chunks),
        "paras_found": found, "paras_lost": lost, "paras_total": total,
        "lost_examples": lost_examples, "all_chunks": all_chunks,
    }


def retrieval_recall(docs: Dict[str, str], chunks: List[Dict], k: int = 5) -> Dict:
    """BM25 recall@k for a phrase taken from each source paragraph."""
    if not chunks:
        return {"recall": 0.0, "hits": 0, "total": 0, "unfindable": 0}

    corpus = [normalize(c["raw_text"]).split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    texts = [normalize(c["raw_text"]) for c in chunks]

    hits = total = unfindable = 0
    for doc_id, text in docs.items():
        for num, body in source_paragraphs(text).items():
            probe = probe_for(body)
            total += 1
            gold = {i for i, t in enumerate(texts) if probe in t}
            if not gold:
                unfindable += 1          # the paragraph is not in the index at all
                continue
            scores = bm25.get_scores(probe.split())
            top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            if gold & set(top):
                hits += 1
    return {"recall": hits / total if total else 0.0, "hits": hits,
            "total": total, "unfindable": unfindable}


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else W.PDF_DIR
    if not source.is_dir():
        raise SystemExit(f"No such directory: {source}")

    docs = load_documents(source)
    if not docs:
        raise SystemExit(f"No readable .pdf or .txt documents in {source}")

    print(f"\nCorpus: {len(docs)} documents from {source}")

    results = {}
    for label, merge in (("LEGACY", False), ("FIXED", True)):
        r = evaluate(docs, merge)
        r["retrieval"] = retrieval_recall(docs, r["all_chunks"])
        results[label] = r
    legacy, fixed = results["LEGACY"], results["FIXED"]

    print("\n" + "=" * 74)
    print(f"{'METRIC':<34}{'LEGACY':>18}{'FIXED':>18}")
    print("=" * 74)
    for name, a, b in [
        ("chunks produced", legacy["chunks"], fixed["chunks"]),
        ("characters kept", legacy["chars_kept"], fixed["chars_kept"]),
        ("source characters", legacy["chars_source"], fixed["chars_source"]),
        ("text retention", pct(legacy["chars_kept"], legacy["chars_source"]),
                           pct(fixed["chars_kept"], fixed["chars_source"])),
        ("paragraphs preserved", f"{legacy['paras_found']}/{legacy['paras_total']}",
                                 f"{fixed['paras_found']}/{fixed['paras_total']}"),
        ("paragraphs LOST", legacy["paras_lost"], fixed["paras_lost"]),
        ("citations captured", legacy["citations"], fixed["citations"]),
        ("provisions captured", legacy["provisions"], fixed["provisions"]),
        ("avg chunk size", f"{legacy['avg_size']:.0f}", f"{fixed['avg_size']:.0f}"),
        ("min / max chunk size", f"{legacy['min_size']}/{legacy['max_size']}",
                                 f"{fixed['min_size']}/{fixed['max_size']}"),
        ("BM25 recall@5", f"{legacy['retrieval']['recall']*100:.1f}%",
                          f"{fixed['retrieval']['recall']*100:.1f}%"),
        ("paras not indexed at all", legacy["retrieval"]["unfindable"],
                                     fixed["retrieval"]["unfindable"]),
    ]:
        print(f"{name:<34}{str(a):>18}{str(b):>18}")
    print("=" * 74)

    if legacy["lost_examples"]:
        print("\nExamples of text the LEGACY chunker discarded:")
        for ex in legacy["lost_examples"]:
            print(f"  - {ex}")

    print("\nVerdict:")
    if fixed["paras_lost"] < legacy["paras_lost"]:
        saved = legacy["paras_lost"] - fixed["paras_lost"]
        print(f"  FIXED preserves {saved} more paragraph(s) "
              f"({pct(saved, legacy['paras_total'])} of the corpus).")
    if fixed["chars_kept"] > legacy["chars_kept"]:
        print(f"  FIXED keeps {fixed['chars_kept'] - legacy['chars_kept']} more characters.")
    dr = fixed["retrieval"]["recall"] - legacy["retrieval"]["recall"]
    if abs(dr) > 0.001:
        print(f"  BM25 recall@5 {'improves' if dr > 0 else 'drops'} by {abs(dr)*100:.1f} points.")
    else:
        print("  BM25 recall@5 is unchanged.")


if __name__ == "__main__":
    main()
