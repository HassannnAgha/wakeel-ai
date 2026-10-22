# Wakeel AI: evaluation

How well Wakeel AI finds the right judgment, and whether its answers cite real sources.
Every number here can be reproduced with `evaluate_retrieval.py` (no API key needed for
the retrieval part).

## Slide summary

| | Result |
|---|---|
| Right judgment in the top 10 (Recall@10, hybrid search) | **99.5%** |
| Rank of the first right judgment (MRR, hybrid search) | **0.970** |
| A search bug we found and fixed (MRR before → after) | **0.764 → 0.970** |
| Paragraphs the old chunker silently dropped, now kept | **151 of 647 → 0** |
| Answers that cite a passage which exists | **8 of 8** checked |

**Two of our own bugs, found by measuring rather than by reading the code:**

1. **The chunker lost a quarter of every judgment.** Short paragraphs were discarded,
   including holdings such as *"The law favours adjudication on merits."* Keyword recall@5
   went from 76.7% to 98.8% once they were kept.
2. **Search kept the wrong paragraph.** Every chunk of a judgment starts with the same
   `[Case: …] [parties]` header, and search identified chunks by their first 50 characters,
   so 770 chunks collapsed into 91. Each judgment kept one arbitrary paragraph, often not
   the one that matched. Keying chunks by their index id fixed it: on the README's own
   example ("bail cases under Section 497 CrPC"), passages mentioning 497 went from 0 to 7.

## Retrieval

50 questions, each labelled with the judgment(s) that should answer it: 39 about a topic,
5 about a statutory provision and 6 naming a case or its parties. A question counts a
judgment as found if any of its passages is in the top 10. MRR is 1 / the rank of the first
right judgment.

| Search | Recall@10 | MRR |
|---|---|---|
| Meaning only (dense, bge-large) | 96.0% | 0.940 |
| Keywords only (BM25) | 98.5% | 0.957 |
| **Hybrid (what the app uses)** | **99.5%** | **0.970** |

By question type (Recall@10 / MRR):

| Type | n | Dense | Keyword | Hybrid |
|---|---|---|---|---|
| Topic | 39 | 100% / 0.99 | 98% / 0.94 | 99% / 0.97 |
| Provision | 5 | 60% / 0.50 | 100% / 1.00 | 100% / 0.90 |
| Case or parties | 6 | 100% / 1.00 | 100% / 1.00 | 100% / 1.00 |

This is why the search is hybrid: meaning-based search misses questions about a specific
section (60% recall), and keyword search covers them.

Before the chunk-key fix, the same 50 questions scored: dense 96.0% / 0.659, hybrid
100% / 0.764.

The only hybrid miss: q09, the 2023 Punjab Assembly election case, which the corpus holds
as four separate orders. Search found three of the four.

## Answers

The first 8 questions were run through the full agent (lawyer mode, Gemini 3.5 Flash-Lite):

| | |
|---|---|
| Answered rather than "not found" | 8 / 8 |
| Answers with a citation that points to a real passage | 8 / 8 |
| … citing a judgment labelled relevant | 8 / 8 |

The run stopped at 8 when the free Gemini quota ran out; run it again with
`--answers 50` on a paid key for the full set.

## Caveats (say these if asked)

- **The labels are a draft.** The 50 questions were written while reading the judgments and
  have not yet been reviewed by a lawyer. Questions written that way share wording with the
  judgments, which flatters keyword search. Each entry in `eval_queries.jsonl` carries
  `"reviewed": false` until checked; `--reviewed-only` reports on the checked ones alone.
- **The corpus is small**: 49 Lahore High Court judgments from 2023, of which two are exact
  duplicates and four are orders in the same case. With so few judgments, Recall@10 sits near
  its ceiling; MRR is the number that separates the methods.
- **8 answers is a small sample** for the citation check.

## Reproduce

```bash
.venv/Scripts/python.exe evaluate_retrieval.py                 # retrieval only, free
.venv/Scripts/python.exe evaluate_retrieval.py --answers 10    # plus 10 real answers
.venv/Scripts/python.exe evaluate_retrieval.py --reviewed-only # only reviewed questions
```
