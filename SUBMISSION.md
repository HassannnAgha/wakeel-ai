# Wakeel AI: submission

**Category:** Intelligent App Creation · **Team:** Agha Ali Hassan and Warda Fatima, FAST-NUCES Lahore (BS Computer Science)

**In one sentence:** a legal research assistant for Pakistani case law that answers from real Lahore High Court
judgments, links every claim to the paragraph it came from, and checks its own answers.

## The problem

Pakistani judgments are long, mostly in English and written for lawyers. Finding the paragraph that decides a question
takes real effort, and most citizens cannot read them at all. A general-purpose chatbot can answer fluently but may
invent a judgment or a section number that sounds right, and in law an invented citation is worse than no answer. The law
is also scattered: a judgment cites sections and earlier cases, and checking each one means finding it somewhere else.

## What we built

A web app (no accounts, no database server) with these parts:

- **Ask:** plain-language questions in English, Urdu or Roman Urdu, by typing or speaking. The answer is written only
  from retrieved passages, in a citizen or lawyer register, and each claim links to its paragraph. If the passages do not
  answer the question it says so instead of guessing.
- **Compare and review tables:** two judgments side by side, or a table of many judgments and your own documents with
  every cell cited, then questions over the table.
- **Explain:** upload an FIR, notice or court order (a scan or photo works): plain-words explanation, red flags ranked by
  severity, deadlines, similar judgments. Uploads are read for one answer and never stored.
- **Draft and workflows:** notices and applications (optionally in your own style), hearing preparation (questions a bench
  may ask), a dated chronology, a case strategy memo and a client letter. Facts you leave out stay as placeholders.
- **Library:** the judgments, 1,687 provisions of the Constitution, PPC, CrPC, CPC and family laws word for word with their
  source, a precedent map (which judgments rely on the same reported cases), usage insights.
- **Calculators:** limitation periods and Hanafi inheritance shares by fixed rules, with the rule shown, no AI.
- **Also:** saved matters exported as one PDF, answers read aloud where the browser has a voice, a guided tour, an MCP
  server so Claude or ChatGPT can use the library as a tool, and an HTTP API.

## How it works

1. **Judgment-aware chunking.** Judgments are split on numbered paragraphs, not at fixed lengths, and every passage keeps its
   case, court, parties and citations. (The first version of our chunker silently dropped a quarter of every judgment; we
   found this by measuring it, fixed it, and keyword recall at 5 went from 76.7% to 98.8%.)
2. **Hybrid retrieval.** Dense embeddings (`bge-large-en-v1.5`, ChromaDB) and BM25 are combined, then neighbouring and closing
   paragraphs are added, because a judgment's outcome is usually in the last ones.
3. **Agent routing (LangGraph).** A structured classifier sends each question to the right branch (answer, summarise, compare)
   and the app shows its plan before it acts. Routing is deterministic: the model never picks a tool.
4. **Grounded writing.** The answer prompt allows only the retrieved passages. A "not found" is rechecked once and then put to a
   model from the other provider (Gemini or OpenAI) before the user is told nothing was found.
5. **Checks that do not use AI.** Every citation must point at a real passage, and a fact check looks for each name, section,
   number and quotation of a cited sentence in the passage it cites. Statutes open word for word from official copies.

## Evidence

Reproducible with `evaluate_retrieval.py` and `evaluate_answers.py` (details and caveats in `EVALUATION.md`).

| Measure | Result |
|---|---|
| Right judgment in the top 10 results, 50 test questions | 99.5% |
| Mean reciprocal rank | 0.970 (0.764 before we fixed a search bug) |
| Questions answered, 30 questions, demo model `gpt-4o-mini` | 30 of 30 |
| Answers whose citations all point at real passages | 29 of 30 |
| Details checked against the cited passage, both models | 425, of which 30 flagged: 28 sit in a neighbouring passage, 2 in none |

The two details in no passage were a child's name and a date a model had worked out itself, caught on the cheaper model.
That model also refused 6 of the 30 questions although the right judgment had been retrieved each time, which is why we
demo on `gpt-4o-mini`.

## Limits we state plainly

- The library is 49 Lahore High Court judgments from 2023. Other courts are untested.
- The 50 test questions are ours and have not been reviewed by a lawyer; the numbers above are a guide, not a benchmark.
- The Urdu output, the legal templates, the guides and the calculators have not been reviewed by a lawyer.
- The fact check is a heuristic: it finds a wrong detail, not a wrongly argued claim worded differently from the source.
- Handwritten or Nastaliq scans are untested; a typeset Urdu FIR photo was read correctly.
- It is a research aid, not legal advice. Every answer links to its sources so a lawyer can verify it.

## Tech

Python 3.11, LangGraph, LangChain, ChromaDB, rank-bm25, sentence-transformers, Gemini and OpenAI behind one interface, a
dependency-free web server and a single-file front end (no framework, no CDN, works offline apart from the model calls),
`pytest` and Node tests (136 passing).

## Run it

```
pip install -r requirements.txt
copy .env.example .env        # add GOOGLE_API_KEY and/or OPENAI_API_KEY
python ui.py                  # http://127.0.0.1:8000, then press "Take the 90-second tour"
python preflight.py           # check the machine before a demo
```
