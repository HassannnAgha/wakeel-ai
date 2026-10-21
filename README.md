# Wakeel AI

**Your AI legal research assistant for Pakistani case law.**

Wakeel AI searches, summarizes and compares Pakistani court judgments (Supreme Court, Lahore,
Sindh and Islamabad High Courts) using a hybrid RAG pipeline and a Gemini-powered LangGraph agent. It
understands how Pakistani judgments are structured. It extracts case numbers, parties, judges,
citations (SCMR, PLD, PLJ) and provisions (PPC, CrPC, Constitution) so you can search by any of them.

> Built for the **4th International AI Championship (AIEF, 2026)**, category *Intelligent App Creation*.
> See [PROPOSAL.md](PROPOSAL.md) for the full project proposal.

## Features

- **Judgment-aware chunking:** splits on numbered paragraphs and tags each chunk as header / facts / arguments / analysis / conclusion
- **Metadata extraction:** court, case number, parties, judges, hearing date, case citations, statutory provisions
- **Hybrid search:** dense embeddings (`BAAI/bge-large-en-v1.5` + ChromaDB) combined with BM25 keyword search
- **Agentic routing:** a LangGraph state machine classifies each query as general / summarize / compare and routes it down a dedicated branch
- **Conversational memory:** follow-ups like *"what was the verdict in that case?"* work
- **Plain-language summaries** and **side-by-side case comparison**
- **Incremental indexing:** only new PDFs get processed

## Architecture

```
PDFs ─► text extraction ─► judgment-aware chunker ─► metadata + citations
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
         ChromaDB (semantic)                        BM25 (keyword)
                   └──────────────► hybrid ranker ◄────────┘
                                         │
query ─► LangGraph state machine (Gemini 3.5 Flash)

   START ─► classify ─┬─ REJECT ─► clarify ─────────────► END
                      ├─ GQ ─► retrieve ─► answer ──────► END
                      ├─ SQ ─► retrieve ─► summarize ───► END
                      └─ CQ ─► retrieve_pair ─► compare ► END
```

Routing is deterministic: the classifier returns a structured intent and the graph
picks the branch, so the model never has to guess which tool to call. Conversation
memory is a LangGraph checkpointer keyed by `thread_id`.

## Getting started

```bash
git clone https://github.com/HassannnAgha/wakeel-ai.git
cd wakeel-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your GOOGLE_API_KEY
```

If you have no NVIDIA GPU (or an older one), install the much smaller CPU build of
PyTorch instead of the default CUDA wheels:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Put judgment PDFs in `pdfs/`, then run either interface:

```bash
python ui.py
```

That serves the web UI at http://127.0.0.1:8000. For the terminal instead:

```bash
python wakeel_ai.py
```

The first run downloads the embedding model and indexes every PDF. Later runs only index new files.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Required. From [Google AI Studio](https://aistudio.google.com/apikey). |
| `WAKEEL_MODEL` | `gemini-3.5-flash` | Chat model. `gemini-2.5-flash` is retired for new keys. |
| `WAKEEL_VERBOSE` | off | Set to `1` to print routing decisions as the graph runs. |
| `WAKEEL_LEGACY_CHUNKING` | off | Set to `1` to restore the original chunking behaviour. |

The Gemini free tier allows only a small number of requests per day per model. When
it runs out the app says so plainly rather than failing silently; enable billing or
switch `WAKEEL_MODEL` to a model with quota left.

### Example queries

- `Find bail cases under Section 497 CrPC`
- `Summarize the Lahore High Court judgment in Criminal Appeal 123 of 2019`
- `Which judgments cite 2016 SCMR 2073?`
- `Compare these two murder appeals: <doc_id_1>, <doc_id_2>`

## Building a corpus

`pdfs/` starts empty. To fetch Lahore High Court judgments (public court records):

```bash
python scrape_judgments.py --year 2023 --start 100 --end 700
```

Judgment numbering is sparse, so a sweep of 600 candidates yields roughly 50 PDFs.
Index them by starting either interface.

## Chunking

The chunker splits judgments on numbered paragraphs rather than on a fixed character
count, and carries a paragraph shorter than `min_chunk_size` into the next chunk
instead of discarding it. Measured on 49 real Lahore High Court judgments
(647 numbered paragraphs), that second detail alone recovers 23% of the corpus:

| Metric | Legacy (drop short) | Fixed (merge short) |
|---|---|---|
| Paragraphs preserved | 496/647 | **647/647** |
| Text retention | 96.1% | **99.9%** |
| Citations captured | 577 | **603** |
| Provisions captured | 783 | **821** |
| BM25 recall@5 | 76.7% | **98.8%** |

Among the paragraphs that would otherwise be dropped was *"The law favours
adjudication on merits. Technicalities should not..."* — a holding, not boilerplate.

Set `WAKEEL_LEGACY_CHUNKING=1` to run with the original behaviour for comparison.

## Roadmap

Full details are in [PROPOSAL.md](PROPOSAL.md#5-key-features--deliverables).

**Round 2: Screening**
- [ ] Streamlit web application with a source-paragraph panel, deployed publicly
- [ ] Answers that cite case, court and paragraph, with a "not found in sources" response
- [ ] Pre-indexed demonstration corpus

**Round 3: Mentorship**
- [ ] Citizen, student and lawyer answer modes
- [ ] Urdu and Roman Urdu support, with speech input
- [ ] Document explanation for FIRs, legal notices and court orders
- [ ] Cross-encoder re-ranking and Reciprocal Rank Fusion
- [ ] Statute linker (Pakistan Code)
- [ ] Evaluation benchmark (Recall@k, MRR, faithfulness)

**Final Round**
- [ ] Precedent citation graph
- [ ] Case timeline extraction
- [ ] OCR for scanned judgments
- [ ] Answer-quality feedback collection

## Team

| Name | GitHub |
|---|---|
| Agha Ali Hassan | [@HassannnAgha](https://github.com/HassannnAgha) |
| Warda Fatima | [@WardaFatima15](https://github.com/WardaFatima15) |

FAST-NUCES, Lahore · BS Computer Science

## Disclaimer

Wakeel AI is a research and educational tool. It is **not legal advice** and does not replace a
qualified lawyer. Always check results against the original judgment.
