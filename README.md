# Wakeel AI

**Your AI legal research assistant for Pakistani case law.**

Wakeel AI searches, summarizes and compares Pakistani court judgments using a hybrid RAG
pipeline and a Gemini-powered LangGraph agent. Rather than splitting documents on a fixed
character count, it understands how judgments are structured: it chunks on numbered
paragraphs and extracts case citations (SCMR, PLD, PLJ) and statutory provisions
(PPC, CrPC, Constitution) so you can search by either.

Built and measured against **Lahore High Court** judgments. The chunker recognises
Supreme Court, Sindh and Islamabad High Court formats too, but those are not yet verified
— see [Current limitations](#current-limitations).

> Built for the **4th International AI Championship (AIEF, 2026)**, category *Intelligent App Creation*.
> See [PROPOSAL.md](PROPOSAL.md) for the full project proposal.

## Features

- **Judgment-aware chunking:** splits on numbered paragraphs and tags each chunk as header / facts / arguments / analysis / conclusion
- **Citation and provision extraction:** case citations (SCMR, PLD, PLJ, AIR) and statutory provisions (PPC, CrPC, Constitution), attached to every chunk
- **Hybrid search:** dense embeddings (`BAAI/bge-large-en-v1.5` + ChromaDB) combined with BM25 keyword search
- **Agentic routing:** a LangGraph state machine classifies each query as general / summarize / compare and routes it down a dedicated branch
- **Conversational memory:** follow-ups like *"what was the verdict in that case?"* work
- **Plain-language summaries** and **side-by-side case comparison**
- **Incremental indexing:** only new PDFs get processed
- **Web app pages:** Ask, Summarize, Compare (two judgments, or a cited review table of many judgments and
  your own documents), Explain (your FIR, notice or order, with red flags), Draft (documents, optionally in
  your own style, and workflows: strategy memo, hearing preparation, chronology, client letter), Guides,
  Calculate (limitation, inheritance), Library (judgments, statutes, insights, activity), Matters
- **Trust:** every answer links each claim to its paragraph; a fact check (no AI) looks for every name,
  section, number and quotation in the cited paragraph; statutes (Constitution, PPC, CrPC, CPC, family
  laws) open word for word from official copies; case citations open the library judgments that rely on them
- **Urdu and English voice input**, and a **provider fallback** (Gemini and OpenAI stand in for each other)

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
| `GOOGLE_API_KEY` | — | Required for Gemini. From [Google AI Studio](https://aistudio.google.com/apikey). |
| `WAKEEL_PROVIDER` | `gemini` | Chat model provider: `gemini` or `openai`. |
| `OPENAI_API_KEY` | — | Required when `WAKEEL_PROVIDER=openai`. From [OpenAI](https://platform.openai.com/api-keys). |
| `WAKEEL_MODEL` | `gemini-3.5-flash` / `gpt-4o-mini` | Chat model; the default depends on the provider. `gemini-2.5-flash` is retired for new keys. |
| `WAKEEL_FALLBACK` | on | When the other provider's key is also set, a failed or hung call (quota, outage) is answered by the other provider instead. Set to `0` to turn off. |
| `WAKEEL_TIMEOUT` | `90` | Seconds one request to the main provider may take before the fallback answers. |
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

## API

Everything in the web app is available over plain HTTP and JSON; the app's own pages use
the same endpoints. Open http://127.0.0.1:8000/developers for the full list with examples.

```bash
curl -s http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message": "Find bail cases under Section 497 CrPC", "mode": "lawyer", "language": "en"}'
```

Main endpoints: `POST /api/chat` (and `/api/chat/stream` for live agent steps),
`/api/summarize`, `/api/compare`, `/api/table`, `/api/explain`, `/api/explain/ask`, `/api/draft`,
`/api/workflow`, `/api/feedback`; `GET /api/judgments`, `/api/judgments/<doc_id>`, `/api/statutes`,
`/api/citations`, `/api/insights`, `/api/activity`, `/api/stats`.

### Use Wakeel AI from Claude or ChatGPT (MCP)

`mcp_server.py` exposes the judgments and statutes as MCP tools (`search_judgments`,
`get_judgment`, `list_judgments`, `get_statute`, `search_statutes`) over stdio, with no extra
dependencies and no API key. Add it to an MCP client, for example Claude Desktop:

```json
{"mcpServers": {"wakeel": {
  "command": "C:/path/to/aief/.venv/Scripts/python.exe",
  "args": ["C:/path/to/aief/mcp_server.py"]}}}
```

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

## Current limitations

Measured honestly, so you know what is and isn't working:

| Area | Status |
|---|---|
| Chunking, citations, provisions | Working. 603 citations and 821 provisions extracted across 49 judgments. |
| Court name | Extracted cleanly for 55% of judgments; the rest run together with following lines. |
| Case number | 26%. Only some of the many case-number formats are matched. |
| Hearing date | 24%. |
| Parties, judges | 4% and 8%. The header patterns assume a format real judgments do not use. |
| Courts other than Lahore | Untested. The Supreme Court site blocks automated downloads and the Sindh portal requires a login, so no corpus was available. |
| Scanned / image-only PDFs | Not handled; text extraction returns empty and the document is skipped. |
| Urdu text | Untested. |

Structural metadata extraction is the main known gap. Search, chunking and the agent do
not depend on it — it affects the case/party/judge labels shown alongside results.

## Roadmap

Full details are in [PROPOSAL.md](PROPOSAL.md#5-key-features--deliverables). Ticked items are built and tested.

**Round 2: Screening**
- [x] Web application with a source-paragraph panel (a dependency-free server rather than Streamlit)
- [x] Answers that cite case, court and paragraph, with a "not found in sources" response
- [x] Pre-indexed demonstration corpus (49 Lahore High Court judgments)
- [ ] Deployed publicly (it runs locally; not deployed)

**Round 3: Mentorship**
- [x] Citizen and lawyer answer modes
- [x] Urdu and Roman Urdu answers, with Urdu and English speech input
- [x] Document explanation for FIRs, legal notices and court orders, with red flags
- [ ] Cross-encoder re-ranking and Reciprocal Rank Fusion (not needed so far: hybrid search already finds
      the right judgment in the top 10 for 99.5% of the 50 test questions)
- [x] Statute linker: the Constitution, PPC, CrPC, CPC and the main family laws, word for word with sources
- [x] Evaluation benchmark: Recall@k and MRR (`evaluate_retrieval.py`), plus a citation fact check on
      every answer

**Final Round**
- [x] Precedent citation graph (each judgment's precedents and the judgments that share them)
- [x] Case timeline extraction (the chronology workflow, with quoted dates)
- [ ] OCR for scanned judgments at indexing time (uploaded scans are already read on the Explain page)
- [x] Answer-quality feedback collection (helpful / not helpful, and an activity view)

**Added beyond the plan:** review tables over many judgments and your own documents, drafting (also in your
own style), workflows (strategy memo, hearing preparation, client letter), calculators, citizen guides, saved
matters, an MCP server and an HTTP API.

## Team

| Name | GitHub |
|---|---|
| Agha Ali Hassan | [@HassannnAgha](https://github.com/HassannnAgha) |
| Warda Fatima | [@WardaFatima15](https://github.com/WardaFatima15) |

FAST-NUCES, Lahore · BS Computer Science

## Disclaimer

Wakeel AI is a research and educational tool. It is **not legal advice** and does not replace a
qualified lawyer. Always check results against the original judgment.
