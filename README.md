# Wakeel AI ⚖️

**Your AI legal research assistant for Pakistani case law.**

Wakeel AI searches, summarizes and compares Pakistani court judgments (Supreme Court, Lahore,
Sindh and Islamabad High Courts) using a hybrid RAG pipeline and a Gemini-powered agent. It
understands how Pakistani judgments are structured. It extracts case numbers, parties, judges,
citations (SCMR, PLD, PLJ) and provisions (PPC, CrPC, Constitution) so you can search by any of them.

> Built for the **4th International AI Championship (AIEF, 2026)**, category *Intelligent App Creation*.
> See [PROPOSAL.md](PROPOSAL.md) for the full project proposal.

## Features

- **Judgment-aware chunking:** splits on numbered paragraphs and tags each chunk as header / facts / arguments / analysis / conclusion
- **Metadata extraction:** court, case number, parties, judges, hearing date, case citations, statutory provisions
- **Hybrid search:** dense embeddings (`BAAI/bge-large-en-v1.5` + ChromaDB) combined with BM25 keyword search
- **Agentic routing:** classifies each query as general / summarize / compare and picks the right tool
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
query ─► intent classifier ─► LangChain agent (Gemini 2.5 Flash)
                               ├─ HybridRAGSearch
                               ├─ Summarizer
                               ├─ CompareTopics
                               └─ IndexNewDocuments
```

## Getting started

```bash
git clone https://github.com/HassannnAgha/wakeel-ai.git
cd wakeel-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your GOOGLE_API_KEY
```

Put judgment PDFs in `pdfs/`, then run:

```bash
python wakeel_ai.py
```

The first run downloads the embedding model and indexes every PDF. Later runs only index new files.

### Example queries

- `Find bail cases under Section 497 CrPC`
- `Summarize the Lahore High Court judgment in Criminal Appeal 123 of 2019`
- `Which judgments cite 2016 SCMR 2073?`
- `Compare these two murder appeals: <doc_id_1>, <doc_id_2>`

## Roadmap

- [ ] Web app with chat interface and a source viewer that highlights cited paragraphs
- [ ] Clickable citations on every answer, with an "insufficient sources" fallback
- [ ] Citizen / Student / Lawyer answer modes
- [ ] Urdu and Roman-Urdu queries, plus voice input
- [ ] "Explain my document" for FIRs, legal notices and court orders
- [ ] Precedent citation graph
- [ ] Cross-encoder re-ranking and OCR for scanned judgments
- [ ] Retrieval evaluation benchmark (Recall@k, MRR, faithfulness)

## Team

| Name | GitHub |
|---|---|
| Agha Ali Hassan | [@HassannnAgha](https://github.com/HassannnAgha) |
| Warda Fatima | [@WardaFatima15](https://github.com/WardaFatima15) |

FAST-NUCES, Lahore · BS Computer Science

## Disclaimer

Wakeel AI is a research and educational tool. It is **not legal advice** and does not replace a
qualified lawyer. Always check results against the original judgment.
