# Wakeel AI

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
