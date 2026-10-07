# Wakeel AI — Your AI Legal Research Assistant for Pakistani Case Law

**Category:** Intelligent App Creation
**Event:** 4th International AI Championship (AIEF), 2026
**Team:** Agha Ali Hassan ([@HassannnAgha](https://github.com/HassannnAgha)) · Warda Fatima ([@WardaFatima15](https://github.com/WardaFatima15))
**University:** FAST-NUCES, Lahore · BS Computer Science, 7th semester
**Repository:** https://github.com/HassannnAgha/wakeel-ai

> **Keywords:** Legal AI · Retrieval-Augmented Generation (RAG) · Hybrid Search · Pakistani Case Law · Agentic LLM

---

## 1. Project Overview & Problem Statement

Pakistan's courts are overloaded. The Law and Justice Commission of Pakistan reported
**2.36 million pending cases** as of 31 December 2024, with about 83% of them in the
district judiciary. Part of what keeps cases slow is legal research, which is still
mostly manual:

- **Lawyers and junior associates** spend hours searching PLD, SCMR, PLJ and High Court
  judgments for relevant precedents, often with keyword-only search.
- **Law students** have no affordable tool that explains *why* a court ruled the way it did.
- **Ordinary citizens** facing an FIR, a property dispute or a bail hearing cannot read a
  50-page English judgment full of Latin and statutory references, and many cannot pay
  for a first consultation.

Generic chatbots (ChatGPT, Gemini) don't help much here. They are not grounded in
Pakistani judgments, they make up citations, and they don't follow how Pakistani
judgments are laid out (case number → parties → coram → facts → arguments → analysis → order).

## 2. Proposed Solution

**Wakeel AI** is an AI-native legal research assistant built on a **Hybrid RAG + agent**
architecture designed for Pakistani court judgments.

```
 PDF judgments ──► Text extraction ──► Judgment-aware chunker ──► Metadata extraction
                                         (paragraph / section)     (court, case no., parties,
                                                                    judges, citations, PPC/CrPC
                                                                    sections, hearing date)
                                                    │
                          ┌─────────────────────────┴──────────────────────────┐
                          ▼                                                    ▼
              Dense index (ChromaDB +                                Sparse index (BM25)
              BAAI/bge-large-en-v1.5)                                keyword / citation match
                          └──────────────► Hybrid ranker ◄─────────────────────┘
                                                │
 User query ──► Intent classifier (GQ/SQ/CQ) ──► LangChain agent (Gemini 2.5 Flash)
                                                ├─ HybridRAGSearch
                                                ├─ Summarizer (layman language)
                                                ├─ CompareCases (similarities / differences)
                                                └─ IndexNewDocuments
```

**What is already built (working prototype):**

| Component | Status |
|---|---|
| PDF text extraction with on-disk caching | ✅ |
| **Pakistani-judgment-aware chunker**: splits on numbered paragraphs, labels each chunk as header / facts / arguments / analysis / conclusion | ✅ |
| Metadata extraction: court, case number, parties, judges, hearing date | ✅ |
| Citation extraction (SCMR, PLD, PLJ, AIR, "X v. Y") and statute extraction (Sections of PPC/CrPC/CPC, Constitutional Articles) | ✅ |
| Contextual chunk enrichment (case header + previous-paragraph context prepended before embedding) | ✅ |
| Hybrid retrieval: semantic (ChromaDB, cosine) + BM25, weighted fusion | ✅ |
| Incremental indexing (only new PDFs are processed) | ✅ |
| Query classifier: General / Summarize / Compare, with follow-up resolution via chat memory | ✅ |
| Off-topic refusal guardrail | ✅ |
| Case comparison and plain-language summarization tools | ✅ |

## 3. Objectives & Expected Outcomes

**Objectives**
1. Cut the time to find relevant precedents from hours to minutes.
2. Return **grounded answers only**: every claim links back to the exact judgment and paragraph.
3. Make judgments readable for non-lawyers in plain English and Urdu.
4. Show that a domain-specific chunking and retrieval pipeline beats generic RAG on Pakistani legal text.

**Measurable outcomes (target by finale)**
| Metric | Target |
|---|---|
| Retrieval Recall@10 on a hand-labelled test set of 50 legal queries | ≥ 0.85 |
| Answers with a valid, verifiable citation | ≥ 95% |
| Hybrid vs. dense-only retrieval improvement (MRR) | measurable gain, reported |
| Median response time | < 8 s |
| Indexed corpus | 500+ SC / LHC / SHC / IHC judgments |

## 4. Target Users

| User | Need | How Wakeel AI helps |
|---|---|---|
| Practising lawyers & associates | Fast precedent search, citation lookup | Search by citation, section ("302 PPC"), judge or facts; compare two rulings |
| Law students | Understand reasoning, exam/moot prep | Section-tagged summaries (facts → arguments → analysis → verdict) |
| Citizens & litigants | Understand their situation and a judgment | Plain-language and Urdu explanations, with a clear "not legal advice" notice |
| Legal-aid NGOs & paralegals | Handle many cases with few lawyers | Quick triage of relevant law and similar past cases |
| Journalists & researchers | Track judicial trends | Search across courts, judges and provisions |

## 5. Key Features / Deliverables

**Core (prototype, already working)**
- Hybrid semantic + keyword search over Pakistani judgments
- Judgment-structure-aware chunking and metadata extraction
- Agentic routing: search, summarize, compare, index
- Follow-up-aware conversational memory

**Planned for the MVP / finale**
1. **Web application** (Streamlit → React): chat interface, source panel, and a PDF viewer that highlights the cited paragraph.
2. **Grounded answers with clickable citations**: every answer cites `[Case No., Court, Para N]`. If no supporting source is found, the assistant says so instead of guessing.
3. **Adaptive user modes** (the "intelligent UX" part): *Citizen*, *Student* and *Lawyer* modes change vocabulary, depth and output format for the same query.
4. **Urdu & Roman-Urdu support**: ask in Urdu or Roman Urdu and get answers in the same language. Voice input with Whisper for users who prefer speaking.
5. **"Explain my document"**: upload an FIR, legal notice or court order. Wakeel AI explains it in plain language, picks out the provisions involved, and finds similar past cases.
6. **Precedent / citation graph**: an interactive graph of which judgments cite which, built from citations the pipeline already extracts.
7. **Statute linker**: when "Section 497 CrPC" or "Article 10-A" appears, show the actual text of the provision inline.
8. **Re-ranking with a cross-encoder** (`bge-reranker`) and **Reciprocal Rank Fusion**, to improve retrieval precision.
9. **OCR pipeline** (Tesseract/EasyOCR) for scanned judgments, which are common in Pakistani court archives.
10. **Evaluation dashboard**: Recall@k, MRR and faithfulness scores on a labelled benchmark, so the claims above can be checked.

## 6. Implementation Plan & Timeline

| Dates (2026) | Milestone | Deliverables |
|---|---|---|
| Done | Core engine | Chunker, metadata extraction, hybrid RAG, agent, CLI |
| Oct 7 – Oct 12 | **Idea + prototype submission** | Code cleanup, `.env` config, Streamlit UI, grounded citations, this proposal |
| Oct 12 – Oct 16 | Screening | Expand corpus to 500+ judgments, build the 50-query evaluation set |
| Oct 16 – Oct 20 | MVP hardening (after shortlist) | Re-ranker, user modes, Urdu support, "Explain my document" |
| Oct 20 – Oct 23 | Mentorship & polish | Citation graph, evaluation dashboard, apply mentor feedback, demo script |
| **Oct 24** | **Final live pitch** at FAST-NUCES, Lahore | Live demo + metrics |

**Tech stack:** Python · LangChain · Google Gemini 2.5 Flash · ChromaDB · BM25 (`rank_bm25`) ·
Sentence-Transformers (`BAAI/bge-large-en-v1.5`) · pdfplumber · Streamlit / React · Whisper (planned)

## 7. Resources / Requirements

- **Data:** publicly available judgments from the Supreme Court of Pakistan and the High Court
  websites (Lahore, Sindh, Islamabad, Peshawar, Balochistan), plus Pakistan Code (pakistancode.gov.pk) for statute text.
- **Compute:** one GPU (or Colab/Kaggle) for bulk embedding. Inference runs on a CPU server.
- **APIs:** Google Gemini API (free tier covers prototype usage).
- **Expertise:** a legal mentor or law faculty member to validate outputs and help label the evaluation set.
- **Hosting:** Hugging Face Spaces / Render for the public demo.

## 8. Potential Challenges & Risks

| Risk | Mitigation |
|---|---|
| **Hallucinated citations or wrong legal information** | Strict grounding (answer only from retrieved text), citation verification against the index, "insufficient sources" fallback, a visible "not legal advice" disclaimer |
| Scanned / poorly formatted PDFs | OCR fallback; regex patterns tuned on real judgments; basic chunker as a fallback |
| Inconsistent judgment formats across courts | Per-court header patterns, with fallback to generic chunking |
| Urdu legal terminology | Bilingual glossary of common legal terms; human review of sample outputs |
| Privacy of user-uploaded documents | Uploads processed in-session, not added to the shared index, deleted after the session |
| LLM API cost / rate limits | Caching, a small model for classification, and a local LLM option (e.g. Llama/Qwen via Ollama) |
| Over-reliance by citizens | Mode-specific disclaimers and links to free legal-aid services |

## 9. Additional Details

- **Why this fits Intelligent App Creation:** the app adapts to the user. It classifies intent,
  remembers conversation context, changes tone and depth for citizens, students and lawyers,
  and picks its own tools (search / summarize / compare) instead of following a fixed flow.
- **Social impact:** better access to justice. A citizen who understands their FIR or a past
  bail judgment is better placed to deal with lawyers and courts.
- **Scalability:** the same pipeline can be extended to tax tribunals, FBR rulings, SECP
  regulations and labour courts by adding citation and section patterns.
- **Ethics:** Wakeel AI is a research and understanding aid, **not a substitute for a lawyer**. It does
  not predict individual case outcomes and is clear about where its sources come from.
