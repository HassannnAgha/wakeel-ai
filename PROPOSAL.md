# Wakeel AI: AI Legal Research Assistant for Pakistani Case Law

| | |
|---|---|
| **Competition** | 4th International AI Championship 2026, Artificial Intelligence Education Foundation (AIEF) |
| **Category** | Intelligent App Creation |
| **Team** | Agha Ali Hassan (GitHub: HassannnAgha), Warda Fatima (GitHub: WardaFatima15) |
| **Institution** | FAST-NUCES, Lahore, BS Computer Science, 7th semester |
| **Repository** | https://github.com/HassannnAgha/wakeel-ai |
| **Keywords** | Legal AI, Retrieval-Augmented Generation, Hybrid Search, Pakistani Case Law, Agentic LLM |

---

## 1. Project Overview & Problem Statement

### Overview

Wakeel AI is an AI-powered legal research assistant for Pakistani case law. Users ask questions
in natural language. The system retrieves the relevant court judgments, summarizes them in plain
language, and compares rulings side by side. It is built on a hybrid retrieval-augmented
generation (RAG) pipeline that is designed around how Pakistani judgments are structured.

### Problem Statement

According to the Law and Justice Commission of Pakistan, 2,362,135 cases were pending in
Pakistani courts as of 31 December 2024. About 83% of them were in the district judiciary.
Legal research is one of the slow, manual steps behind this backlog:

1. **Legal research takes too long.** Lawyers and associates search through PLD, SCMR, PLJ and
   High Court reports by hand or with keyword search. Finding relevant precedents for a single
   matter can take hours.
2. **Judgments are hard for non-lawyers to read.** A litigant dealing with an FIR, a bail hearing
   or a property dispute cannot easily follow a long English judgment full of technical and
   statutory language. Many cannot pay for a first consultation just to understand their position.
3. **General-purpose AI tools are unreliable for this.** General chatbots are not grounded in
   Pakistani judgments, can make up citations, and do not account for the standard layout of a
   judgment (case number, parties, coram, facts, arguments, analysis, order).

## 2. Proposed Solution

Wakeel AI combines a domain-specific document pipeline with an LLM agent:

1. **Judgment-aware ingestion.** Judgments are split by their numbered paragraphs, not cut into
   fixed-size pieces. Each chunk is labelled as header, facts, arguments, analysis or conclusion.
   The pipeline extracts the court, case number, parties, judges, hearing date, case citations
   (SCMR, PLD, PLJ, AIR) and statutory provisions (PPC, CrPC, CPC, Constitution).
2. **Hybrid retrieval.** Dense semantic search (BAAI/bge-large-en-v1.5 embeddings in ChromaDB)
   is combined with BM25 keyword search. The keyword side matches exact citations and section
   numbers, which semantic search alone often misses.
3. **Agentic reasoning.** Each query is first classified as a general query, a summarization
   request or a comparison request. A LangChain agent running on Google Gemini 2.5 Flash then
   chooses the right tool and keeps conversation memory, so follow-up questions work.

### System Architecture

```
PDF judgments -> Text extraction -> Judgment-aware chunker -> Metadata and citation extraction
                                              |
                     +------------------------+------------------------+
                     v                                                 v
        Dense index (ChromaDB, bge-large)                 Sparse index (BM25)
                     +-----------------> Hybrid ranker <---------------+
                                              |
User query -> Intent classifier -> LangChain agent (Gemini 2.5 Flash)
                                     - Hybrid search tool
                                     - Summarization tool
                                     - Case comparison tool
                                     - Document indexing tool
```

## 3. Objectives & Expected Outcomes

### Objectives

1. Cut the time needed to find relevant precedents from hours to minutes.
2. Ground every answer in source material, with traceable references to the judgment and paragraph.
3. Make judgments understandable to non-lawyers in English and Urdu.
4. Show that domain-specific chunking and hybrid retrieval beat generic RAG on Pakistani legal text.

### Expected Outcomes

| Metric | Target by Final Round |
|---|---|
| Retrieval Recall@10 on a labelled set of 50 legal queries | 0.85 or higher |
| Answers containing a verifiable source citation | 95% or higher |
| Hybrid vs. dense-only retrieval (Mean Reciprocal Rank) | Measured and reported |
| Median end-to-end response time | Under 8 seconds |
| Indexed corpus | 500+ Supreme Court and High Court judgments |

## 4. Target Users

| User Group | Need | How Wakeel AI Helps |
|---|---|---|
| Practising lawyers and associates | Fast precedent and citation research | Search by citation, statutory section, judge, party or facts. Compare rulings. |
| Law students | Understanding judicial reasoning; exam and moot court preparation | Summaries organized by facts, arguments, analysis and verdict |
| Citizens and litigants | Understanding their legal situation | Plain-language explanations in English and Urdu |
| Legal aid organizations and paralegals | Handling high caseloads with limited staff | Quick identification of relevant law and similar past cases |
| Researchers and journalists | Tracking judicial trends | Search across courts, judges and provisions |

## 5. Key Features / Deliverables

### Implemented in the Prototype

1. PDF text extraction with on-disk caching
2. Judgment-aware chunking based on numbered paragraphs and section types
3. Metadata extraction: court, case number, parties, judges, hearing date
4. Extraction of case citations and statutory provisions
5. Contextual chunk enrichment (case header and previous-paragraph context are added before embedding)
6. Hybrid semantic and BM25 retrieval with weighted score fusion
7. Incremental indexing that processes only new documents
8. Query classification (general, summarize, compare) with follow-up resolution from chat memory
9. Refusal of off-topic queries
10. Plain-language summarization and structured case comparison

### Planned Deliverables

**Round 2: Screening (by October 16)**
1. Web application (Streamlit) with a chat interface and a panel showing source paragraphs, deployed publicly
2. Answers that cite case number, court and paragraph, with a "not found in sources" response instead of guessing
3. A pre-indexed demonstration corpus of real judgments
4. Fixes to how documents are looked up for comparison and how classified queries are routed

**Round 3: Mentorship (by October 23)**
1. Adaptive answer modes for citizens, students and lawyers, which change vocabulary, depth and output format
2. Urdu and Roman Urdu input and output, with speech input using Whisper
3. Document explanation: users upload an FIR, legal notice or court order and get a plain-language explanation and the provisions involved
4. Cross-encoder re-ranking and Reciprocal Rank Fusion for better retrieval precision
5. A statute linker that shows the text of a cited provision from Pakistan Code
6. An evaluation benchmark reporting Recall@k, MRR and answer faithfulness

**Final Round (October 24)**
1. An interactive precedent graph showing which judgments cite which
2. Case timeline extraction (FIR, arrest, trial, appeal)
3. Aggregate statistics by provision, court and year, presented as statistics and not as predictions for individual cases
4. OCR support for scanned judgments
5. User feedback collection on answer quality
6. A WhatsApp interface, as a stretch goal, for citizens who do not use web applications

## 6. Implementation Plan & Timeline

| Period (2026) | Phase | Activities |
|---|---|---|
| Completed | Core engine | Ingestion pipeline, metadata extraction, hybrid retrieval, agent, command-line interface |
| October 7 to 12 | Idea and prototype submission | Code cleanup, configuration management, proposal and repository |
| October 12 to 16 | Screening | Web application, grounded citations, demonstration corpus, public deployment |
| October 16 to 20 | MVP development | Adaptive modes, Urdu support, document explanation, re-ranking |
| October 20 to 23 | Mentorship and refinement | Evaluation benchmark, statute linker, precedent graph, changes from mentor feedback |
| October 24 | Final championship | Live demonstration and presentation at FAST-NUCES, Lahore |

### Technology Stack

| Layer | Technology |
|---|---|
| Language model | Google Gemini 2.5 Flash |
| Orchestration | LangChain |
| Embeddings | Sentence-Transformers (BAAI/bge-large-en-v1.5) |
| Vector store | ChromaDB |
| Keyword retrieval | BM25 (rank_bm25) |
| Document processing | pdfplumber, Tesseract OCR (planned) |
| Speech | Whisper (planned) |
| Interface | Streamlit (planned) |

## 7. Resources or Requirements

1. **Data:** publicly available judgments from the Supreme Court of Pakistan and the Lahore, Sindh,
   Islamabad, Peshawar and Balochistan High Courts. Statute text from Pakistan Code (pakistancode.gov.pk).
2. **Compute:** GPU access (local or cloud notebook) for bulk embedding. Inference runs on a CPU server.
3. **API access:** Google Gemini API.
4. **Hosting:** Hugging Face Spaces or an equivalent platform for the public demonstration.
5. **Domain expertise:** guidance from a legal practitioner or law faculty member to check outputs
   and help label the evaluation dataset.

## 8. Potential Challenges or Risks

| Risk | Mitigation |
|---|---|
| Incorrect or fabricated legal information | Answers come only from retrieved text, citations are checked against the index, the system says when sources are insufficient, and a disclaimer is always shown |
| Scanned or poorly formatted PDFs | OCR fallback; extraction patterns tuned on real judgments; generic chunking as a fallback |
| Different judgment formats across courts | Court-specific header patterns, with fallback to generic chunking |
| Accuracy of Urdu legal terminology | A curated bilingual glossary of legal terms; manual review of sample outputs |
| Privacy of user-uploaded documents | Uploads are processed only within the session and are not added to the shared index |
| API cost and rate limits | Response caching, a lighter model for classification, and an option to run an open-source model locally |
| Over-reliance by non-expert users | Disclaimers in each answer mode and links to free legal aid services |

## 9. Additional Details

1. **Fit with the category:** the application changes its behaviour based on what the user needs.
   It classifies intent, keeps track of the conversation, adjusts its answers to the type of user,
   and chooses its own tools for each request.
2. **Social impact:** helping citizens understand their legal position supports access to justice
   and makes them better prepared to work with lawyers and the courts.
3. **Scalability:** the same pipeline can be extended to tax tribunals, FBR rulings, SECP regulations
   and labour courts by adding domain-specific citation and provision patterns.
4. **Ethical position:** Wakeel AI is a research and educational tool. It does not provide legal
   advice, does not predict outcomes for individual cases, and always shows the sources behind its answers.
