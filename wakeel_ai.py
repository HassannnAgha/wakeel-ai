import os
import re
import json
from typing import Annotated, List, Literal, Optional, Tuple, Dict
from typing_extensions import TypedDict  # pydantic needs this flavour on py<3.12
from pathlib import Path
from dataclasses import dataclass
import pickle

# === NLP / ML ===
import pdfplumber
from tqdm import tqdm
import numpy as np

# === LangChain / LangGraph ===
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages

# === RAG Components ===
# chromadb and sentence_transformers are imported lazily (see get_collection /
# get_embed_model) so that chunking, tests and benchmarks can run without loading
# a 1.3 GB embedding model.
from rank_bm25 import BM25Okapi

# === Load Google API Key from .env ===
from dotenv import load_dotenv
load_dotenv()


def require_api_key() -> None:
    """Called before the LLM is constructed. Indexing and chunking do not need a key."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")


# =====================================================
# RAG SYSTEM COMPONENTS
# =====================================================

# ----------------------
# CONFIG
# ----------------------
PDF_DIR = Path("pdfs")
TEXT_DIR = Path("texts")
METADATA_DIR = Path("metadata")
BM25_DIR = Path("bm25_index")
MODEL_NAME = "BAAI/bge-large-en-v1.5"
# Gemini chat model. gemini-2.5-flash was retired for new API keys in 2026.
# Measured on the routing test set: 3.5-flash scored 9/9 at ~5.5s median, while
# 3.6-flash was rate-limited and 3-flash-preview scored 6/9 with 190s outliers.
LLM_MODEL = os.getenv("WAKEEL_MODEL", "gemini-3.5-flash")
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 300
TOP_K_CHUNKS = 10
HYBRID_ALPHA = 0.5
# Carry paragraphs shorter than min_chunk_size into the next chunk instead of
# dropping them. Set WAKEEL_LEGACY_CHUNKING=1 to restore the original behaviour.
MERGE_SHORT_PARAGRAPHS = os.getenv("WAKEEL_LEGACY_CHUNKING", "").strip().lower() not in {"1", "true", "yes"}

# Create directories
TEXT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
BM25_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------
# CHROMA SETUP (lazy)
# ----------------------
chroma_client = None
collection = None


def get_collection():
    """Open the Chroma collection on first use."""
    global chroma_client, collection
    if collection is None:
        import chromadb
        chroma_client = chromadb.PersistentClient(path="chroma_storeee")
        collection = chroma_client.get_or_create_collection(
            name="legal_cases",
            metadata={"hnsw:space": "cosine"}
        )
    return collection


# ----------------------
# EMBEDDING MODEL (lazy)
# ----------------------
embed_model = None


def get_embed_model():
    """Load the sentence-transformer on first use."""
    global embed_model
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        embed_model = SentenceTransformer(MODEL_NAME)
    return embed_model


def get_embeddings(texts: List[str]) -> List[List[float]]:
    return get_embed_model().encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()


# ----------------------
# TEXT EXTRACTION
# ----------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    txt_path = TEXT_DIR / (pdf_path.stem + ".txt")
    if txt_path.exists() and txt_path.stat().st_size > 0:
        return txt_path.read_text(encoding="utf-8", errors="ignore")

    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        print(f"[extract] failed {pdf_path}: {e}")
        return ""

    text = "\n".join(text_parts)
    txt_path.write_text(text, encoding="utf-8")
    return text


# ----------------------
# PAKISTANI COURT JUDGMENT CHUNKER
# ----------------------
@dataclass
class PakistaniLegalChunk:
    """Structured chunk for Pakistani court judgments"""
    text: str
    paragraph_number: str
    section_type: str
    case_citations: List[str]
    legal_provisions: List[str]
    chunk_position: int
    metadata: Dict


class PakistaniCourtJudgmentChunker:
    """Specialized chunker for Pakistani High Court and Supreme Court judgments"""
    
    def __init__(self,
                 target_chunk_size: int = 2000,
                 max_chunk_size: int = 3000,
                 min_chunk_size: int = 400,
                 merge_short_paragraphs: bool = True):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        # When True, a paragraph shorter than min_chunk_size is carried into the next
        # chunk instead of being discarded. False reproduces the original (lossy)
        # behaviour and exists so the two can be benchmarked against each other.
        self.merge_short_paragraphs = merge_short_paragraphs
        
        # Pakistani case citation patterns
        self.citation_patterns = [
            r'\b\d{4}\s+SCMR\s+\d+\b',
            r'\bPLD\s+\d{4}\s+[A-Z]+\s+\d+\b',
            r'\bAIR\s+\d{4}\s+[A-Z]+\s+\d+\b',
            r'\b\d{4}\s+[A-Z]{2,5}\s+\d+\b',
            r'\bPLJ\s+\d{4}\s+[A-Za-z.()]+\s+\d+\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s+(?:v\.|vs\.|versus)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}(?:\s+\(\d{4}\s+[A-Z]+\s+\d+\))?',
        ]
        
        # Legal provision patterns
        self.provision_patterns = [
            r'(?:Section|Article|Rule|Order)\s+\d+[A-Z]?(?:-[A-Z])?\s+(?:PPC|Cr\.?P\.?C\.?|CPC)',
            r'Article\s+\d+(?:\([a-z0-9]+\))?(?:\s+of\s+the\s+Constitution)?',
            r'Section\s+\d+[A-Z]?(?:\([a-z0-9]+\))?',
        ]
    
    def extract_case_metadata(self, text: str, filename: str) -> Dict:
        """Extract metadata from Pakistani court judgment header"""
        metadata = {
            "doc_id": filename,
            "court": "",
            "case_number": "",
            "parties": {"appellant": "", "respondent": ""},
            "judges": [],
            "counsel": {"appellant": [], "respondent": [], "state": []},
            "date_of_hearing": "",
            "judgment_type": ""
        }
        
        header = text[:3000]
        
        # Court name
        court_match = re.search(r'IN\s+THE\s+([A-Z\s]+COURT[A-Z\s]*)', header, re.IGNORECASE)
        if court_match:
            metadata["court"] = court_match.group(1).strip()
        
        # Case numbers
        case_patterns = [
            r'Criminal\s+Appeal\s+No\.?\s*(\d+)\s+of\s+(\d{4})',
            r'Civil\s+Appeal\s+No\.?\s*(\d+)\s+of\s+(\d{4})',
            r'Murder\s+Reference\s+No\.?\s*(\d+)\s+of\s+(\d{4})',
            r'Writ\s+Petition\s+No\.?\s*(\d+)\s+of\s+(\d{4})',
        ]
        for pattern in case_patterns:
            match = re.search(pattern, header, re.IGNORECASE)
            if match:
                metadata["judgment_type"] = re.search(r'[A-Za-z\s]+', pattern).group(0).strip()
                metadata["case_number"] = f"{match.group(1)}/{match.group(2)}"
                break
        
        # Party names
        party_match = re.search(r'\(([^)]+?)\s+(?:v\.|versus)\s+([^)]+?)\)', header, re.IGNORECASE)
        if party_match:
            metadata["parties"]["appellant"] = party_match.group(1).strip()
            metadata["parties"]["respondent"] = party_match.group(2).strip()
        
        # Judge names
        judge_section = re.search(r'(?:JUDGMENT|CORAM)[:\s]+(.*?)(?:Date of hearing|JUDGMENT|$)', 
                                 header, re.DOTALL | re.IGNORECASE)
        if judge_section:
            judge_text = judge_section.group(1)
            judge_names = re.findall(
                r'(?:HON\'BLE\s+)?(?:MR\.|MS\.|MRS\.)\s+(?:JUSTICE\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
                judge_text, re.IGNORECASE
            )
            metadata["judges"] = [name.strip() for name in judge_names]
        
        # Date of hearing
        hearing_date = re.search(r'Date\s+of\s+hearing:\s*(\d{2}\.\d{2}\.\d{4})', header)
        if hearing_date:
            metadata["date_of_hearing"] = hearing_date.group(1)
        
        return metadata
    
    def extract_citations(self, text: str) -> List[str]:
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        return list(set(citations))
    
    def extract_legal_provisions(self, text: str) -> List[str]:
        provisions = []
        for pattern in self.provision_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            provisions.extend(matches)
        return list(set(provisions))
    
    def identify_section_type(self, text: str, para_num: str) -> str:
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['facts of the case', 'epitomized facts', 'brief facts']):
            return 'facts'
        elif any(kw in text_lower for kw in ['contended', 'submissions', 'argued', 'counsel']):
            return 'arguments'
        elif any(kw in text_lower for kw in ['analysis', 'findings', 'observed', 'perusal of record']):
            return 'analysis'
        elif any(kw in text_lower for kw in ['conclusion', 'for the reasons', 'appeal is', 'acquitted', 'convicted']):
            return 'conclusion'
        elif para_num == "" or para_num == "0":
            return 'header'
        else:
            return 'analysis'
    
    def chunk_judgment(self, text: str, doc_metadata: Dict) -> List[PakistaniLegalChunk]:
        chunks = []
        lines = text.split('\n')
        
        current_chunk = []
        current_para_num = "0"
        current_para_last = "0"
        current_chunk_size = 0
        chunk_position = 0

        in_header = True
        header_end_markers = ['JUDGMENT', 'ORDER', 'BRIEF FACTS']

        def paragraph_label(first: str, last: str) -> str:
            """A chunk covering paragraphs 4 through 6 is labelled "4-6"."""
            return f"{first}-{last}" if last and last != first else first

        def build_chunk(chunk_lines, first, last, position):
            body = '\n'.join(chunk_lines).strip()
            label = paragraph_label(first, last)
            return PakistaniLegalChunk(
                text=body,
                paragraph_number=label,
                section_type=self.identify_section_type(body, label),
                case_citations=self.extract_citations(body),
                legal_provisions=self.extract_legal_provisions(body),
                chunk_position=position,
                metadata=doc_metadata
            )

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            if in_header and any(marker in line.upper() for marker in header_end_markers):
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if len(chunk_text) > 100:
                        chunks.append(PakistaniLegalChunk(
                            text=chunk_text,
                            paragraph_number="0",
                            section_type='header',
                            case_citations=[],
                            legal_provisions=[],
                            chunk_position=chunk_position,
                            metadata=doc_metadata
                        ))
                        chunk_position += 1
                current_chunk = []
                current_chunk_size = 0
                in_header = False
            
            para_match = re.match(r'^(\d+)\.\s+', line)
            
            if para_match and not in_header:
                if current_chunk and current_chunk_size > self.min_chunk_size:
                    chunks.append(build_chunk(current_chunk, current_para_num,
                                              current_para_last, chunk_position))
                    chunk_position += 1
                    current_para_num = para_match.group(1)
                    current_para_last = current_para_num
                    current_chunk = [line]
                    current_chunk_size = len(line)
                elif current_chunk and self.merge_short_paragraphs:
                    # Paragraph is too short to stand alone. Carry it forward into the
                    # next chunk instead of discarding it (the original dropped it).
                    current_chunk.append(line)
                    current_chunk_size += len(line)
                    if current_para_num in ("", "0"):
                        current_para_num = para_match.group(1)
                    current_para_last = para_match.group(1)
                else:
                    current_para_num = para_match.group(1)
                    current_para_last = current_para_num
                    current_chunk = [line]
                    current_chunk_size = len(line)
                i += 1

            elif current_chunk_size + len(line) > self.max_chunk_size:
                chunk_text = '\n'.join(current_chunk).strip()
                # The incoming line alone exceeds the budget, so the buffer cannot be
                # carried forward. Emit it even if short rather than losing it.
                if chunk_text and (len(chunk_text) > self.min_chunk_size
                                   or self.merge_short_paragraphs):
                    chunks.append(build_chunk(current_chunk, current_para_num,
                                              current_para_last, chunk_position))
                    chunk_position += 1

                current_chunk = [line]
                current_chunk_size = len(line)
                i += 1
            else:
                current_chunk.append(line)
                current_chunk_size += len(line)
                i += 1

        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            # Trailing text is the conclusion of the judgment. Never drop it.
            if chunk_text and (current_chunk_size > self.min_chunk_size
                               or self.merge_short_paragraphs):
                chunks.append(build_chunk(current_chunk, current_para_num,
                                          current_para_last, chunk_position))

        return chunks
    
    def add_contextual_metadata(self, chunks: List[PakistaniLegalChunk]) -> List[Dict]:
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            context_parts = []
            
            if chunk.metadata.get('case_number'):
                context_parts.append(f"[Case: {chunk.metadata['case_number']}]")
            
            if chunk.metadata.get('parties', {}).get('appellant'):
                parties = chunk.metadata['parties']
                context_parts.append(f"[{parties['appellant']} v. {parties['respondent']}]")
            
            if chunk.paragraph_number != "0":
                context_parts.append(f"[Paragraph {chunk.paragraph_number}]")
            
            if chunk.section_type:
                context_parts.append(f"[Section: {chunk.section_type.title()}]")
            
            context_prefix = ' '.join(context_parts)
            if context_prefix:
                context_prefix += '\n\n'
            
            citation_context = ""
            if chunk.case_citations:
                top_citations = chunk.case_citations[:3]
                citation_context = f"[Key Citations: {', '.join(top_citations)}]\n\n"
            
            prev_context = ""
            if i > 0:
                prev_text = chunks[i-1].text
                if len(prev_text) > 150:
                    prev_context = f"[Previous: ...{prev_text[-150:]}]\n\n"
            
            enhanced_text = context_prefix + citation_context + prev_context + chunk.text
            
            enhanced_chunks.append({
                'text': enhanced_text,
                'raw_text': chunk.text,
                'paragraph_number': chunk.paragraph_number,
                'section_type': chunk.section_type,
                'case_citations': chunk.case_citations,
                'legal_provisions': chunk.legal_provisions,
                'chunk_position': chunk.chunk_position,
                'total_chunks': len(chunks),
                'metadata': chunk.metadata
            })
        
        return enhanced_chunks


# ----------------------
# ENHANCED CHUNKING FUNCTION
# ----------------------
def chunk_text(text: str, metadata: Dict,
               merge_short_paragraphs: bool = MERGE_SHORT_PARAGRAPHS) -> List[Dict]:
    """
    Intelligent chunking for Pakistani court judgments.
    Falls back to basic chunking if not recognized as a judgment.

    merge_short_paragraphs=False restores the original behaviour, which discarded
    any paragraph shorter than min_chunk_size. Kept so the two can be compared;
    set WAKEEL_LEGACY_CHUNKING=1 to select it.
    """
    try:
        chunker = PakistaniCourtJudgmentChunker(
            target_chunk_size=CHUNK_SIZE_CHARS,
            max_chunk_size=3000,
            min_chunk_size=400,
            merge_short_paragraphs=merge_short_paragraphs
        )
        
        doc_metadata = chunker.extract_case_metadata(text, metadata['doc_id'])
        doc_metadata.update(metadata)
        
        is_judgment = any(marker in text[:2000].upper() for marker in 
                         ['LAHORE HIGH COURT', 'SUPREME COURT', 'SINDH HIGH COURT', 
                          'ISLAMABAD HIGH COURT', 'JUDGMENT', 'CRIMINAL APPEAL'])
        
        if is_judgment:
            chunks = chunker.chunk_judgment(text, doc_metadata)
            enhanced = chunker.add_contextual_metadata(chunks)
            
            result = []
            for i, chunk_data in enumerate(enhanced):
                result.append({
                    "chunk_id": i,
                    "text": chunk_data['text'],
                    "raw_text": chunk_data['raw_text'],
                    "doc_id": doc_metadata["doc_id"],
                    "case_no": doc_metadata.get("case_number", ""),
                    "title": f"{doc_metadata['parties']['appellant']} v {doc_metadata['parties']['respondent']}" if doc_metadata['parties']['appellant'] else "",
                    "court": doc_metadata.get("court", ""),
                    "judges": doc_metadata.get("judges", []),
                    "parties": doc_metadata.get("parties", {}),
                    "paragraph_number": chunk_data['paragraph_number'],
                    "section_type": chunk_data['section_type'],
                    "citations": chunk_data['case_citations'],
                    "provisions": chunk_data['legal_provisions'],
                })
            return result
    except Exception as e:
        print(f"[warn] Pakistani chunker failed, falling back to basic: {e}")
    
    # Fallback to basic chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    result = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "chunk_id": i,
            "text": chunk,
            "raw_text": chunk,
            "doc_id": metadata["doc_id"],
            "case_no": metadata.get("case_no", ""),
            "title": metadata.get("title", ""),
            "court": "",
            "judges": [],
            "parties": {},
            "paragraph_number": "",
            "section_type": "",
            "citations": [],
            "provisions": [],
        }
        result.append(chunk_data)
    return result


# ----------------------
# BM25 INDEX
# ----------------------
class BM25Index:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.bm25 = None
        self.chunks_data = []
        self.tokenized_corpus = []
        
    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def build(self, chunks: List[Dict]):
        self.chunks_data = chunks
        self.tokenized_corpus = [self.tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.save()
    
    def save(self):
        with open(self.index_path / "bm25.pkl", "wb") as f:
            pickle.dump((self.bm25, self.chunks_data, self.tokenized_corpus), f)
    
    def load(self):
        index_file = self.index_path / "bm25.pkl"
        if index_file.exists():
            with open(index_file, "rb") as f:
                self.bm25, self.chunks_data, self.tokenized_corpus = pickle.load(f)
            return True
        return False
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.bm25:
            return []
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


# ----------------------
# INDEX INTO CHROMA + BM25
# ----------------------
def index_pdfs():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Checking {len(pdf_files)} PDFs for indexing...")

    existing_ids = set()
    try:
        existing_data = get_collection().get(include=["metadatas"], limit=None)
        if existing_data and "metadatas" in existing_data:
            for meta in existing_data["metadatas"]:
                if meta and "doc_id" in meta:
                    existing_ids.add(meta["doc_id"])
    except Exception as e:
        print(f"[warn] Could not fetch existing docs: {e}")

    all_chunks = []
    new_docs_processed = False

    for pdf_path in tqdm(pdf_files):
        doc_id = pdf_path.stem

        if doc_id in existing_ids:
            metadata_file = METADATA_DIR / f"{doc_id}.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    saved_chunks = json.load(f)
                    all_chunks.extend(saved_chunks)
            continue

        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"Skipping {doc_id} (empty or unreadable text)")
            continue

        metadata = {"doc_id": doc_id}
        chunks = chunk_text(text, metadata)
        
        if not chunks:
            print(f"Skipping {doc_id} (no chunks generated)")
            continue

        metadata_file = METADATA_DIR / f"{doc_id}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        chunk_texts = [c["text"] for c in chunks]
        vectors = get_embeddings(chunk_texts)

        ids = [f"{doc_id}_{c['chunk_id']}" for c in chunks]
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "case_no": c["case_no"],
                "title": c["title"],
                "court": c.get("court", ""),
                "judges": json.dumps(c.get("judges", [])),
                "parties": json.dumps(c.get("parties", {})),
                "paragraph_number": c.get("paragraph_number", ""),
                "section_type": c.get("section_type", ""),
                "citations": json.dumps(c.get("citations", [])),
                "provisions": json.dumps(c.get("provisions", [])),
            } 
            for c in chunks
        ]

        get_collection().upsert(
            documents=chunk_texts,
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas
        )
        
        all_chunks.extend(chunks)
        new_docs_processed = True
        
        case_info = f"{chunks[0].get('case_no', 'N/A')}"
        court_info = chunks[0].get('court', 'Unknown')
        print(f"Indexed {doc_id} | {len(chunks)} chunks | {court_info} | Case: {case_info}")

    print("\nBuilding BM25 keyword index...")
    bm25_index = BM25Index(BM25_DIR)
    
    if not all_chunks:
        for metadata_file in METADATA_DIR.glob("*.json"):
            with open(metadata_file, "r", encoding="utf-8") as f:
                all_chunks.extend(json.load(f))
    
    if all_chunks:
        bm25_index.build(all_chunks)
        print(f"BM25 index built with {len(all_chunks)} chunks")
    
    print("\nIndexing completed (only new PDFs processed).")


# ----------------------
# HYBRID SEARCH
# ----------------------
def search_hybrid(query: str, top_k=TOP_K_CHUNKS, alpha=HYBRID_ALPHA):
    """Hybrid search combining semantic and keyword-based retrieval"""
    q_vec = get_embeddings([query])[0]
    semantic_results = get_collection().query(
        query_embeddings=[q_vec],
        n_results=top_k * 2,
        include=["documents", "metadatas", "distances"]
    )
    
    semantic_scores = {}
    for doc, meta, dist in zip(
        semantic_results["documents"][0],
        semantic_results["metadatas"][0],
        semantic_results["distances"][0]
    ):
        chunk_id = f"{meta['doc_id']}_{doc[:50]}"
        semantic_scores[chunk_id] = {
            "score": 1 - dist,
            "doc": doc,
            "meta": meta
        }
    
    bm25_index = BM25Index(BM25_DIR)
    if not bm25_index.load():
        print("BM25 index not found. Using semantic search only.")
        alpha = 1.0
    
    keyword_scores = {}
    if alpha < 1.0 and bm25_index.bm25:
        bm25_results = bm25_index.search(query, top_k=top_k * 2)
        
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results)
            if max_bm25_score > 0:
                for idx, score in bm25_results:
                    chunk_data = bm25_index.chunks_data[idx]
                    chunk_id = f"{chunk_data['doc_id']}_{chunk_data['text'][:50]}"
                    keyword_scores[chunk_id] = {
                        "score": score / max_bm25_score,
                        "doc": chunk_data["text"],
                        "meta": {
                            "doc_id": chunk_data["doc_id"],
                            "case_no": chunk_data["case_no"],
                            "title": chunk_data["title"]
                        }
                    }
    
    combined_scores = {}
    all_chunk_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    
    for chunk_id in all_chunk_ids:
        sem_score = semantic_scores.get(chunk_id, {}).get("score", 0)
        key_score = keyword_scores.get(chunk_id, {}).get("score", 0)
        
        hybrid_score = alpha * sem_score + (1 - alpha) * key_score
        
        data = semantic_scores.get(chunk_id) or keyword_scores.get(chunk_id)
        
        combined_scores[chunk_id] = {
            "score": hybrid_score,
            "semantic_score": sem_score,
            "keyword_score": key_score,
            "doc": data["doc"],
            "meta": data["meta"]
        }
    
    ranked = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    
    return ranked


# =====================================================
# AGENT SYSTEM WITH INTEGRATED RAG (LangGraph)
# =====================================================

# Set WAKEEL_VERBOSE=1 to print routing decisions as the graph runs.
VERBOSE = os.getenv("WAKEEL_VERBOSE", "").strip().lower() in {"1", "true", "yes"}


def _trace(message: str) -> None:
    if VERBOSE:
        print(f"[graph] {message}")


def message_text(response) -> str:
    """Plain text from a chat model response.

    Gemini 3.x returns `content` as a list of content blocks
    ([{'type': 'text', 'text': ...}, ...]); Gemini 2.x returned a plain string.
    Reading `.content` directly therefore yields a list on newer models, so always
    go through here.
    """
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):
        try:
            value = text()
            if isinstance(value, str):
                return value
        except Exception:
            pass

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# ----------------------
# SEARCH RESULT FORMATTING
# ----------------------
def format_search_results(results: List[Dict]) -> str:
    """Render hybrid search hits as readable context for the LLM."""
    if not results:
        return "No relevant documents found for your query. Try rephrasing or using different keywords."

    formatted_output = [f"Found {len(results)} relevant legal documents:\n"]

    for i, hit in enumerate(results, 1):
        formatted_output.append(f"\n{'='*60}")
        formatted_output.append(f"Result #{i} (Relevance: {hit['score']:.3f})")
        formatted_output.append(f"{'='*60}")

        meta = hit["meta"]
        formatted_output.append(f"Document ID: {meta.get('doc_id', 'N/A')}")
        formatted_output.append(f"Case Number: {meta.get('case_no', 'N/A')}")
        formatted_output.append(f"Court: {meta.get('court', 'N/A')}")
        formatted_output.append(f"Title: {meta.get('title', 'N/A')}")

        judges = meta.get("judges", "[]")
        if isinstance(judges, str):
            try:
                judges = json.loads(judges)
            except Exception:
                judges = []
        if judges:
            formatted_output.append(f"Judges: {', '.join(judges)}")

        parties = meta.get("parties", "{}")
        if isinstance(parties, str):
            try:
                parties = json.loads(parties)
            except Exception:
                parties = {}
        if parties and parties.get("appellant"):
            formatted_output.append(f"Parties: {parties['appellant']} v. {parties.get('respondent', '')}")

        citations = meta.get("citations", "[]")
        if isinstance(citations, str):
            try:
                citations = json.loads(citations)
            except Exception:
                citations = []
        if citations:
            formatted_output.append(f"Key Citations: {', '.join(citations[:3])}")

        provisions = meta.get("provisions", "[]")
        if isinstance(provisions, str):
            try:
                provisions = json.loads(provisions)
            except Exception:
                provisions = []
        if provisions:
            formatted_output.append(f"Legal Provisions: {', '.join(provisions[:3])}")

        formatted_output.append(f"Section Type: {meta.get('section_type', 'N/A')}")
        formatted_output.append(f"Paragraph: {meta.get('paragraph_number', 'N/A')}")

        formatted_output.append("\nContent:")
        formatted_output.append(f"{hit['doc'][:800]}...")
        formatted_output.append("")

    return "\n".join(formatted_output)


def hybrid_rag_search(query: str, top_k: int = TOP_K_CHUNKS, alpha: float = HYBRID_ALPHA) -> str:
    """Hybrid search returning formatted context. Never raises."""
    try:
        return format_search_results(search_hybrid(query, top_k=top_k, alpha=alpha))
    except Exception as e:
        return f"Error performing RAG search: {str(e)}"


def index_documents_tool(action: str = "index") -> str:
    """Index new PDFs from the pdfs/ directory. Returns a confirmation message."""
    try:
        if action.lower() in ['index', 'reindex']:
            index_pdfs()
            return ("Successfully indexed all PDF documents in the 'pdfs' directory. "
                    "Documents are now searchable.")
        return "Invalid action. Use 'index' or 'reindex' to process documents."
    except Exception as e:
        return f"Error indexing documents: {str(e)}"


def sentiment_tool_func(company_name: str) -> str:
    """Placeholder for sentiment analysis tool"""
    return f"Sentiment analysis for {company_name} not implemented yet."


# ----------------------
# PROMPTS
# ----------------------
CLASSIFICATION_PROMPT = PromptTemplate.from_template(
    """
ROLE:
 - You are a Legal AI assistant. You help users retrieve, summarize, and compare legal documents and cases.

INSTRUCTION
    - Only answer queries with an actual legal/regulatory/compliance context.
    - Avoid making non-legal queries appear legal.
    - Consider follow-up questions: If the user query seems related to previous conversation or uses
      pronouns/context ("What about that case?", "And the previous judgment?"), use the chat history
      to understand it.
    - If the query is unrelated to law, politely ask the user for clarification.

TASK:
    - Classify the input according to the rules below.
    - Resolve follow-up references using the chat history.

CLASSIFICATION RULES:
    1) If the query is NOT related to legal/regulatory/compliance matters, or is too broad to act on,
       set route = "REJECT" and put a short, polite message in `clarification`.
       EXAMPLES:
         - "What's the weather in New York?"
           -> REJECT, clarification: "Sorry, I can't answer this query as it is not legal related.
              Please provide more context or a legal question."
         - "Summarize legal docs"
           -> REJECT, clarification: "Please specify which legal documents or cases you want summarized."
         - "Compare these cases"
           -> REJECT, clarification: "Please provide the names or references of the cases you want to compare."

    2) If related to legal/regulatory/compliance, set route to one of:
         GQ = General legal query (standalone, or a follow-up to a previous general query)
         SQ = Summarizing request (summary of a law, regulation, case, or document)
         CQ = Compare request (compare two or more legal documents, laws, or regulations)
       and put the restated query in `restated_query`.
       For CQ also fill `compare_a` and `compare_b` with the two things being compared.

EXAMPLES:
    "Retrieve past corruption cases in Pakistan"
        -> GQ, restated_query: "Retrieve past corruption cases in Pakistan"
    "Summarize the Lahore High Court judgment on property dispute"
        -> SQ, restated_query: "Summarize the Lahore High Court judgment on property dispute"
    "Compare GDPR with CCPA"
        -> CQ, restated_query: "Compare GDPR with CCPA", compare_a: "GDPR", compare_b: "CCPA"
    "What were the penalties in last anti-corruption case?"
        -> GQ, restated_query: "Find previous anti-corruption cases and penalties"
    "Has Person A been involved in any property cases?"
        -> GQ, restated_query: "Find property cases involving Person A"
    "Summarize the judgment involving Person B in tax evasion case"
        -> SQ, restated_query: "Summarize the tax evasion judgment involving Person B"
    "Compare corruption rulings for Person A and Person C"
        -> CQ, restated_query: "Compare corruption rulings involving Person A and Person C",
           compare_a: "corruption ruling involving Person A", compare_b: "corruption ruling involving Person C"
    "Compare Lahore High Court and Supreme Court decisions on property disputes"
        -> CQ, restated_query: "Compare Lahore High Court vs Supreme Court property dispute decisions",
           compare_a: "Lahore High Court property dispute decision",
           compare_b: "Supreme Court property dispute decision"

EXAMPLES (Follow-up / Context-aware queries):
    "And the previous corruption case?"
        -> GQ, restated_query: "Retrieve details about the previous corruption case mentioned in chat history"
    "What was the verdict in that case?"
        -> GQ, restated_query: "Find the verdict of the corruption case mentioned earlier in the chat"
    "Summarize it for me"
        -> SQ, restated_query: "Summarize the previously discussed judgment or legal document"
    "Compare this with the 2020 property case"
        -> CQ, restated_query: "Compare the currently discussed case with the 2020 property case"
    "Any penalties mentioned there?"
        -> GQ, restated_query: "Retrieve penalties mentioned in the previous legal case discussed in chat history"
    "Who were the defendants again?"
        -> GQ, restated_query: "List defendants involved in the previously discussed case"
    "And the Lahore High Court judgment?"
        -> SQ, restated_query: "Summarize the Lahore High Court judgment mentioned in prior conversation"

RULES:
    - Keep the restated query close to what the user asked (fix only obvious grammar issues).
    - Use the chat history to clarify references in follow-up questions.

Chat history:
{history}

Input Query:
{query}"""
)

# Legacy tag-based classification prompt, used only if structured output fails.
CLASSIFICATION_PROMPT_FALLBACK = PromptTemplate.from_template(
    """Classify the legal query below.

Reply with exactly one of:
  <answer><valid>[GQ] restated query</valid></answer>   (general legal query)
  <answer><valid>[SQ] restated query</valid></answer>   (summarize request)
  <answer><valid>[CQ] restated query</valid></answer>   (compare request)
  <answer>clarifying message</answer>                   (not legal, or too broad)

No extra text outside the tags.

Chat history:
{history}

Input Query:
{query}"""
)

ANSWER_PROMPT = PromptTemplate.from_template(
    """
You are a Legal AI assistant specializing in Pakistani case law. Answer the user's question using
only the retrieved excerpts below.

Instructions:
- Ground every statement in the excerpts. Do not invent cases, citations, judges or provisions.
- Cite the case number, court and paragraph number when you rely on an excerpt.
- If the excerpts do not answer the question, say so plainly and suggest how to narrow the search.
- Write for a reader who is not a lawyer, but keep the legal meaning exact.

Question:
{query}

Retrieved excerpts:
{context}
"""
)

COMPARE_PROMPT = PromptTemplate.from_template(
    """
Role:
You are a Legal AI assistant specializing in Pakistani case law. You help users analyze, summarize, and compare legal cases, judgments, and statutes. Focus on reasoning, precedents, verdicts, penalties, and procedural context.

Task:
Analyze the two legal cases/documents provided and identify all similarities and differences between them. Focus on legal reasoning, outcomes, and relevant details.

Instructions:
- Treat content as similar if legal reasoning, verdicts, cited statutes, or precedents are equivalent.
- Treat content as different if reasoning, verdicts, penalties, precedents, parties, or procedural context differ.
- Include the parties involved, the court, jurisdiction, and case dates when relevant.
- Highlight subtle differences in interpretation, application, or scope of law.
- Include maximum possible similarities and differences without making assumptions.
- Always reference the specific parts of the documents when possible.

Examples:
- Similarity: Both cases applied Section 302 of the Pakistan Penal Code to determine criminal liability.
- Difference: Case 2 imposed a fine of PKR 500,000, while Case 1 imposed imprisonment of 3 years for the same offense.
- Similarity: Both judgments cited previous decisions on procedural requirements for appeals.
- Difference: Case 1 considered mitigating circumstances, whereas Case 2 did not.

Output Format:
- Divide output into two sections: SIMILARITIES and DIFFERENCES
- Each entry should include:
    1. Heading: Short title summarizing the topic (e.g., "Penalties", "Precedent", "Legal Reasoning")
    2. Explanation: Brief explanation of the similarity or difference
    3. doc1_line: Quote or reference from document 1
    4. doc2_line: Quote or reference from document 2
- Return the output directly, without introductory text or commentary.

Analyze:
- Read both documents carefully.
- Extract every relevant similarity and difference according to the above rules.
- Avoid assumptions; rely only on what is explicitly or logically supported in the documents.
- Return the structured but readable output directly.

DOCS are:
doc1
{doc1}

doc2
{doc2}
"""
)

SUMMARIZE_PROMPT = PromptTemplate.from_template(
    """
You are a legal analyst and educator. Your task is to summarize the following legal document in **clear, simple, layman-friendly language** while preserving the **exact legal meaning and context**.

Instructions:
- Simplify complex legal jargon without losing key obligations, rights, or compliance requirements.
- Keep the structure logical and easy to read.
- Retain any critical definitions, terms, deadlines, or requirements.
- Highlight practical implications for a layperson.
- Avoid skipping any sections with important legal content.

Output Format:
- Use numbered points or short paragraphs.
- Each point should convey one main idea.
- Include any deadlines, obligations, or conditions explicitly.

Document to summarize:
{text}
"""
)


# ----------------------
# GRAPH STATE & SCHEMA
# ----------------------
class QueryClassification(BaseModel):
    """Structured result of the intent classifier."""

    route: Literal["GQ", "SQ", "CQ", "REJECT"] = Field(
        description="GQ = general legal query, SQ = summarize, CQ = compare, REJECT = not legal or too broad"
    )
    restated_query: str = Field(default="", description="The query restated for retrieval")
    compare_a: str = Field(default="", description="First item to compare (CQ only)")
    compare_b: str = Field(default="", description="Second item to compare (CQ only)")
    clarification: str = Field(default="", description="Message shown to the user when route is REJECT")


class WakeelState(TypedDict):
    """State carried through the graph."""

    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    route: str
    restated_query: str
    compare_a: str
    compare_b: str
    context: str
    doc1: str
    doc2: str
    answer: str


def _render_history(messages: List[AnyMessage], limit: int = 6) -> str:
    """Render the last few turns for the classifier, excluding the current query."""
    past = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))][:-1]
    if not past:
        return "(no previous conversation)"
    lines = []
    for m in past[-limit:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        text = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {text[:500]}")
    return "\n".join(lines)


#: Errors where retrying the same call cannot help (quota, auth, missing model).
#: Falling back to a second call on these just burns quota twice.
_UNRETRYABLE = ("RESOURCE_EXHAUSTED", "429", "PERMISSION_DENIED", "UNAUTHENTICATED",
                "401", "403", "NOT_FOUND", "404", "API key")


def is_unretryable_api_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in _UNRETRYABLE)


def _api_error_message(exc: Exception) -> str:
    """A short, human-readable reason instead of a wall of JSON."""
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return ("The Gemini API quota for this key is exhausted. The free tier allows "
                "a limited number of requests per day; wait for the quota to reset or "
                "enable billing. (Set WAKEEL_MODEL to try a different model.)")
    if any(m in text for m in ("PERMISSION_DENIED", "UNAUTHENTICATED", "401", "403", "API key")):
        return "The Gemini API rejected the API key. Check GOOGLE_API_KEY in your .env file."
    if "NOT_FOUND" in text or "404" in text:
        return (f"The configured model is not available to this API key. "
                f"Set WAKEEL_MODEL to a model you can access (current: {LLM_MODEL}).")
    return f"The language model call failed: {text[:300]}"


def parse_classification_response(response: str) -> Tuple[Optional[str], str]:
    """Parse the legacy <answer><valid>[GQ] ...</valid></answer> format."""
    match = re.search(r"<valid>\[(GQ|SQ|CQ|SA)\]\s*(.*?)</valid>", response, re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    payload_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return None, payload_match.group(1).strip() if payload_match else "Invalid classification response."


# ----------------------
# WAKEEL AGENT (LangGraph)
# ----------------------
class WakeelAgent:
    """Legal research agent built as an explicit LangGraph state machine.

    Flow:
        START -> classify -+-> clarify                      -> END
                           +-> retrieve -> answer           -> END   (GQ)
                           +-> retrieve -> summarize        -> END   (SQ)
                           +-> retrieve_pair -> compare     -> END   (CQ)
    """

    def __init__(self, llm, checkpointer=None, thread_id: str = "wakeel-cli"):
        self.llm = llm
        self.thread_id = thread_id
        try:
            self.classifier = llm.with_structured_output(QueryClassification)
        except Exception as e:
            print(f"[warn] structured output unavailable, using tag parsing: {e}")
            self.classifier = None
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.graph = self._build_graph().compile(checkpointer=self.checkpointer)

    # ===== Helper methods (kept from the original agent API) =====

    def is_legal_document(self, text: str) -> bool:
        """Cheap yes/no check that a piece of text is Pakistani legal content.

        Works on a short query or on the extracted text of an uploaded document,
        so it can gate document upload (an FIR, legal notice or court order)
        before indexing. Query routing itself is handled by classify_node.
        """
        try:
            classification_prompt = f"""
            Text:
            \"\"\"{text[:6000]}\"\"\"

ROLE:
You are a legal analyst who identifies Pakistani legal, regulatory and judicial content.

TASK:
Decide whether the text above is, or is derived from, a Pakistani legal document or a
question about Pakistani law.

Criteria (any one is enough):
- A judgment, order, decree or writ of a Pakistani court (Supreme Court, a High Court,
  or a district or sessions court)
- An FIR, charge sheet, bail application, legal notice, plaint, written statement,
  power of attorney or affidavit
- Cites Pakistani case law (SCMR, PLD, PLJ, CLC, YLR, MLD, AIR)
- References Pakistani statutes or provisions: PPC, Cr.P.C., CPC, Qanun-e-Shahadat,
  the Constitution of Pakistan, or a numbered Section, Article, Order or Rule
- Uses litigation vocabulary such as appellant, respondent, petitioner, accused,
  complainant, counsel, or names a presiding judge
- Sets out rights, obligations, penalties, sentences or procedural requirements
  under Pakistani law

Answer no for general news, commentary, academic writing or commercial documents that
merely mention the law in passing.

Output Instructions:
- Respond with **only** one word: yes or no
- Do **not** explain, elaborate, or add any other text
- The output must be **only** one of: yes or no
- No punctuation. No formatting. All lowercase.
            """
            raw_result = message_text(self.llm.invoke(classification_prompt))
            result = raw_result.strip().lower()
            words = re.findall(r"\b\w+\b", result)
            return "yes" in words
        except Exception:
            return False

    def compare_docs(self, doc_input: str) -> str:
        """Compare two documents given as "doc1_id, doc2_id"."""
        try:
            doc_ids = [d.strip() for d in doc_input.split(',')]
            if len(doc_ids) != 2:
                return "Please provide exactly two document IDs separated by comma."

            doc1_results = search_hybrid(doc_ids[0], top_k=1)
            doc2_results = search_hybrid(doc_ids[1], top_k=1)

            if not doc1_results or not doc2_results:
                return "Could not find one or both documents. Please check document IDs."

            return message_text(self.llm.invoke(COMPARE_PROMPT.format(
                doc1=doc1_results[0]['doc'], doc2=doc2_results[0]['doc'])))
        except Exception as e:
            return f"Error comparing documents: {str(e)}"

    def summarize_text(self, text: str) -> str:
        """Summarize the given text."""
        try:
            return message_text(self.llm.invoke(SUMMARIZE_PROMPT.format(text=text)))
        except Exception as e:
            return f"Error summarizing text: {str(e)}"

    # ===== Nodes =====

    def classify_node(self, state: WakeelState) -> Dict:
        query = state["query"]
        history = _render_history(state["messages"])

        if self.classifier is not None:
            try:
                result = self.classifier.invoke(
                    CLASSIFICATION_PROMPT.format(query=query, history=history)
                )
                if result is not None:
                    _trace(f"classified as {result.route}")
                    return {
                        "route": result.route,
                        "restated_query": result.restated_query or query,
                        "compare_a": result.compare_a,
                        "compare_b": result.compare_b,
                        "answer": result.clarification if result.route == "REJECT" else "",
                    }
            except Exception as e:
                # Quota/auth/model errors will fail the same way again, so do not
                # spend a second call on them.
                if is_unretryable_api_error(e):
                    _trace(f"classification failed, not retryable: {e}")
                    return {"route": "ERROR", "answer": _api_error_message(e)}
                _trace(f"structured classification failed ({e}); falling back to tag parsing")

        # Fallback: legacy tag-based classification.
        try:
            raw = message_text(self.llm.invoke(
                CLASSIFICATION_PROMPT_FALLBACK.format(query=query, history=history)
            ))
        except Exception as e:
            # An API failure is not a judgement about the query. Keep it distinct
            # from REJECT so callers do not report "not a legal question".
            return {"route": "ERROR", "answer": _api_error_message(e)}

        query_type, payload = parse_classification_response(raw)
        if not query_type:
            _trace("classified as REJECT (fallback)")
            return {"route": "REJECT", "answer": payload}

        route = "GQ" if query_type == "SA" else query_type
        _trace(f"classified as {route} (fallback)")
        return {
            "route": route,
            "restated_query": payload or query,
            "compare_a": "",
            "compare_b": "",
            "answer": "",
        }

    def clarify_node(self, state: WakeelState) -> Dict:
        answer = state.get("answer") or (
            "Sorry, I can't answer this query as it is not legal related. "
            "Please provide more context or a legal question."
        )
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    def retrieve_node(self, state: WakeelState) -> Dict:
        query = state.get("restated_query") or state["query"]
        _trace(f"retrieving for: {query}")
        return {"context": hybrid_rag_search(query)}

    def retrieve_pair_node(self, state: WakeelState) -> Dict:
        """Retrieve both sides of a comparison."""
        query = state.get("restated_query") or state["query"]
        side_a = state.get("compare_a") or query
        side_b = state.get("compare_b") or query
        _trace(f"retrieving pair: {side_a!r} vs {side_b!r}")

        try:
            docs_a = search_hybrid(side_a, top_k=3)
            docs_b = search_hybrid(side_b, top_k=3)
        except Exception as e:
            return {"context": f"Error retrieving documents for comparison: {str(e)}"}

        if not docs_a or not docs_b:
            return {"context": ""}

        return {
            "context": "ok",
            "doc1": format_search_results(docs_a),
            "doc2": format_search_results(docs_b),
        }

    def answer_node(self, state: WakeelState) -> Dict:
        try:
            answer = message_text(self.llm.invoke(
                ANSWER_PROMPT.format(
                    query=state.get("restated_query") or state["query"],
                    context=state["context"],
                )
            ))
        except Exception as e:
            answer = _api_error_message(e)
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    def summarize_node(self, state: WakeelState) -> Dict:
        try:
            answer = message_text(self.llm.invoke(SUMMARIZE_PROMPT.format(text=state["context"])))
        except Exception as e:
            answer = _api_error_message(e)
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    def compare_node(self, state: WakeelState) -> Dict:
        context = state.get("context") or ""
        doc1, doc2 = state.get("doc1") or "", state.get("doc2") or ""

        if not context:
            answer = "Could not find one or both documents. Please check the case names or document IDs."
            return {"answer": answer, "messages": [AIMessage(content=answer)]}

        if not doc1 or not doc2:
            # retrieve_pair_node put an error message in context instead of documents
            return {"answer": context, "messages": [AIMessage(content=context)]}

        try:
            answer = message_text(self.llm.invoke(COMPARE_PROMPT.format(doc1=doc1, doc2=doc2)))
        except Exception as e:
            answer = _api_error_message(e)
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    # ===== Edges =====

    @staticmethod
    def route_after_classify(state: WakeelState) -> str:
        route = state.get("route", "REJECT")
        if route == "CQ":
            return "retrieve_pair"
        if route in ("GQ", "SQ"):
            return "retrieve"
        return "clarify"

    @staticmethod
    def route_after_retrieve(state: WakeelState) -> str:
        return "summarize" if state.get("route") == "SQ" else "answer"

    # ===== Graph =====

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(WakeelState)

        builder.add_node("classify", self.classify_node)
        builder.add_node("clarify", self.clarify_node)
        builder.add_node("retrieve", self.retrieve_node)
        builder.add_node("retrieve_pair", self.retrieve_pair_node)
        builder.add_node("answer", self.answer_node)
        builder.add_node("summarize", self.summarize_node)
        builder.add_node("compare", self.compare_node)

        builder.add_edge(START, "classify")
        builder.add_conditional_edges(
            "classify",
            self.route_after_classify,
            {"retrieve": "retrieve", "retrieve_pair": "retrieve_pair", "clarify": "clarify"},
        )
        builder.add_conditional_edges(
            "retrieve",
            self.route_after_retrieve,
            {"answer": "answer", "summarize": "summarize"},
        )
        builder.add_edge("retrieve_pair", "compare")
        builder.add_edge("clarify", END)
        builder.add_edge("answer", END)
        builder.add_edge("summarize", END)
        builder.add_edge("compare", END)

        return builder

    # ===== Public API =====

    def run(self, query: str, thread_id: Optional[str] = None) -> str:
        """Run one turn. Conversation memory is keyed by thread_id."""
        return self.run_detailed(query, thread_id)["answer"]

    def run_detailed(self, query: str, thread_id: Optional[str] = None) -> Dict:
        """Run one turn and report which branch of the graph handled it."""
        config = {"configurable": {"thread_id": thread_id or self.thread_id}}
        try:
            final_state = self.graph.invoke(
                {
                    "messages": [HumanMessage(content=query)],
                    "query": query,
                    "route": "",
                    "restated_query": "",
                    "compare_a": "",
                    "compare_b": "",
                    "context": "",
                    "doc1": "",
                    "doc2": "",
                    "answer": "",
                },
                config=config,
            )
            return {
                "answer": final_state.get("answer", "") or "No answer produced.",
                "route": final_state.get("route", "") or "UNKNOWN",
                "restated_query": final_state.get("restated_query", ""),
            }
        except Exception as e:
            return {"answer": f"Error processing query: {str(e)}", "route": "ERROR",
                    "restated_query": ""}


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("INITIALIZING WAKEEL AI - LEGAL AGENT WITH INTEGRATED RAG SYSTEM")
    print("="*60)

    # Index PDFs if needed
    print("\nChecking for documents to index...")
    index_pdfs()

    # Initialize Gemini LLM
    require_api_key()
    print("\nInitializing Gemini AI model...")
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Initialize Agent
    print("Initializing Wakeel agent (LangGraph)...")
    agent = WakeelAgent(llm)

    print("\n" + "="*60)
    print("AGENT READY - Enhanced with Hybrid RAG Search")
    print("="*60)
    print("\nCapabilities:")
    print("  - Hybrid search (semantic + keyword)")
    print("  - Pakistani court judgment analysis")
    print("  - Case comparison and summarization")
    print("  - Document indexing and retrieval")
    print("  - Legal classification and routing")
    print("\nFeatures:")
    print("  - Search by case name, number, or citation")
    print("  - Search by legal provisions (Section 302 PPC)")
    print("  - Extract judge names, parties, and precedents")
    print("  - Intelligent chunking for court judgments")
    print(f"\nPowered by: Google {LLM_MODEL} on LangGraph")
    print("="*60)

    # Interactive loop
    print("\nChat with Wakeel AI (type 'exit' to quit, 'index' to pick up new PDFs)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye.")
            break

        if user_input.lower() in ['index', 'reindex']:
            try:
                index_pdfs()
                print("\nIndexing complete. New documents are searchable.\n")
            except Exception as e:
                print(f"\nError indexing documents: {str(e)}\n")
            continue

        try:
            print("\nWakeel AI: ", end="", flush=True)
            response = agent.run(user_input)
            print(response)
            print()
        except Exception as e:
            print(f"\nError: {str(e)}\n")
