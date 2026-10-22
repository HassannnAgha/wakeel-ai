"""Wakeel AI as an MCP server, so Claude, ChatGPT or any MCP client can use it as a tool.

    .venv/Scripts/python.exe mcp_server.py

Speaks the Model Context Protocol over stdio (JSON-RPC 2.0, one message per line) with no
extra dependencies. The tools search and read the indexed judgments and statutes; they do
not call a language model, so they need no API key and the calling AI does the reasoning.

Claude Desktop / Claude Code config (adjust the paths):

    {"mcpServers": {"wakeel": {
        "command": "C:/path/to/aief/.venv/Scripts/python.exe",
        "args": ["C:/path/to/aief/mcp_server.py"]}}}
"""
import json
import os
import re
import sys

# The protocol owns stdout. Everything else (model loading messages, warnings) goes to stderr.
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # the index paths are relative

import wakeel_ai as W  # noqa: E402

SERVER_INFO = {"name": "wakeel-ai", "version": "1.0"}
DEFAULT_PROTOCOL = "2025-06-18"

TOOLS = [
    {"name": "search_judgments",
     "description": "Search indexed Pakistani court judgments (Lahore High Court) with hybrid meaning + keyword "
                    "search. Returns the best-matching passages with case number, court, parties, paragraph, "
                    "provisions and text. Cite results by case number and paragraph.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "A legal question or topic, in English"},
         "top_k": {"type": "integer", "description": "How many passages (1-20, default 8)"}},
         "required": ["query"]}},
    {"name": "get_judgment",
     "description": "Read one indexed judgment in full: its case details and every passage, in order.",
     "inputSchema": {"type": "object", "properties": {
         "doc_id": {"type": "string", "description": "The judgment's id, e.g. lhc2023_466"}},
         "required": ["doc_id"]}},
    {"name": "list_judgments",
     "description": "List the indexed judgments (id, case number, parties, court), optionally filtered by words.",
     "inputSchema": {"type": "object", "properties": {
         "filter": {"type": "string", "description": "Words that must all appear in the judgment (its parties, "
                                                     "case number, provisions or text)"}}}},
    {"name": "get_statute",
     "description": "The text of a section of a Pakistani statute, or an article of the Constitution, word for "
                    "word, with its source. Laws: ppc (Pakistan Penal Code), crpc (Code of Criminal Procedure), "
                    "cpc (Code of Civil Procedure), gwa (Guardians and Wards Act), mflo (Muslim Family Laws "
                    "Ordinance), dmma (Dissolution of Muslim Marriages Act), fca (Punjab Family Courts Act), "
                    "constitution.",
     "inputSchema": {"type": "object", "properties": {
         "law": {"type": "string", "enum": ["ppc", "crpc", "cpc", "gwa", "mflo", "dmma", "fca", "constitution"]},
         "number": {"type": "string", "description": "e.g. 489-F (PPC), 561-A (CrPC), 12 (CPC) or 199 (Constitution)"}},
         "required": ["law", "number"]}},
    {"name": "search_statutes",
     "description": "Find sections of the Penal Code, the procedure codes and the family laws, and articles of "
                    "the Constitution, by words in their title or text.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"}}, "required": ["query"]}},
]


def _statutes():
    if not hasattr(_statutes, "data"):
        try:
            with open("statutes.json", encoding="utf-8") as f:
                _statutes.data = json.load(f)
        except (OSError, ValueError):
            _statutes.data = {}
    return _statutes.data


def _passage(s: dict) -> dict:
    keep = ("doc_id", "case_no", "judgment_type", "court", "parties", "paragraph", "section_type",
            "provisions", "citations", "score", "excerpt")
    return {k: s.get(k) for k in keep}


def search_judgments(query: str, top_k: int = 8):
    hits = W.search_hybrid(query, top_k=max(1, min(int(top_k or 8), 20)))
    return {"query": query, "passages": [_passage(s) for s in W.source_records(hits)]}


def get_judgment(doc_id: str):
    hits = W.judgment_hits(doc_id, max_chars=10 ** 9)
    if not hits:
        raise ValueError(f"No judgment with id {doc_id!r} in the index.")
    return {"doc_id": doc_id, "passages": [_passage(s) for s in W.source_records(hits)]}


def list_judgments(filter: str = ""):
    words = (filter or "").lower().split()
    out = []
    for name in sorted(os.listdir(W.METADATA_DIR)):
        if not name.endswith(".json"):
            continue
        chunks = W.load_judgment(name[:-5])
        if not chunks:
            continue
        head = chunks[0]
        provisions = sorted({p for c in chunks for p in c.get("provisions", [])})
        row = {"doc_id": name[:-5], "case": f"{head.get('judgment_type', '')} {head.get('case_no', '')}".strip(),
               "parties": head.get("parties", {}), "court": re.sub(r"\s+", " ", head.get("court", "")),
               "passages": len(chunks)}
        hay = (json.dumps(row) + " " + " ".join(provisions) + " " + " ".join(c.get("text", "") for c in chunks)).lower()
        if all(w in hay for w in words):
            out.append(row)
    return {"judgments": out}


def get_statute(law: str, number: str):
    number = str(number).strip().upper()
    if law in ("ppc", "constitution"):
        key = ("ppc:" if law == "ppc" else "const:") + number
    else:
        key = f"{law}:" + number.replace("-", "")   # the PDF Acts' keys drop the hyphen: 561-A -> crpc:561A
    found = _statutes().get(key)
    if not found:
        raise ValueError(f"{law} {number} is not in the statute set.")
    return found


def search_statutes(query: str):
    words = (query or "").lower().split()
    hits = [(k, v) for k, v in _statutes().items()
            if words and all(w in (v["title"] + " " + v["text"]).lower() for w in words)]
    hits.sort(key=lambda kv: (not all(w in kv[1]["title"].lower() for w in words), kv[0]))
    return {"results": [{"key": k, "act": v["act"], "number": v["number"], "title": v["title"]} for k, v in hits[:25]]}


HANDLERS = {"search_judgments": search_judgments, "get_judgment": get_judgment, "list_judgments": list_judgments,
            "get_statute": get_statute, "search_statutes": search_statutes}


def handle(msg: dict):
    """The response to one JSON-RPC message, or None for a notification."""
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if mid is None:
        return None   # notifications (e.g. notifications/initialized) need no reply
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL,
                  "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in HANDLERS:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"Unknown tool: {name}"}}
        try:
            text, is_error = json.dumps(HANDLERS[name](**args), ensure_ascii=False), False
        except Exception as e:
            text, is_error = f"{type(e).__name__}: {e}", True
        result = {"content": [{"type": "text", "text": text}], "isError": is_error}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line))
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        if reply is not None:
            _PROTOCOL.write(json.dumps(reply, ensure_ascii=False) + "\n")
            _PROTOCOL.flush()


if __name__ == "__main__":
    main()
