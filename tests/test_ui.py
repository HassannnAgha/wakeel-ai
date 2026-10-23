"""Tests for the plain web UI: routes, JSON contract, error handling.

Run:  .venv/bin/python -m pytest tests/ -v
 or:  .venv/bin/python tests/test_ui.py

Starts the real server on an ephemeral port with a stub agent.
"""
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ui  # noqa: E402


class StubAgent:
    def __init__(self):
        self.seen = []
        self.styles = []

    def run_detailed(self, message, thread_id=None, mode=None, language=None):
        self.seen.append((message, thread_id))
        self.styles.append((mode, language))
        if "boom" in message:
            raise RuntimeError("simulated crash")
        return {"answer": "answer to " + message, "route": "GQ",
                "restated_query": message, "grounded": True,
                "sources": [{"n": 1, "doc_id": "d1", "case_no": "12/2019",
                             "judgment_type": "Writ Petition", "court": "LAHORE HIGH COURT",
                             "title": "A v B", "paragraph": "7", "section_type": "analysis",
                             "judges": ["Ali"], "citations": [], "provisions": [],
                             "parties": {}, "score": 0.9, "excerpt": "7. Judgment text."}]}


def start_server(agent=None, index_fn=None):
    agent = agent or StubAgent()
    handler = ui.make_handler(agent, index_fn or (lambda: "Indexed 0 PDFs."))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, agent, f"http://127.0.0.1:{server.server_address[1]}"


def post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_serves_the_page():
    server, _, base = start_server()
    try:
        status, body = get(base, "/")
        assert status == 200
        assert "<title>Wakeel AI</title>" in body
        assert "/api/chat" in body
    finally:
        server.shutdown()


def test_page_has_the_source_panel():
    server, _, base = start_server()
    try:
        _, body = get(base, "/")
        assert 'id="panel"' in body
        assert "openSource" in body
        assert "renderSourceList" in body
    finally:
        server.shutdown()


def test_page_has_mode_and_language_toggles():
    server, _, base = start_server()
    try:
        _, body = get(base, "/")
        assert 'id="mode"' in body and 'id="language"' in body
        assert 'value="roman_ur"' in body
    finally:
        server.shutdown()


def test_mode_and_language_are_passed_through_and_defaulted():
    server, agent, base = start_server()
    try:
        post(base, "/api/chat", {"message": "a", "mode": "lawyer", "language": "ur"})
        post(base, "/api/chat", {"message": "b"})
        assert agent.styles[0] == ("lawyer", "ur")
        assert agent.styles[1] == (ui.W.DEFAULT_MODE, ui.W.DEFAULT_LANGUAGE)
    finally:
        server.shutdown()


class StreamingStubAgent(StubAgent):
    def run_stream(self, message, thread_id=None, mode=None, language=None):
        self.seen.append((message, thread_id))
        self.styles.append((mode, language))
        yield {"type": "step", "node": "classify", "route": "GQ", "restated_query": message}
        yield {"type": "step", "node": "retrieve", "passages": 3, "judgments": 2}
        if "boom" in message:
            raise RuntimeError("simulated crash mid-stream")
        result = StubAgent.run_detailed(self, message, thread_id)
        self.seen.pop(); self.styles.pop()   # run_detailed recorded this turn a second time
        yield {"type": "result", **result}


def post_lines(base, path, payload):
    """POST and return the response as a list of decoded JSON lines."""
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.headers["Content-Type"].startswith("application/x-ndjson")
        return [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]


def test_stream_sends_steps_then_the_result():
    server, agent, base = start_server(StreamingStubAgent())
    try:
        events = post_lines(base, "/api/chat/stream",
                            {"message": "hello", "mode": "lawyer", "language": "ur"})
        assert [e["type"] for e in events] == ["step", "step", "result"]
        assert events[0]["node"] == "classify" and events[1]["passages"] == 3
        assert events[-1]["answer"] == "answer to hello"
        assert agent.styles[0] == ("lawyer", "ur")
    finally:
        server.shutdown()


def test_stream_falls_back_for_agents_that_cannot_stream():
    server, _, base = start_server()   # StubAgent has no run_stream
    try:
        events = post_lines(base, "/api/chat/stream", {"message": "hello"})
        assert len(events) == 1 and events[0]["type"] == "result"
        assert events[0]["answer"] == "answer to hello"
    finally:
        server.shutdown()


def test_stream_crash_ends_with_an_error_result_and_server_survives():
    server, _, base = start_server(StreamingStubAgent())
    try:
        events = post_lines(base, "/api/chat/stream", {"message": "boom"})
        assert events[-1]["type"] == "result" and events[-1]["route"] == "ERROR"
        assert "simulated crash" in events[-1]["answer"]
        status, data = post(base, "/api/chat", {"message": "after"})
        assert status == 200 and data["route"] == "GQ"
    finally:
        server.shutdown()


def test_deep_link_with_a_question_serves_the_page():
    server, _, base = start_server()
    try:
        status, body = get(base, "/?q=bail%20cases&language=ur")
        assert status == 200 and "<title>Wakeel AI</title>" in body
    finally:
        server.shutdown()


def test_stats_endpoint():
    agent = StubAgent()
    handler = ui.make_handler(agent, lambda: "ok", stats_fn=lambda: {"judgments": 49, "passages": 770})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, body = get(base, "/api/stats")
        assert status == 200 and json.loads(body) == {"judgments": 49, "passages": 770}
    finally:
        server.shutdown()


def test_every_page_path_serves_the_app():
    server, _, base = start_server()
    try:
        for path in ("/", "/ask", "/summarize", "/compare", "/library"):
            status, body = get(base, path)
            assert status == 200 and "<title>Wakeel AI</title>" in body, path
        assert all(f'data-page="{p}"' in body for p in ("home", "ask", "summarize", "compare", "library"))
        assert "/api/chat/stream" in body and "/api/judgments" in body and "renderMarkdown" in body
    finally:
        server.shutdown()


class TaskStubAgent(StubAgent):
    def summarize_judgment(self, doc_id, mode=None, language=None):
        self.styles.append((mode, language))
        return {"answer": "summary of " + doc_id, "route": "SQ", "sources": [], "grounded": True}

    def compare_judgments(self, doc_a, doc_b, mode=None, language=None):
        return {"answer": f"{doc_a} vs {doc_b}", "route": "CQ", "sources": [], "grounded": True}


def test_summarize_and_compare_endpoints():
    server, agent, base = start_server(TaskStubAgent())
    try:
        status, data = post(base, "/api/summarize", {"doc_id": "lhc_1", "mode": "lawyer", "language": "ur"})
        assert status == 200 and data["answer"] == "summary of lhc_1"
        assert agent.styles[-1] == ("lawyer", "ur")
        status, data = post(base, "/api/compare", {"doc_a": "lhc_1", "doc_b": "lhc_2"})
        assert status == 200 and data["answer"] == "lhc_1 vs lhc_2"
        status, data = post(base, "/api/compare", {"doc_a": "lhc_1"})
        assert status == 400
    finally:
        server.shutdown()


class ExplainStubAgent(StubAgent):
    def explain_document(self, text, images, mode=None, language=None):
        return {"answer": f"text={len(text)} images={len(images)}", "route": "DOC", "sources": [], "grounded": False}


def test_explain_endpoint_accepts_text_and_photos_and_rejects_the_rest():
    import base64
    server, _, base = start_server(ExplainStubAgent())
    try:
        status, data = post(base, "/api/explain", {"text": "FIR under section 489-F PPC"})
        assert status == 200 and data["answer"] == "text=27 images=0"
        png = base64.b64encode(b"\x89PNG fake").decode()
        status, data = post(base, "/api/explain", {"filename": "fir.png", "mime": "image/png", "data": png})
        assert status == 200 and data["answer"] == "text=0 images=1"
        status, data = post(base, "/api/explain", {"filename": "x.exe", "mime": "application/x-msdownload", "data": png})
        assert status == 400 and "PDF or a photo" in data["answer"]
        assert post(base, "/api/explain", {})[0] == 400
        status, data = post(base, "/api/explain", {"filename": "bad.pdf", "mime": "application/pdf", "data": png})
        assert status == 400 and "PDF could not be opened" in data["answer"]
    finally:
        server.shutdown()


def start_cached_server(agent):
    import tempfile
    cache = ui.AnswerCache(tempfile.mkdtemp())
    handler = ui.make_handler(agent, lambda: "ok", stats_fn=lambda: {}, cache=cache, replay_delay=0)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, cache, f"http://127.0.0.1:{server.server_address[1]}"


def test_cache_replays_a_repeated_first_question_without_the_agent():
    agent = StreamingStubAgent()
    server, cache, base = start_cached_server(agent)
    try:
        first = post_lines(base, "/api/chat/stream", {"message": "hello", "thread_id": "a"})
        again = post_lines(base, "/api/chat/stream", {"message": "hello", "thread_id": "b"})
        assert len(agent.seen) == 1                       # second answer came from the cache
        assert [e["type"] for e in again] == [e["type"] for e in first] == ["step", "step", "result"]
        status, plain = post(base, "/api/chat", {"message": "hello", "thread_id": "c"})
        assert status == 200 and plain["answer"] == "answer to hello" and len(agent.seen) == 1
    finally:
        server.shutdown()


def test_cache_skips_follow_ups_and_never_saves_failures():
    agent = StreamingStubAgent()
    server, cache, base = start_cached_server(agent)
    try:
        post_lines(base, "/api/chat/stream", {"message": "hello", "thread_id": "t"})
        post_lines(base, "/api/chat/stream", {"message": "follow up", "thread_id": "t"})
        post_lines(base, "/api/chat/stream", {"message": "follow up", "thread_id": "u"})
        assert [m for m, _ in agent.seen].count("follow up") == 2   # follow-up was not cached
        post_lines(base, "/api/chat/stream", {"message": "boom", "thread_id": "v"})
        post_lines(base, "/api/chat/stream", {"message": "boom", "thread_id": "w"})
        assert [m for m, _ in agent.seen].count("boom") == 2        # failure was not cached
    finally:
        server.shutdown()


def test_cache_covers_summaries():
    agent = TaskStubAgent()
    server, cache, base = start_cached_server(agent)
    try:
        post(base, "/api/summarize", {"doc_id": "lhc_1"})
        status, data = post(base, "/api/summarize", {"doc_id": "lhc_1"})
        assert status == 200 and data["answer"] == "summary of lhc_1" and len(agent.styles) == 1
        post(base, "/api/summarize", {"doc_id": "lhc_1", "language": "ur"})
        assert len(agent.styles) == 2                     # a different language is a new answer
    finally:
        server.shutdown()


class CoreStubAgent(TaskStubAgent):
    def draft_document(self, kind, facts, mode=None, language=None, sample=""):
        return {"answer": f"{kind}: {sorted(facts)}" + (f" in the style of {len(sample)} chars" if sample else ""),
                "route": "DRAFT", "sources": [], "grounded": False}

    def ask_document(self, text, question, mode=None, language=None):
        return {"answer": f"{question} ({len(text)} chars)", "route": "DOC", "sources": [], "grounded": True}

    def summarize_document(self, text, images, mode=None, language=None):
        return {"answer": f"brief of {len(text)} chars", "route": "SQ", "sources": [], "grounded": True}


class TableStubAgent(StubAgent):
    def review_table(self, doc_ids, columns, mode=None, language=None, documents=None):
        rows = [{"doc_id": d, "cells": []} for d in doc_ids]
        rows += [{"doc_id": f"upload-{i}", "name": d["name"], "text": d["text"], "cells": []}
                 for i, d in enumerate(documents or [], 1)]
        return {"answer": "| x |", "route": "TABLE", "columns": columns, "rows": rows, "sources": [], "grounded": True}

    def ask_table(self, question, table, mode=None, language=None):
        return {"answer": "from table: " + question, "route": "TABLE", "sources": [], "grounded": True}

    def strategy_memo(self, facts, mode=None, language=None):
        return {"answer": "memo on " + ", ".join(sorted(facts)), "route": "STRATEGY", "sources": [], "grounded": True}

    def agent_room(self, situation, document="", doc_ids=None, mode=None, language=None):
        yield {"type": "team", "kind": "situation", "team": [{"id": "orchestrator", "name": "Orchestrator", "role": "r", "ai": True}],
               "edges": [], "reason": "r", "brief": {}, "doc_ids": doc_ids or []}
        yield {"type": "agent", "id": "orchestrator", "status": "done",
               "note": f"read {situation!r}, {len(document)} chars and {len(doc_ids or [])} judgments"}
        yield {"type": "done"}

    def ask_judgment(self, doc_id, question, mode=None, language=None):
        return {"answer": f"{question} about {doc_id}", "route": "JUDGMENT", "sources": [], "grounded": True}

    def run_workflow(self, kind, facts, mode=None, language=None, doc_id=""):
        return {"answer": f"{kind} on {sorted(facts)} {doc_id}".strip(), "route": kind.upper(), "sources": [], "grounded": False}


def test_ask_judgment_endpoint():
    server, _, base = start_server(TableStubAgent())
    try:
        status, data = post(base, "/api/judgment/ask", {"doc_id": "lhc_1", "question": "Why?"})
        assert status == 200 and data["answer"] == "Why? about lhc_1" and data["route"] == "JUDGMENT"
        assert post(base, "/api/judgment/ask", {"doc_id": "lhc_1", "question": " "})[0] == 400
        assert post(base, "/api/judgment/ask", {"question": "Why?"})[0] == 400
        page = ui.PAGE
        assert 'id="ask-scope"' in page and "/api/judgment/ask" in page and 'data-act="ask"' in page
    finally:
        server.shutdown()


def test_workflow_endpoint():
    server, _, base = start_server(TableStubAgent())
    try:
        status, data = post(base, "/api/workflow", {"kind": "hearing", "facts": {"The facts": "x"}})
        assert status == 200 and data["answer"] == "hearing on ['The facts']"
        status, data = post(base, "/api/workflow", {"kind": "chronology", "doc_id": "lhc_1"})
        assert status == 200 and data["answer"] == "chronology on [] lhc_1"     # a judgment needs no facts
        assert post(base, "/api/workflow", {"kind": "nope", "facts": {"a": "b"}})[0] == 400
        assert post(base, "/api/workflow", {"kind": "hearing", "facts": {}})[0] == 400
    finally:
        server.shutdown()


def test_strategy_endpoint():
    server, _, base = start_server(TableStubAgent())
    try:
        status, data = post(base, "/api/strategy", {"facts": {"What happened": "x", "Your side": "accused"}})
        assert status == 200 and data["answer"] == "memo on What happened, Your side"
        assert post(base, "/api/strategy", {"facts": {"What happened": " "}})[0] == 400
    finally:
        server.shutdown()


def test_review_table_endpoints():
    server, _, base = start_server(TableStubAgent())
    try:
        status, data = post(base, "/api/table", {"doc_ids": ["a", "b"], "columns": ["Outcome"]})
        assert status == 200 and [r["doc_id"] for r in data["rows"]] == ["a", "b"]
        assert post(base, "/api/table", {"doc_ids": [], "columns": ["Outcome"]})[0] == 400
        # The user's own document, uploaded as text: read in memory and passed through as a row.
        note = base64.b64encode(b"%PDF-not-really").decode()
        status, data = post(base, "/api/table", {"doc_ids": ["a"], "columns": ["Outcome"],
                                                 "files": [{"filename": "notice.pdf", "mime": "application/pdf", "data": note}]})
        assert status == 400 and "notice.pdf" in data["answer"]          # a broken PDF is named, not ignored
        status, data = post(base, "/api/table", {"doc_ids": [], "columns": ["Outcome"],
                                                 "files": [{"filename": "fir.txt", "text": "FIR under section 489-F PPC"}]})
        assert status == 200 and data["rows"][0]["name"] == "fir.txt"
        assert data["rows"][0]["text"] == "FIR under section 489-F PPC"
        status, data = post(base, "/api/table/ask", {"question": "why?", "table": "| x |"})
        assert status == 200 and data["answer"] == "from table: why?"
        assert post(base, "/api/table/ask", {"question": "why?"})[0] == 400
    finally:
        server.shutdown()


def test_draft_ask_and_uploaded_brief_endpoints():
    server, _, base = start_server(CoreStubAgent())
    try:
        status, data = post(base, "/api/draft", {"kind": "cheque_notice", "facts": {"Amount": "900000"}})
        assert status == 200 and data["answer"] == "cheque_notice: ['Amount']"
        assert post(base, "/api/draft", {"kind": "nope", "facts": {}})[0] == 400
        status, data = post(base, "/api/draft", {"kind": "cheque_notice", "facts": {"Amount": "900000"},
                                                 "sample": "Dear Sir, TAKE NOTICE that ..."})
        assert data["answer"] == "cheque_notice: ['Amount'] in the style of 30 chars"
        status, data = post(base, "/api/explain/ask", {"text": "Pay in 15 days.", "question": "Deadline?"})
        assert status == 200 and data["answer"] == "Deadline? (15 chars)"
        assert post(base, "/api/explain/ask", {"text": "", "question": "x"})[0] == 400
        status, data = post(base, "/api/summarize", {"text": "JUDGMENT text"})
        assert status == 200 and data["answer"] == "brief of 13 chars"
        status, data = post(base, "/api/summarize", {"doc_id": "lhc_1"})
        assert data["answer"] == "summary of lhc_1"           # a library judgment still works
    finally:
        server.shutdown()


def test_statutes_endpoint_serves_the_statute_file():
    saved = ui._statutes
    ui._statutes = {"ppc:489-F": {"act": "Pakistan Penal Code, 1860", "kind": "section", "number": "489-F",
                                  "title": "Dishonestly issuing a cheque", "text": "Whoever dishonestly issues a cheque ...",
                                  "source": "http://example", "retrieved": "2026-10-22"}}
    server, _, base = start_server()
    try:
        status, body = get(base, "/api/statutes")
        data = json.loads(body)["statutes"]
        assert status == 200 and data["ppc:489-F"]["title"] == "Dishonestly issuing a cheque"
        _, page = get(base, "/library")
        assert "linkStatutes" in page and 'data-tab="statutes"' in page
    finally:
        ui._statutes = saved
        server.shutdown()


def test_statute_file_has_the_hand_checked_provisions():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "statutes.json")
    if not os.path.exists(path):
        import pytest
        pytest.skip("statutes.json not built")
    data = json.load(open(path, encoding="utf-8"))
    assert data["ppc:489-F"]["text"].startswith("Whoever dishonestly issues a cheque towards repayment of a loan")
    assert data["const:10A"]["title"] == "Right to fair trial"
    assert all(v["text"] and v["source"] and v["note"] for v in data.values())
    if any(k.startswith("crpc:") for k in data):
        bail = data["crpc:497"]
        assert bail["title"] == "When bail may be taken in case of non-bailable offence"
        assert bail["text"].startswith("(1) When any person accused of any non-bailable offence is arrested")
        assert data["crpc:561A"]["number"] == "561-A"
        assert data["crpc:561A"]["text"].startswith("Nothing in this Code shall be deemed to limit or affect the inherent")
        # No footnote markers, amendment brackets or chapter rules left in the text.
        assert not [k for k, v in data.items() if k.startswith("crpc:") and re.search(r"[\[\]*_]", v["text"])]
    if any(k.startswith("cpc:") for k in data):
        assert data["cpc:12"]["title"] == "Bar to further suit" and "on the plea of fraud" in data["cpc:12"]["text"]
        assert data["cpc:115"]["title"] == "Revision"
        assert data["gwa:25"]["title"] == "Title of guardian to custody of ward"
        assert data["mflo:7"]["title"] == "Talaq" and data["dmma:2"]["text"].startswith("A woman married under Muslim Law")
        assert data["fca:5"]["text"].startswith("(1) Subject to the provisions of the Muslim Family Laws Ordinance")
        assert data["crpc:106"]["text"].endswith("exercising its powers of revision.")   # no cross-heading after it


def test_statute_linker():
    node = shutil.which("node")
    if not node:
        import pytest
        pytest.skip("Node.js is not installed")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linker_test.js")
    run = subprocess.run([node, script], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr


def test_new_pages_are_served():
    server, _, base = start_server()
    try:
        for path in ("/draft", "/guides", "/calculate", "/matters", "/pitch", "/developers"):
            status, body = get(base, path)
            assert status == 200 and f'data-page="{path[1:]}"' in body, path
    finally:
        server.shutdown()


def test_judgment_list_and_detail(tmp_path=None):
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    (tmp / "lhc_1.json").write_text(json.dumps([
        {"chunk_id": 0, "text": "[Case: 1/2023] Header text", "doc_id": "lhc_1", "case_no": "1/2023",
         "judgment_type": "W.P", "title": "A v B", "court": "LAHORE HIGH COURT", "judges": ["J"],
         "parties": {"appellant": "A", "respondent": "B"}, "paragraph_number": "0",
         "provisions": ["Section 12"], "citations": []},
        {"chunk_id": 1, "text": "[Paragraph 2] Body", "doc_id": "lhc_1", "paragraph_number": "2",
         "provisions": ["section 12", "Article 199"], "citations": ["2016 SCMR 1"]},
    ]), encoding="utf-8")
    saved = ui.W.METADATA_DIR
    ui.W.METADATA_DIR = tmp
    server, _, base = start_server()
    try:
        status, body = get(base, "/api/judgments")
        listed = json.loads(body)["judgments"]
        assert status == 200 and len(listed) == 1 and listed[0]["passages"] == 2
        assert listed[0]["provisions"][0].lower() == "section 12"   # counted once, ignoring case
        status, body = get(base, "/api/judgments/lhc_1")
        detail = json.loads(body)
        assert status == 200 and [p["paragraph"] for p in detail["passages"]] == ["0", "2"]
        assert detail["passages"][1]["text"] == "Body"
        assert get(base, "/api/judgments/nope")[0] == 404
        assert get(base, "/api/judgments/..%2F..%2Fsecret")[0] == 404
        stats = ui.corpus_stats()
        assert stats["judgments"] == 1 and stats["provisions"] == 2 and stats["citations"] == 1
        status, body = get(base, "/api/insights")
        ins = json.loads(body)
        assert status == 200 and ins["case_types"] == [{"label": "Writ petition", "count": 1}]
        assert ins["years"] == [{"label": "2023", "count": 1}]
        assert ins["citations"][0]["label"] == "2016 SCMR 1" and ins["citations"][0]["docs"] == ["lhc_1"]
    finally:
        ui.W.METADATA_DIR = saved
        server.shutdown()


def test_healthz():
    server, _, base = start_server()
    try:
        assert get(base, "/healthz")[0] == 200
    finally:
        server.shutdown()


def test_unknown_path_is_404():
    server, _, base = start_server()
    try:
        assert get(base, "/nope")[0] == 404
    finally:
        server.shutdown()


def test_chat_returns_answer_route_and_sources():
    server, agent, base = start_server()
    try:
        status, data = post(base, "/api/chat", {"message": "hello", "thread_id": "t1"})
        assert status == 200
        assert data["answer"] == "answer to hello"
        assert data["route"] == "GQ"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["paragraph"] == "7"
        assert data["grounded"] is True
        assert agent.seen == [("hello", "t1")]
    finally:
        server.shutdown()


def test_thread_id_is_passed_through_and_defaulted():
    server, agent, base = start_server()
    try:
        post(base, "/api/chat", {"message": "a", "thread_id": "mine"})
        post(base, "/api/chat", {"message": "b"})
        assert agent.seen[0][1] == "mine"
        assert agent.seen[1][1] and agent.seen[1][1] != "mine"
    finally:
        server.shutdown()


def test_empty_message_is_rejected():
    server, agent, base = start_server()
    try:
        status, _ = post(base, "/api/chat", {"message": "   "})
        assert status == 400
        assert agent.seen == []
    finally:
        server.shutdown()


def test_malformed_json_is_rejected():
    server, _, base = start_server()
    try:
        req = urllib.request.Request(base + "/api/chat", data=b"{not json",
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            assert False, "expected 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        server.shutdown()


def test_agent_crash_becomes_an_error_response_not_a_500():
    server, _, base = start_server()
    try:
        status, data = post(base, "/api/chat", {"message": "boom", "thread_id": "t"})
        assert status == 200
        assert data["route"] == "ERROR"
        assert "simulated crash" in data["answer"]
    finally:
        server.shutdown()


def test_server_survives_a_crash_and_serves_the_next_request():
    server, _, base = start_server()
    try:
        post(base, "/api/chat", {"message": "boom", "thread_id": "t"})
        status, data = post(base, "/api/chat", {"message": "after", "thread_id": "t"})
        assert status == 200 and data["route"] == "GQ"
    finally:
        server.shutdown()


def test_index_endpoint():
    server, _, base = start_server(index_fn=lambda: "Indexed 3 PDFs.")
    try:
        status, data = post(base, "/api/index", {})
        assert status == 200 and data["message"] == "Indexed 3 PDFs."
    finally:
        server.shutdown()


def test_index_failure_is_reported():
    def boom():
        raise RuntimeError("disk full")

    server, _, base = start_server(index_fn=boom)
    try:
        status, data = post(base, "/api/index", {})
        assert status == 500 and "disk full" in data["message"]
    finally:
        server.shutdown()


if __name__ == "__main__":
    funcs = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in funcs:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed.append(name)
    print(f"\n{len(funcs) - len(failed)}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)


def test_voice_input_is_wired_to_the_ask_explain_and_draft_boxes():
    page = ui.PAGE
    assert 'id="ask-mic"' in page and 'id="doc-mic"' in page
    assert "attachMic($('ask-mic'), $('input'))" in page and "attachMic($('doc-mic'), $('doc-text'))" in page
    assert "'ur-PK'" in page                      # Urdu speech when the answer language is Urdu


def test_fact_check():
    node = shutil.which("node")
    if not node:
        import pytest
        pytest.skip("Node.js is not installed")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "factcheck_test.js")
    run = subprocess.run([node, script], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr


def test_citations_endpoint_lists_each_citation_with_its_judgment():
    if not any(ui.W.METADATA_DIR.glob("*.json")):
        import pytest
        pytest.skip("no indexed judgments")
    server, _, base = start_server()
    try:
        status, body = get(base, "/api/citations")
        rows = json.loads(body)["citations"]
        assert status == 200 and rows and {"doc_id", "paragraph", "citation"} <= set(rows[0])
        assert any("SCMR" in r["citation"] for r in rows) and all("\n" not in r["citation"] for r in rows)
    finally:
        server.shutdown()


def test_activity_log_counts_features_and_votes_but_never_content(tmp_path=None):
    import tempfile
    log = ui.ActivityLog(os.path.join(tempfile.mkdtemp(), "logs", "activity.jsonl"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.make_handler(TableStubAgent(), lambda: "ok", stats_fn=lambda: {},
                                                                    activity=log))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        post(base, "/api/workflow", {"kind": "hearing", "facts": {"The facts": "a secret fact"}})
        post(base, "/api/workflow", {"kind": "nope", "facts": {"a": "b"}})
        post(base, "/api/feedback", {"helpful": True, "route": "HEARING", "facts_checked": 5, "facts_missing": 1})
        summary = json.loads(get(base, "/api/activity")[1])
        assert summary["requests"] == 1 and summary["ok"] == 1 and summary["errors"] == 0   # the 400 is not a failure
        assert summary["features"] == [{"label": "Workflows", "count": 1}]
        assert summary["helpful"] == 1 and summary["facts_checked"] == 5 and summary["facts_missing"] == 1
        assert "secret" not in log.path.read_text(encoding="utf-8")       # content is never logged
    finally:
        server.shutdown()


def test_simulated_replay_waits_like_the_model_and_is_off_by_default(monkeypatch):
    import tempfile
    waits = []
    monkeypatch.setattr(ui.time, "sleep", lambda s: waits.append(s))
    agent = StreamingStubAgent()
    cache = ui.AnswerCache(tempfile.mkdtemp())
    handler = ui.make_handler(agent, lambda: "ok", stats_fn=lambda: {}, cache=cache, replay_delay=0, simulate=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        post_lines(base, "/api/chat/stream", {"message": "hello", "thread_id": "a"})
        waits.clear()
        post_lines(base, "/api/chat/stream", {"message": "hello", "thread_id": "b"})
        assert len(agent.seen) == 1                       # still served from the cache
        assert waits and all(w >= 0.8 for w in waits)     # ... but each step took a believable time
    finally:
        server.shutdown()


def test_answer_reveal_helpers():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reveal_test.js")
    run = subprocess.run([node, script], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr


def test_the_tour_points_at_things_that_exist():
    page = ui.PAGE
    block = page[page.index("const TOUR = ["): page.index("let tourI = -1")]
    paths = re.findall(r"path: '([^']+)'", block)
    targets = re.findall(r"target: '([^']+)'", block)
    assert len(paths) == len(targets) >= 8
    for path in paths:
        assert path in ui.APP_PATHS, path
    for target in targets:
        for token in target.split():                             # e.g. "#thread .body .ground", ".home-l h1"
            if token.startswith("#"):
                assert f'id="{token[1:]}"' in page, target
            elif token.startswith("."):
                name = token[1:]
                assert re.search(r'class="[^"]*\b' + re.escape(name) + r'\b|[\'"]' + re.escape(name) + r'\b', page), target
    assert 'id="tour-start"' in page and "params.get('tour')" in page


def test_pitch_page_states_only_what_the_evaluation_supports():
    page = ui.PAGE
    block = page[page.index('data-page="pitch"'): page.index("<!-- LIBRARY -->")]
    assert "99.5%" in block and "0.97" in block                      # EVALUATION.md's retrieval numbers
    assert "1,687" in block                                          # statute provisions, as built
    assert "not been reviewed by a lawyer" in block and "Not legal advice" in block
    assert "stored" in block                                         # the privacy claim is stated
    assert "49" not in block                                         # the corpus size comes from the live stats
    assert "EVALUATION.md" in block and "github.com" not in block    # the repository may be private: no link to it


def test_case_desk_endpoint_streams_the_team():
    server, _, base = start_server(TableStubAgent())
    try:
        events = post_lines(base, "/api/case/stream", {"situation": "My cheque bounced", "text": "FIR No. 1"})
        assert [e["type"] for e in events] == ["team", "agent", "done"]
        assert events[1]["note"] == "read 'My cheque bounced', 9 chars and 0 judgments"
        events = post_lines(base, "/api/case/stream", {"doc_ids": ["lhc_1", "lhc_2"]})     # judgments chosen in the Library
        assert events[1]["note"] == "read '', 0 chars and 2 judgments"
        assert post(base, "/api/case/stream", {"situation": " "})[0] == 400
        page = ui.PAGE
        assert 'data-page="case"' in page and 'data-nav="case"' in page and "openCase(" in page
    finally:
        server.shutdown()


def test_case_desk_samples_match_the_demo_cache_steps():
    import pytest
    demo_prep = pytest.importorskip("demo_prep")   # demo files are kept out of the public repo
    cases = [p for label, path, p in demo_prep.STEPS if path == "/api/case/stream"]
    assert cases, "demo_prep.py should fill the cache for the case desk"
    for payload in cases:
        assert payload["situation"] in ui.PAGE, "the case desk's Try button must send exactly this text"


class WatchedStubAgent(TaskStubAgent):
    """Reports its agents as the real one does: a reader, then a writer whose words come in pieces."""
    def summarize_judgment(self, doc_id, mode=None, language=None):
        ui.W.watch_stage("load-" + doc_id, "Judgment reader", "done", "Read 3 passages")
        ui.W.watch_stage("write", "Summary agent", "working", "Writing", ai=True)
        for i, piece in enumerate(["summary ", "of ", doc_id]):
            ui.W.watch_emit({"type": "chunk", "id": "write", "text": piece, "new": i == 0})
        ui.W.watch_stage("write", "Summary agent", "done", "Wrote 3 words", ai=True)
        return super().summarize_judgment(doc_id, mode, language)


def test_a_page_can_watch_the_agents_work_and_replays_show_the_same():
    agent = WatchedStubAgent()
    server, cache, base = start_cached_server(agent)
    try:
        events = post_lines(base, "/api/summarize", {"doc_id": "lhc_1", "watch": True})
        assert [e["type"] for e in events] == ["stage", "stage", "chunk", "chunk", "chunk", "stage", "result"]
        assert "".join(e["text"] for e in events if e["type"] == "chunk") == "summary of lhc_1"
        assert events[-1]["status"] == 200 and events[-1]["answer"] == "summary of lhc_1" and "watch" not in events[-1]
        again = post_lines(base, "/api/summarize", {"doc_id": "lhc_1", "watch": True})     # from the cache
        assert len(agent.styles) == 1 and [e["type"] for e in again] == [e["type"] for e in events]
        status, data = post(base, "/api/summarize", {"doc_id": "lhc_1"})                    # not watching: plain JSON
        assert status == 200 and "watch" not in data
        refused = post_lines(base, "/api/compare", {"doc_a": "lhc_1", "watch": True})
        assert refused == [{"type": "result", "status": 400, "answer": "Choose a judgment first.", "route": "ERROR", "sources": []}]
    finally:
        server.shutdown()


def test_extract_returns_a_files_text_and_keeps_nothing():
    server, agent, base = start_server(StubAgent())
    try:
        status, data = post(base, "/api/extract", {"filename": "notice.txt", "text": "  LEGAL NOTICE under section 489-F PPC  "})
        assert status == 200 and data["text"] == "LEGAL NOTICE under section 489-F PPC" and data["chars"] == 36
        assert data["filename"] == "notice.txt" and data["ocr"] is False
        status, data = post(base, "/api/extract", {"filename": "empty.txt", "text": "   "})
        assert status == 400
    finally:
        server.shutdown()


def test_calendar_helpers():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calendar_test.js")
    run = subprocess.run([node, script], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr


def test_brief_checker():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "briefcheck_test.js")
    run = subprocess.run([node, script], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr


def test_helplines_are_the_checked_ones_and_dial():
    page = ui.PAGE
    block = page[page.index("const HELPLINES = ["): page.index("function helpStrip()")]
    for number, tel in (("1818", "tel:1818"), ("0800-70806", "tel:080070806"), ("1099", "tel:1099"), ("15", "tel:15")):
        assert f"'{number}', '{tel}'" in block
    assert "Built to Pakistan's judicial AI guidelines" in page and 'id="voice-mic"' in page


def test_glossary_words_and_their_laws():
    node = shutil.which("node")
    if node:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glossary_test.js")
        run = subprocess.run([node, script], capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert run.returncode == 0, run.stdout + run.stderr
    page = ui.PAGE
    block = page[page.index("const GLOSSARY = ["): page.index("let GLOSS_RES = null;")]
    statutes = ui.load_statutes()
    laws = re.findall(r"law: '([a-z]+:[0-9A-Z]+)'", block)
    assert len(laws) >= 15
    for key in laws:
        assert key in statutes, key                       # every meaning points at a section we hold


def test_autopilot_steps_point_at_things_that_exist():
    page = ui.PAGE
    block = page[page.index("const AUTOPILOT = ["): page.index("function autoShow()")]
    solves = page[page.index("const SOLVES = {"): page.index("function renderSolve(")]
    for who, start in re.findall(r"solveCard\('(\w+)', '([^']+)'\)", block):
        assert re.search(r"\{q: '" + re.escape(start), solves), start      # the card the step presses is there
    for fn in re.findall(r"\b(wsStartFlow|openRights|openGloss|renderSolve|wsShow|wsUpdate|closePanel|closeGloss)\(", block):
        assert "function " + fn + "(" in page
    for sel in re.findall(r"\$\('([\w-]+)'\)", block):
        assert f'id="{sel}"' in page or f"id='{sel}'" in page or sel in ("live-text", "live-form"), sel
    assert 'id="pitch-auto"' in page and "F9" in page



def test_demo_prep_fills_exactly_what_the_autopilot_sends():
    import importlib
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pytest
    pytest.importorskip("demo_prep")   # demo files are kept out of the public repo
    prep = importlib.import_module("demo_prep")
    page = ui.PAGE
    auto = page[page.index("const AUTOPILOT = ["): page.index("function autoShow()")]
    assert "const COURT_SAMPLE = '" + prep.MOOT_FACTS + "';" in page  # the Courtroom sample's facts, word for word
    assert "$('court-sample').click()" in auto                         # which the autopilot's hearing uses
    assert prep.MOOT_ANSWER in auto and "a post-arrest bail application" in page
    assert prep.MOOT["side"] in page and prep.MOOT["hearing"] in page
    js_petition = page[page.index("const SAMPLE_PETITION = '") + len("const SAMPLE_PETITION = '"):]
    js_petition = js_petition[:js_petition.index("';")].replace(chr(92) + "n", chr(10))
    assert js_petition == prep.SAMPLE_PETITION                       # the petition the opposing-counsel card sends


def test_inheritance_fields_take_only_whole_numbers_within_their_limit():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "countfield_test.js")
    run = subprocess.run([node, script], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
