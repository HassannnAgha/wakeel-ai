"""Tests for the MCP server: the protocol handshake and the statute tools, over real stdio.

Run:  .venv/bin/python -m pytest tests/ -v

Search is not exercised here (it loads the embedding model); the protocol and the
deterministic tools are.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def talk(messages):
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "mcp_server.py")],
                          input="\n".join(json.dumps(m) for m in messages) + "\n",
                          capture_output=True, text=True, timeout=120, encoding="utf-8")
    return [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]


def test_handshake_tool_list_and_statute_tools():
    replies = talk([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "get_statute", "arguments": {"law": "ppc", "number": "489-f"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "get_statute", "arguments": {"law": "ppc", "number": "9999"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "nope"},
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
         "params": {"name": "get_statute", "arguments": {"law": "crpc", "number": "561-a"}}},
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
         "params": {"name": "get_statute", "arguments": {"law": "cpc", "number": "12"}}},
    ])
    by_id = {r["id"]: r for r in replies}
    assert len(replies) == 7                                   # no reply to the notification
    assert by_id[1]["result"]["protocolVersion"] == "2025-06-18"
    assert by_id[1]["result"]["serverInfo"]["name"] == "wakeel-ai"
    names = {t["name"] for t in by_id[2]["result"]["tools"]}
    assert {"search_judgments", "get_judgment", "list_judgments", "get_statute", "search_statutes"} <= names
    if os.path.exists(os.path.join(ROOT, "statutes.json")):
        statute = json.loads(by_id[3]["result"]["content"][0]["text"])
        assert statute["title"] == "Dishonestly issuing a cheque" and not by_id[3]["result"]["isError"]
        crpc = json.loads(by_id[6]["result"]["content"][0]["text"])
        assert crpc["title"] == "Saving of inherent power of High Court" and crpc["number"] == "561-A"
        if not by_id[7]["result"]["isError"]:
            assert json.loads(by_id[7]["result"]["content"][0]["text"])["title"] == "Bar to further suit"
    assert by_id[4]["result"]["isError"] is True
    assert by_id[5]["error"]["code"] == -32601
