"""Plain web UI for Wakeel AI.

A single-page chat served from the Python standard library: no Flask, no Gradio,
no CDN, nothing to install beyond what the agent already needs.

    python ui.py [--port 8000] [--no-index]

Then open http://127.0.0.1:8000
"""
import argparse
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import wakeel_ai as W

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wakeel AI</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --line: #e5e7eb;
    --user-bg: #1f2937; --user-fg: #ffffff; --bot-bg: #f3f4f6;
    --accent: #15803d; --err: #b91c1c;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1115; --fg: #e8e8e8; --muted: #9ca3af; --line: #272b33;
      --user-bg: #2563eb; --user-fg: #ffffff; --bot-bg: #1a1d24;
      --accent: #4ade80; --err: #f87171;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg); height: 100vh;
    display: flex; flex-direction: column;
    font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }
  header {
    padding: 14px 16px; border-bottom: 1px solid var(--line);
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }
  header h1 { font-size: 17px; margin: 0; font-weight: 650; }
  header .sub { color: var(--muted); font-size: 13px; }
  header .spacer { flex: 1; }
  button {
    font: inherit; font-size: 13px; padding: 6px 12px; cursor: pointer;
    background: var(--bg); color: var(--fg);
    border: 1px solid var(--line); border-radius: 7px;
  }
  button:hover:not(:disabled) { border-color: var(--muted); }
  button:disabled { opacity: .5; cursor: default; }
  #log { flex: 1; overflow-y: auto; padding: 18px 16px; }
  #inner { max-width: 780px; margin: 0 auto; display: flex; flex-direction: column; gap: 14px; }
  .msg { display: flex; flex-direction: column; gap: 4px; max-width: 86%; }
  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.bot  { align-self: flex-start; }
  .bubble { padding: 10px 14px; border-radius: 12px; white-space: pre-wrap; word-wrap: break-word; }
  .user .bubble { background: var(--user-bg); color: var(--user-fg); }
  .bot .bubble  { background: var(--bot-bg); border: 1px solid var(--line); }
  .tag { font-size: 11px; color: var(--muted); letter-spacing: .04em; }
  .tag b { color: var(--accent); font-weight: 600; }
  .bot.error .bubble { border-color: var(--err); color: var(--err); }
  .empty { color: var(--muted); text-align: center; margin-top: 12vh; font-size: 14px; }
  .empty code {
    display: inline-block; margin: 3px; padding: 4px 9px; cursor: pointer;
    background: var(--bot-bg); border: 1px solid var(--line); border-radius: 6px;
    font-size: 13px; color: var(--fg);
  }
  .empty code:hover { border-color: var(--accent); }
  footer { border-top: 1px solid var(--line); padding: 12px 16px; }
  form { max-width: 780px; margin: 0 auto; display: flex; gap: 8px; }
  textarea {
    flex: 1; resize: none; font: inherit; padding: 10px 12px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--bg); color: var(--fg);
    max-height: 160px;
  }
  textarea:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .send { padding: 0 18px; }
  .dots::after { content: ''; animation: d 1.2s steps(4,end) infinite; }
  @keyframes d { 0%{content:''} 25%{content:'.'} 50%{content:'..'} 75%{content:'...'} }
</style>
</head>
<body>
<header>
  <h1>Wakeel AI</h1>
  <span class="sub">Pakistani case law research</span>
  <span class="spacer"></span>
  <button id="reindex" title="Scan pdfs/ for new judgments">Index PDFs</button>
  <button id="reset">New chat</button>
</header>

<div id="log"><div id="inner"><div class="empty" id="empty">
  Ask about the judgments indexed in <b>pdfs/</b>.<br>
  <code>Find bail cases under Section 497 CrPC</code>
  <code>Summarize the Lahore High Court judgment on maintenance</code>
  <code>Compare the 2023 writ petition with the criminal appeal</code>
</div></div></div>

<footer>
  <form id="form">
    <textarea id="input" rows="1" placeholder="Ask a legal question&hellip;" autofocus></textarea>
    <button class="send" id="send" type="submit">Send</button>
  </form>
</footer>

<script>
const inner = document.getElementById('inner');
const input = document.getElementById('input');
const form  = document.getElementById('form');
const send  = document.getElementById('send');
const empty = document.getElementById('empty');
let thread = crypto.randomUUID ? crypto.randomUUID() : String(Math.random());
let busy = false;

const ROUTES = {GQ:'general query', SQ:'summarize', CQ:'compare', REJECT:'needs clarification', ERROR:'error'};

function add(role, text, tag, isError) {
  if (empty) empty.remove();
  const m = document.createElement('div');
  m.className = 'msg ' + role + (isError ? ' error' : '');
  if (tag) {
    const t = document.createElement('div');
    t.className = 'tag';
    t.innerHTML = tag;
    m.appendChild(t);
  }
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  m.appendChild(b);
  inner.appendChild(m);
  document.getElementById('log').scrollTop = 1e9;
  return b;
}

async function ask(text) {
  if (busy || !text.trim()) return;
  busy = true; send.disabled = true;
  add('user', text);
  const pending = add('bot', 'thinking');
  pending.className = 'bubble dots';
  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, thread_id: thread})
    });
    const data = await r.json();
    pending.closest('.msg').remove();
    const label = ROUTES[data.route] || data.route || '';
    add('bot', data.answer, label ? 'routed to <b>' + label + '</b>' : '', data.route === 'ERROR');
  } catch (e) {
    pending.closest('.msg').remove();
    add('bot', 'Could not reach the server: ' + e.message, '', true);
  }
  busy = false; send.disabled = false; input.focus();
}

form.addEventListener('submit', e => {
  e.preventDefault();
  const t = input.value;
  input.value = ''; input.style.height = 'auto';
  ask(t);
});

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
});

document.addEventListener('click', e => {
  if (e.target.tagName === 'CODE' && e.target.closest('.empty')) ask(e.target.textContent);
});

document.getElementById('reset').addEventListener('click', () => {
  thread = crypto.randomUUID ? crypto.randomUUID() : String(Math.random());
  inner.innerHTML = '';
  add('bot', 'Started a new conversation. Earlier turns are no longer in context.');
});

document.getElementById('reindex').addEventListener('click', async (e) => {
  e.target.disabled = true;
  const b = add('bot', 'Indexing pdfs/');
  b.className = 'bubble dots';
  try {
    const r = await fetch('/api/index', {method: 'POST'});
    const d = await r.json();
    b.className = 'bubble'; b.textContent = d.message;
  } catch (err) {
    b.className = 'bubble'; b.textContent = 'Indexing failed: ' + err.message;
  }
  e.target.disabled = false;
});
</script>
</body>
</html>
"""


def make_handler(agent, index_fn):
    """Build the request handler. agent/index_fn are injected so tests can stub them."""

    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter than the default
            sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, payload: dict):
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/healthz":
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"

            if self.path == "/api/index":
                with lock:
                    try:
                        self._json(200, {"message": index_fn()})
                    except Exception as e:
                        self._json(500, {"message": f"Indexing failed: {e}"})
                return

            if self.path != "/api/chat":
                self._json(404, {"error": "not found"})
                return

            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except ValueError:
                self._json(400, {"answer": "Malformed request.", "route": "ERROR"})
                return

            message = (data.get("message") or "").strip()
            if not message:
                self._json(400, {"answer": "Empty message.", "route": "ERROR"})
                return

            thread_id = data.get("thread_id") or str(uuid.uuid4())
            # The agent and its checkpointer are not thread-safe; serialise turns.
            with lock:
                try:
                    result = agent.run_detailed(message, thread_id=thread_id)
                except Exception as e:
                    result = {"answer": f"Error: {e}", "route": "ERROR"}
            self._json(200, result)

    return Handler


def build_agent():
    W.require_api_key()
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=W.LLM_MODEL, temperature=0.3,
        max_tokens=None, timeout=None, max_retries=2,
    )
    return W.WakeelAgent(llm)


def main():
    ap = argparse.ArgumentParser(description="Plain web UI for Wakeel AI")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-index", action="store_true", help="skip indexing pdfs/ at startup")
    args = ap.parse_args()

    if not args.no_index:
        print("Checking for documents to index...")
        W.index_pdfs()

    print("Initializing Gemini AI model...")
    agent = build_agent()

    handler = make_handler(agent, lambda: W.index_documents_tool("index"))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"\nWakeel AI is running at http://{args.host}:{args.port}  (Ctrl-C to stop)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
