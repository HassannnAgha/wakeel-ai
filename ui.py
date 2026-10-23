"""Web UI for Wakeel AI.

A single-page app served from the Python standard library: no Flask, no Gradio,
no CDN, nothing to install beyond what the agent already needs. It works offline,
so a venue's Wi-Fi cannot break the demo.

    python ui.py [--port 8000] [--no-index]

Then open http://127.0.0.1:8000

Pages (one HTML file, switched in the browser): /  /ask  /summarize  /compare  /library

Endpoints:
    GET  /api/stats            corpus counts and the model in use
    GET  /api/judgments        every indexed judgment, for the pickers and the library
    GET  /api/judgments/<id>   one judgment with all its passages
    POST /api/chat             one routed turn, answered as a single JSON object
    POST /api/chat/stream      one routed turn, streamed as JSON lines: a "step" event as
                               each agent step finishes, then a "result" event
    POST /api/summarize        summarize a chosen judgment   {doc_id, mode, language}
    POST /api/compare          compare two chosen judgments  {doc_a, doc_b, mode, language}
    POST /api/index            index new PDFs
"""
import argparse
import base64
import binascii
import hashlib
import io
import json
import random
import re
import sys
import queue
import threading
import time
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

import wakeel_ai as W

APP_PATHS = ("/", "/index.html", "/ask", "/summarize", "/compare", "/explain", "/draft",
             "/guides", "/calculate", "/library", "/matters", "/case", "/pitch", "/developers", "/workspace", "/court", "/verify")
# Requests a page can watch: with "watch": true the answer comes as a stream of what each agent does, then the result.
WATCHED_PATHS = ("/api/summarize", "/api/compare", "/api/explain", "/api/explain/ask", "/api/draft", "/api/table",
                 "/api/table/ask", "/api/strategy", "/api/workflow", "/api/judgment/ask", "/api/citator", "/api/moot", "/api/video")
POST_PATHS = ("/api/chat", "/api/chat/stream", "/api/summarize", "/api/compare", "/api/explain",
              "/api/explain/ask", "/api/draft", "/api/table", "/api/table/ask", "/api/strategy", "/api/workflow", "/api/feedback", "/api/judgment/ask", "/api/case/stream", "/api/extract", "/api/citator", "/api/moot", "/api/speak", "/api/video", "/api/fill")

#: Uploads arrive base64-encoded inside JSON, about a third larger than the file.
UPLOAD_LIMIT = 10 * 1024 * 1024
BODY_LIMIT = UPLOAD_LIMIT * 4 // 3 + 64 * 1024
IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wakeel AI</title>
<style>
  :root {
    /* palette from the reference: deep blue, slate frame, black panel, rock, white, plus sand beige */
    --black: #0a0a0a; --white: #ffffff; --beige: #f4f0e8; --paper: #e9e2d4; --sand: #d8cdb9;
    --slate: #5c7174; --deep: #1f3a56; --deep-2: #2b4c6d; --deep-ink: #dfe6ee; --rock: #2c2e30;
    --ink: #111111; --muted: #736d63; --line: #dcd4c6; --line-dark: #2b2b2b; --muted-dark: #9a9a96;
    --accent: #2b4c6d; --marker: #d8cdb9;
    --display: "Bahnschrift", "DIN Condensed", "Oswald", "Arial Narrow", "Roboto Condensed", sans-serif;
    --text: "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body { margin: 0; background: var(--beige); color: var(--ink); font: 15px/1.6 var(--text); }
  .md, .turn .q, .chosen .sub { color: var(--rock); }
  button, input, select, textarea { font: inherit; color: inherit; }
  button { cursor: pointer; background: none; border: 0; padding: 0; }
  button:disabled { opacity: .4; cursor: default; }
  a { color: inherit; }

  .display { font-family: var(--display); font-weight: 700; font-stretch: 75%;
             font-variation-settings: "wdth" 75, "wght" 700; text-transform: uppercase;
             letter-spacing: -.005em; line-height: .92; }
  .eyebrow { font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
  .rule { height: 1px; background: var(--line); border: 0; margin: 0; }

  /* ---------- header ---------- */
  header { position: sticky; top: 0; z-index: 20; background: var(--beige); border-bottom: 1px solid var(--line);
           display: flex; align-items: center; gap: 24px; padding: 0 24px; height: 56px; }
  .mark { font-size: 22px; text-decoration: none; }
  nav { display: flex; gap: 20px; height: 100%; min-width: 0; overflow-x: auto; scrollbar-width: none; }
  nav::-webkit-scrollbar { display: none; }
  .mark, nav a, .toggle button, .toggle .lbl { white-space: nowrap; flex: none; }
  nav a { display: flex; align-items: center; text-decoration: none; font-size: 12px; letter-spacing: .12em;
          text-transform: uppercase; color: var(--muted); border-bottom: 2px solid transparent; margin-bottom: -1px; }
  nav a:hover { color: var(--accent); }
  nav a.on { color: var(--ink); border-bottom-color: var(--deep); }
  .grow { flex: 1; }
  .toggle { display: flex; align-items: center; gap: 10px; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; }
  .toggle .lbl { color: var(--muted); }
  .toggle button { color: var(--muted); padding: 2px 0; border-bottom: 1px solid transparent; }
  .toggle button.on { color: var(--ink); border-bottom-color: var(--ink); }
  .toggle .sep { color: var(--line); }
  @media (max-width: 1100px) { header { height: auto; flex-wrap: wrap; padding: 10px 18px; gap: 12px 22px; }
                               nav { height: 36px; } }
  /* Phones: the page links scroll sideways in one row; both toggles share the next row. */
  @media (max-width: 640px) {
    header { padding: 10px 16px; gap: 8px 16px; }
    nav { order: 3; width: 100%; overflow-x: auto; scrollbar-width: none; gap: 18px; }
    nav::-webkit-scrollbar { display: none; }
    nav a { flex: none; }
    header .grow { display: none; }
    .toggle { gap: 7px; font-size: 11px; }
    .home-l, .home-r { padding: 30px 22px; }
    .home-l .court b { font-size: 17px; }
    .specs b { font-size: 20px; }
    .ptitle { padding: 26px 22px 24px; }
    .ptitle h1 { font-size: clamp(40px, 13vw, 72px); }
    .apitable { display: block; overflow-x: auto; }
  }

  main > section { display: none; }
  main > section.on { display: block; }
  .wrap { max-width: 860px; margin: 0 auto; padding: 48px 28px 140px; }
  .wrap.wide { max-width: 1180px; }

  /* ---------- buttons, inputs ---------- */
  .btn { display: inline-flex; align-items: center; gap: 10px; height: 44px; padding: 0 22px;
         font-size: 12px; letter-spacing: .14em; text-transform: uppercase; text-decoration: none;
         background: var(--deep); color: var(--white); border: 1px solid var(--deep); }
  .btn.line { background: transparent; color: inherit; border-color: currentColor; }
  .btn:hover:not(:disabled) { opacity: .85; }
  .field { width: 100%; height: 42px; padding: 0 12px; border: 1px solid var(--sand); background: var(--white);
           border-radius: 0; outline: none; }
  .field:focus { border-color: var(--ink); }
  select.field { appearance: none; background-image: none; cursor: pointer; }

  /* ---------- the live map, inside the original home ---------- */
  .netcanvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
  .netblock { position: relative; margin: 0 -48px; padding: 18px 48px 22px; min-height: 290px; }
  .netblock .bignums { position: relative; z-index: 1; pointer-events: none; }
  .netblock .bignums b { text-shadow: 0 0 18px var(--black), 0 0 6px var(--black); }
  .live-ask { background: var(--white); border: 1px solid var(--black); padding: 14px 14px 12px; margin: 0 0 22px; max-width: 640px;
              transition: box-shadow .3s; }
  .live-ask:focus-within { box-shadow: 6px 6px 0 var(--black); }
  .live-ask .eyebrow { display: block; color: #6b6358; margin-bottom: 6px; }
  .live-ask textarea { width: 100%; background: transparent; border: 0; outline: none; resize: none; overflow: hidden; color: var(--ink);
                       font: 17px/1.5 var(--text); min-height: 52px; }
  .live-ask textarea::placeholder { color: var(--muted); }
  .live-bar { display: flex; align-items: center; gap: 10px; margin-top: 8px; flex-wrap: wrap; }
  .live-bar .btn { background: var(--black); color: var(--white); border-color: var(--black); }
  .live-try { display: flex; flex-wrap: wrap; gap: 6px; flex: 1; }
  .live-try button { background: transparent; border: 1px solid rgba(0,0,0,.25); color: var(--rock); padding: 4px 10px;
                     font: 12px var(--text); cursor: pointer; transition: border-color .2s, background .2s; }
  .live-try button:hover { border-color: var(--black); background: #efe7d8; }
  .live-feed { list-style: none; margin: -6px 0 0; padding: 0; font-size: 12.5px; color: var(--muted-dark); min-height: 66px; }
  .live-feed li { padding: 2px 0; animation: arrive .5s ease-out both; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .live-feed li:nth-last-child(n+3) { opacity: .5; }
  .live-feed li:nth-child(-n+2):nth-last-child(n+4) { display: none; }
  .live-feed li b { color: var(--white); font-weight: 600; }
  @media (max-width: 640px) { .netblock { margin: 0 -22px; padding: 14px 22px 18px; } }
  .netbox { position: relative; height: 280px; margin: 0 0 12px; background: var(--black); border: 1px solid rgba(255,255,255,.07); overflow: hidden; }
  .netlabel { position: absolute; right: 12px; top: 10px; font: 11px var(--display); letter-spacing: .12em; text-transform: uppercase; color: rgba(223,230,238,.5); }
  .netcap { position: absolute; left: 12px; bottom: 10px; list-style: none; margin: 0; padding: 0; font-size: 12px; color: #dfe6ee; max-width: 60%; }
  .netcap li { padding: 2px 0; animation: arrive .5s ease-out both; }
  .netcap li b { color: var(--white); }
  @media (max-width: 760px) { .live-feed { display: none; } .live-key { left: 16px; right: auto; } .netbox { height: 220px; } }

  .solve { background: var(--slate); color: var(--white); padding: 0 32px 32px; }
  .solve-in { background: var(--black); padding: 56px 48px 60px; }
  .solve .eyebrow { color: var(--muted-dark); }
  .solve h2 { font-size: clamp(30px, 3.6vw, 52px); line-height: 1.02; margin: 14px 0 34px; max-width: 980px; color: var(--white); }
  .solve h2 span { display: block; font-size: .42em; color: var(--muted-dark); font-family: var(--text); font-weight: 400; letter-spacing: 0;
                   text-transform: none; margin-top: 16px; line-height: 1.5; }
  .solve-tabs { display: flex; border-bottom: 1px solid var(--line-dark); margin-bottom: 28px; }
  .solve-tabs button { background: none; border: 0; border-bottom: 2px solid transparent; color: var(--muted-dark); padding: 12px 0;
                       margin-right: 34px; font: 600 15px var(--display); letter-spacing: .1em; text-transform: uppercase; cursor: pointer; }
  .solve-tabs button b { color: var(--marker); margin-left: 6px; }
  .solve-tabs button.on { color: var(--white); border-bottom-color: var(--marker); }
  .solve-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 12px; }
  .sv { background: #e4d9c6; border: 1px solid #e4d9c6; padding: 22px 22px 18px; display: flex; flex-direction: column; gap: 14px; color: var(--ink);
        cursor: pointer; animation: arrive .5s ease-out both; animation-delay: var(--d, 0s); text-align: left; font: inherit;
        transition: transform .25s, box-shadow .25s, background .25s; }
  .sv:hover, .sv:focus-visible { transform: translateY(-3px); box-shadow: 6px 6px 0 var(--slate); background: #ebe2d2; outline: none; }
  .sv .sv-n { font: 12px var(--display); letter-spacing: .12em; color: #6b6358; }
  .sv .sv-q { font-size: 19px; line-height: 1.35; color: var(--ink); margin: 0; font-weight: 600; }
  .sv .sv-was { font-size: 12.5px; color: #6b6358; margin: 0; }
  .sv .sv-a { border-top: 1px solid rgba(0,0,0,.15); padding-top: 12px; margin-top: auto; }
  .sv .sv-agent { display: inline-block; background: var(--black); color: var(--white); font: 600 11px var(--display); letter-spacing: .1em;
                  text-transform: uppercase; padding: 3px 8px; }
  .sv .sv-agent.rules { background: transparent; color: var(--black); border: 1px solid var(--black); }
  .sv .sv-how { font-size: 13.5px; color: var(--rock); margin: 9px 0 0; line-height: 1.5; }
  .sv .sv-go { font: 600 12px var(--display); letter-spacing: .12em; text-transform: uppercase; color: var(--black); margin-top: 4px; }
  @media (max-width: 760px) { .solve { padding: 0 14px 14px; } .solve-in { padding: 36px 20px 40px; } }

  /* ---------- workspace ---------- */
  .p-workspace footer { display: none; }
  .ws { display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 56px); }
  .ws-side { background: var(--black); color: #dfe6ee; padding: 20px 18px; display: flex; flex-direction: column; gap: 8px;
             position: sticky; top: 56px; height: calc(100vh - 56px); overflow-y: auto; }
  .ws-who { display: grid; grid-template-columns: 1fr 1fr; border: 1px solid rgba(255,255,255,.16); margin-bottom: 12px; }
  .ws-who button { background: none; border: 0; color: rgba(223,230,238,.66); padding: 9px 6px; font: 600 11.5px var(--text); cursor: pointer; line-height: 1.3; }
  .ws-who button.on { background: #d8cdb9; color: #0a0a0a; }
  .ws-l { font: 11px var(--display); letter-spacing: .12em; text-transform: uppercase; color: rgba(223,230,238,.66); margin-top: 8px; }
  .ws-sel { width: 100%; background: #1c1c1c; color: #f4f0e8; border: 1px solid rgba(255,255,255,.16); padding: 9px 10px; font: 14px var(--text); }
  .ws-link { background: none; border: 0; color: #d8cdb9; text-align: left; padding: 2px 0; font: 13px var(--text); cursor: pointer; }
  .ws-nav { display: flex; flex-direction: column; margin: 14px 0 6px; border-top: 1px solid rgba(255,255,255,.08); }
  .ws-nav button { background: none; border: 0; border-left: 2px solid transparent; color: #dfe6ee; text-align: left; padding: 11px 12px;
                   font: 600 14px var(--text); cursor: pointer; display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,.06); }
  .ws-nav button span { color: rgba(223,230,238,.5); font-weight: 400; }
  .ws-nav button.on { color: #f4f0e8; border-left-color: #d8cdb9; background: rgba(216,205,185,.07); }
  .ws-facts { width: 100%; background: #1c1c1c; color: #f4f0e8; border: 1px solid rgba(255,255,255,.16); padding: 10px; resize: vertical;
              font: 13px/1.5 var(--text); }
  .ws-facts::placeholder { color: rgba(223,230,238,.5); }
  .ws-priv { font-size: 11.5px; color: rgba(223,230,238,.5); margin: 6px 0 0; }
  .ws-main { background: #f7f4ee; padding: 34px clamp(16px, 4vw, 52px) 90px; min-width: 0; }
  .ws-h { font-size: clamp(30px, 3.4vw, 46px); margin: 8px 0 22px; }
  .ws-ask { background: var(--white); border: 1px solid var(--sand); padding: 14px; box-shadow: 0 10px 30px rgba(15,27,40,.06); }
  .ws-ask:focus-within { border-color: var(--deep); box-shadow: 0 0 0 3px rgba(31,58,86,.12), 0 10px 30px rgba(15,27,40,.08); }
  .ws-ask textarea { width: 100%; border: 0; outline: none; resize: none; font: 17px/1.5 var(--text); background: transparent; color: var(--ink); }
  .ws-bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 8px; }
  .ws-bar .grow { flex: 1; }
  .ws-src { display: flex; border: 1px solid var(--sand); }
  .ws-src button { background: none; border: 0; padding: 7px 12px; font: 12.5px var(--text); color: var(--muted); cursor: pointer; }
  .ws-src button.on { background: var(--deep); color: var(--white); }
  .ws-sugg { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 26px; }
  .ws-sugg button { background: var(--white); border: 1px solid var(--sand); padding: 7px 12px; font: 13px var(--text); color: var(--ink); cursor: pointer; }
  .ws-sugg button:hover { border-color: var(--deep); }
  .ws-turn { border-top: 1px solid var(--sand); padding: 22px 0 6px; }
  .ws-turn > .q { font-size: 19px; font-weight: 600; margin: 0 0 4px; }
  .ws-turn > .src { font-size: 12px; color: var(--muted); letter-spacing: .06em; text-transform: uppercase; }
  .ws-flows { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; }
  .ws-flow { background: var(--white); border: 1px solid var(--sand); padding: 16px 16px 14px; text-align: left; cursor: pointer; font: inherit; color: inherit;
             display: flex; flex-direction: column; gap: 6px; transition: border-color .2s, transform .2s, box-shadow .2s; }
  .ws-flow:hover, .ws-flow.on { border-color: var(--deep); transform: translateY(-2px); box-shadow: 0 10px 24px rgba(15,27,40,.08); }
  .ws-flow.on { box-shadow: 0 0 0 2px var(--deep); }
  .ws-flow .tag { font: 600 10.5px var(--display); letter-spacing: .12em; text-transform: uppercase; color: var(--deep); }
  .ws-flow b { font-size: 16px; }
  .ws-flow span { font-size: 13px; color: var(--muted); }
  .ws-flow i { font-style: normal; font-size: 11.5px; color: var(--muted); margin-top: auto; padding-top: 6px; border-top: 1px solid var(--sand); }
  .ws-formbox { background: var(--white); border: 1px solid var(--sand); padding: 22px; margin: 22px 0; }
  .ws-formbox h3 { margin: 0 0 4px; font-size: 22px; }
  .ws-formbox .field { margin: 4px 0 12px; }
  .ws-formbox textarea.field { height: auto; padding: 10px 12px; }
  .ws-drop { display: block; border: 1px dashed var(--muted); background: var(--white); padding: 22px; cursor: pointer; transition: border-color .2s, background .2s; }
  .ws-drop:hover, .ws-drop.over { border-color: var(--deep); background: #fbf9f4; }
  .ws-drop b { display: block; font: 600 20px var(--display); letter-spacing: .04em; text-transform: uppercase; }
  .ws-drop span { font-size: 13px; color: var(--muted); }
  .ws-docs { margin: 16px 0 0; }
  .ws-doc { display: grid; grid-template-columns: 24px 1fr auto; gap: 12px; align-items: center; padding: 11px 4px; border-bottom: 1px solid var(--sand); }
  .ws-doc b { font-size: 14.5px; }
  .ws-doc span { font-size: 12px; color: var(--muted); }
  .ws-doc p { grid-column: 2 / 4; margin: 0; font-size: 12.5px; color: var(--rock); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ws-acts { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }
  .ws-empty { color: var(--muted); font-size: 14px; padding: 14px 0; }
  .ws-hitem { border-top: 1px solid var(--sand); padding: 18px 0; }
  .ws-hitem .eyebrow { margin-bottom: 4px; }
  .ws-hitem h3 { margin: 0 0 8px; }
  @media (max-width: 860px) { .ws { grid-template-columns: 1fr; } .ws-side { position: static; height: auto; } }

  .calist { list-style: none; margin: 14px 0; padding: 0; border-top: 1px solid var(--line); }
  .calist li { border-bottom: 1px solid var(--line); }
  .calist label { display: grid; grid-template-columns: 22px 1fr; gap: 4px 10px; padding: 11px 0; cursor: pointer; }
  .calist label > span:first-of-type { display: flex; gap: 10px; align-items: baseline; }
  .calist em { font-style: normal; font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
  .calist em.up { color: var(--white); background: var(--black); padding: 1px 6px; }
  .calist .cl { grid-column: 2; font-size: 13px; color: var(--rock); }
  .calrem { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin: 6px 0 16px; font-size: 13.5px; }

  .brief-check { margin: 18px 0; }
  .bc-sum { display: grid; grid-template-columns: repeat(3, 110px) 1fr; gap: 0; border: 1px solid var(--black); background: var(--white); }
  .bc-sum > div { padding: 12px 16px; border-right: 1px solid var(--line); }
  .bc-sum b { display: block; font: 700 36px var(--display); line-height: 1; }
  .bc-sum span { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
  .bc-sum > div:first-child b { color: #a33a22; }
  .bc-sum p { margin: 0; padding: 12px 16px; font-size: 13px; color: var(--rock); align-self: center; }
  .bc-list { list-style: none; margin: 14px 0 0; padding: 0; }
  .bc-list li { display: grid; grid-template-columns: 92px 1fr; gap: 14px; padding: 12px 0; border-bottom: 1px solid var(--line); }
  .bc-list .bc-tag { font: 600 11px var(--display); letter-spacing: .12em; text-transform: uppercase; padding: 3px 0; text-align: center; height: fit-content; }
  .bc-verified .bc-tag { border: 1px solid var(--black); color: var(--black); }
  .bc-check .bc-tag { background: var(--sand); color: var(--black); }
  .bc-wrong .bc-tag { background: #a33a22; color: var(--white); }
  .bc-list b { font-size: 15px; }
  .bc-list p { margin: 2px 0 6px; font-size: 13.5px; color: var(--rock); }
  .help { margin: 16px 0 6px; border: 1px solid var(--black); background: var(--white); padding: 12px 14px; }
  .help > b { font: 600 12px var(--display); letter-spacing: .12em; text-transform: uppercase; }
  .help-lines { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 6px 16px; margin: 8px 0 6px; }
  .help-lines a { text-decoration: none; display: grid; grid-template-columns: auto 1fr; gap: 0 10px; padding: 6px 0; border-top: 1px solid var(--line); }
  .help-lines strong { font: 700 22px var(--display); grid-row: span 2; align-self: center; color: var(--black); }
  .help-lines span { font-size: 13px; font-weight: 600; color: var(--ink); }
  .help-lines em { font-style: normal; font-size: 12px; color: var(--muted); }
  .help-lines a:hover strong { color: var(--deep); }
  .help p { margin: 0; font-size: 11.5px; color: var(--muted); }
  .ws-help { display: flex; flex-direction: column; gap: 4px; margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,.12); }
  .ws-help a { color: var(--white); text-decoration: none; font-size: 12.5px; }
  .ws-help a b { color: var(--marker); margin-right: 6px; font-family: var(--display); font-size: 15px; }
  .ws-voice { background: var(--black) !important; color: var(--white) !important; border-color: var(--black) !important; }
  .guide-map { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin: 14px 0 34px; }
  .guide-map > div { background: var(--white); border: 1px solid var(--line); border-top: 3px solid var(--black); padding: 14px 16px; }
  .guide-map b { display: block; font: 600 15px var(--display); letter-spacing: .08em; text-transform: uppercase; margin-bottom: 6px; }
  .guide-map span { font-size: 13.5px; color: var(--rock); line-height: 1.5; }
  .voice { position: fixed; inset: 0; z-index: 60; background: var(--black); color: var(--white); overflow-y: auto; }
  body.voice-on { overflow: hidden; }
  .voice-x { position: absolute; top: 18px; right: 22px; color: var(--muted-dark); font-size: 13px; letter-spacing: .1em; text-transform: uppercase; }
  .voice-x:hover { color: var(--white); }
  .voice-in { width: min(820px, calc(100% - 32px)); margin: 0 auto; padding: 70px 0 60px; text-align: center; }
  .voice .eyebrow { color: var(--muted-dark); }
  .voice-lang { display: flex; width: fit-content; border: 1px solid var(--line-dark); margin: 22px auto 34px; }
  .voice-lang button { padding: 8px 22px; color: var(--muted-dark); font-size: 15px; }
  .voice-lang button.on { background: var(--marker); color: var(--black); }
  .voice-mic { width: 148px; height: 148px; border-radius: 50%; background: var(--marker); color: var(--black); display: flex; margin: 0 auto;
               align-items: center; justify-content: center; position: relative; transition: transform .2s; }
  .voice-mic svg { width: 54px; height: 54px; }
  .voice-mic:hover { transform: scale(1.04); }
  .voice[data-state="listening"] .voice-mic { background: var(--white); }
  .voice[data-state="listening"] .voice-mic::before, .voice[data-state="listening"] .voice-mic::after {
    content: ''; position: absolute; inset: -10px; border-radius: 50%; border: 2px solid var(--marker); animation: vpulse 1.6s ease-out infinite; }
  .voice[data-state="listening"] .voice-mic::after { animation-delay: .8s; }
  @keyframes vpulse { from { transform: scale(1); opacity: .9; } to { transform: scale(1.45); opacity: 0; } }
  .voice[data-state="working"] .voice-mic { opacity: .5; }
  .voice-state { font-size: 20px; margin: 26px 0 8px; color: var(--marker); }
  .voice-heard { font-size: 24px; line-height: 1.45; margin: 0 0 18px; min-height: 1.5em; color: var(--white); }
  .voice-type { display: flex; gap: 8px; max-width: 560px; margin: 0 auto 18px; }
  .voice-ans { background: var(--beige); color: var(--ink); text-align: left; padding: 22px 26px; margin-top: 14px; font-size: 16px; }
  @media (prefers-reduced-motion: reduce) { .voice-mic::before, .voice-mic::after { animation: none !important; } }
  @media (max-width: 640px) { .bc-sum { grid-template-columns: repeat(3, 1fr); } .bc-sum p { grid-column: 1 / -1; } .bc-list li { grid-template-columns: 1fr; gap: 4px; } }

  .citator { margin-top: 16px; }
  .ct-sum { border: 1px solid var(--black); padding: 12px 14px; background: var(--white); }
  .ct-sum.warn { border-color: #a33a22; border-left-width: 4px; }
  .ct-sum b { display: block; font: 600 18px var(--display); letter-spacing: .04em; text-transform: uppercase; }
  .ct-sum.warn b { color: #a33a22; }
  .ct-sum span { font-size: 12.5px; color: var(--muted); }
  .citator ul { list-style: none; margin: 10px 0 0; padding: 0; }
  .citator li { display: grid; grid-template-columns: 104px 1fr; gap: 12px; padding: 12px 0; border-bottom: 1px solid var(--line); }
  .ct-tag { font: 600 11px var(--display); letter-spacing: .12em; text-transform: uppercase; padding: 3px 0; text-align: center; height: fit-content; border: 1px solid var(--black); }
  .ct-followed .ct-tag { background: var(--black); color: var(--white); }
  .ct-distinguished .ct-tag { background: var(--sand); }
  .ct-doubted .ct-tag { background: #a33a22; color: var(--white); border-color: #a33a22; }
  .citator li em { font-style: normal; font-size: 12px; color: var(--muted); margin-left: 8px; }
  .citator li p { margin: 4px 0; font-size: 13.5px; color: var(--rock); }
  .citator li q { display: block; font-size: 13px; color: var(--ink); border-left: 2px solid var(--black); padding-left: 10px; font-style: italic; }
  .citator .ct-noquote { font-size: 12px; color: #a33a22; }
  .rights h2 { font-size: 40px; margin: 8px 0 14px; }
  .rights[dir="rtl"] h2, .rights[dir="rtl"] .rights-do, .rights[dir="rtl"] .rights-list b { font-family: var(--text); text-transform: none; }
  .rights-bar { display: flex; gap: 0; border: 1px solid var(--marker); width: fit-content; margin-bottom: 16px; }
  .rights-bar button { padding: 6px 14px; font-size: 13px; color: var(--marker); }
  .rights-bar button.on { background: var(--marker); color: var(--black); }
  .rights-bar .rights-say { border-left: 1px solid var(--marker); font-weight: 600; }
  .rights-do { margin: 0 0 18px; font-size: 15px; background: #e4d9c6; color: var(--ink); padding: 12px 14px 12px 34px; }
  .rights[dir="rtl"] .rights-do { padding: 12px 34px 12px 14px; font-size: 17px; }
  .rights-do li { margin: 3px 0; }
  .rights-list { list-style: none; margin: 0; padding: 0; counter-reset: r; }
  .rights-list li { padding: 14px 0; border-top: 1px solid var(--line); }
  .rights-list b { display: block; font-size: 16px; margin-bottom: 6px; }
  .rights[dir="rtl"] .rights-list b { font-size: 19px; line-height: 1.7; }
  .rights-list blockquote { margin: 0 0 6px; padding: 8px 12px; background: var(--white); border-left: 3px solid var(--black); font-size: 13px; color: var(--rock); text-align: left; }
  .rights-list .statref { font-size: 12px; color: var(--marker); }
  .ws-own-new { border-style: dashed !important; }
  .ws-own .tag { color: #a33a22 !important; }

  .court { background: var(--black); color: var(--white); padding: 18px 20px 20px; margin: 18px 0; }
  .court-h { display: flex; justify-content: space-between; gap: 12px; font: 600 12px var(--display); letter-spacing: .14em; text-transform: uppercase;
             color: var(--marker); border-bottom: 1px solid var(--line-dark); padding-bottom: 10px; }
  .court-h b { color: var(--white); }
  .court-log { list-style: none; margin: 0; padding: 0; }
  .court-log li { padding: 14px 0; border-bottom: 1px solid var(--line-dark); animation: arrive .5s ease-out both; }
  .court-log .who { display: block; font: 600 11px var(--display); letter-spacing: .14em; text-transform: uppercase; margin-bottom: 6px; }
  .court-judge .who { color: var(--marker); }
  .court-judge .md { color: var(--white); font-size: 17px; }
  .court-judge .md a, .court-judge .md .cite { color: var(--marker); }
  .court-counsel { padding-left: 40px !important; }
  .court-counsel .who { color: var(--muted-dark); }
  .court-counsel p { margin: 0; color: #dfe6ee; font-size: 15px; }
  .court-ans { margin-top: 14px; }
  .court-ans .field { background: #161616; color: var(--white); border-color: var(--line-dark); }
  .court-ans .btn.line { color: var(--white); border-color: var(--line-dark); }
  .court-ans [data-a="suggest"] { border-color: var(--marker); color: var(--marker); }
  .court-ans .mic { color: var(--marker); border-color: var(--line-dark); background: transparent; }
  .court-review { background: var(--beige); color: var(--ink); }
  .court-review:not(:empty) { padding: 18px 20px; margin-top: 16px; }
  .mt-speak { display: block; margin: 4px 0 12px; font-size: 13.5px; }
  .ev-docs { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 6px 0 14px; font-size: 13.5px; }
  tr.ev-ok td:nth-child(3) { color: var(--black); font-weight: 600; }
  tr.ev-contra td { background: #f6e3dd; }
  tr.ev-contra td:nth-child(3) { color: #a33a22; font-weight: 700; }
  tr.ev-none td:nth-child(3) { color: #8a6d3b; font-weight: 600; }
  .ev-q { display: inline-block; font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; padding: 1px 5px; margin-left: 4px; }
  .ev-q.ok { border: 1px solid var(--black); }
  .ev-q.no { background: #a33a22; color: var(--white); }
  .airec { background: var(--white); border: 1px solid var(--black); padding: 18px 20px; margin: 10px 0 20px; }
  .airec h3 { margin: 4px 0 6px; }
  .airec p { font-size: 13.5px; color: var(--rock); margin: 0 0 12px; }
  .airec td { font-size: 13px; vertical-align: top; }
  .airec td i { font-style: normal; font-size: 10px; letter-spacing: .08em; text-transform: uppercase; background: var(--black); color: var(--white); padding: 0 4px; }
  .airec td i.rules { background: transparent; color: var(--black); border: 1px solid var(--black); }

  .gloss { font: inherit; color: inherit; background: none; border: 0; padding: 0; cursor: help;
           text-decoration: underline dotted; text-decoration-thickness: 1.5px; text-underline-offset: 3px; text-decoration-color: currentColor; }
  .gloss:hover, .gloss:focus-visible { background: rgba(216,205,185,.45); outline: none; }
  .gpop { position: fixed; z-index: 70; background: var(--white); color: var(--ink); border: 1px solid var(--black);
          box-shadow: 6px 6px 0 rgba(10,10,10,.85); padding: 14px 16px 12px; font-size: 14px; animation: arrive .25s ease-out both; }
  .gpop .gw { display: block; font: 700 20px var(--display); letter-spacing: .03em; text-transform: uppercase; margin-bottom: 4px; }
  .gpop p { margin: 0 0 8px; line-height: 1.5; }
  .gpop .gur { font-size: 16px; line-height: 1.8; color: var(--rock); border-top: 1px solid var(--line); padding-top: 6px; }
  .gpop blockquote { margin: 0 0 6px; padding: 7px 10px; background: var(--beige); border-left: 3px solid var(--black); font-size: 12.5px; color: var(--rock); }
  .gpop .statref { font-size: 12.5px; color: var(--deep); }
  .gpop small { display: block; margin-top: 8px; font-size: 11px; color: var(--muted); }

  .auto { position: fixed; left: 0; right: 0; bottom: 0; z-index: 80; background: var(--black); color: var(--white);
          box-shadow: 0 -8px 30px rgba(0,0,0,.25); }
  .auto-bar { height: 3px; background: #1c1c1c; }
  .auto-bar i { display: block; height: 100%; background: var(--marker); transition: width .6s; }
  .auto-in { display: flex; align-items: center; gap: 18px; padding: 14px 24px; max-width: 1400px; margin: 0 auto; }
  .auto-n { font: 700 14px var(--display); letter-spacing: .1em; color: var(--marker); white-space: nowrap; }
  .auto-in p { flex: 1; margin: 0; font-size: 17px; line-height: 1.45; animation: arrive .4s ease-out both; }
  .auto-keys { display: flex; gap: 6px; }
  .auto-keys button { color: var(--muted-dark); border: 1px solid var(--line-dark); padding: 5px 10px; font-size: 11px;
                      letter-spacing: .1em; text-transform: uppercase; }
  .auto-keys button:hover { color: var(--white); border-color: var(--marker); }
  body.auto-on { padding-bottom: 72px; }
  @media (max-width: 760px) { .auto-in { flex-wrap: wrap; gap: 8px; padding: 10px 14px; } .auto-in p { flex-basis: 100%; order: 3; font-size: 14px; } }

  .navmore { position: relative; flex: none; }
  .navmore > button { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); padding: 6px 0; }
  .navmore > button::after { content: ' \25BE'; }
  .navmore > button.on, .navmore > button:hover { color: var(--ink); }
  .navmenu { position: absolute; top: calc(100% + 12px); left: -14px; z-index: 30; background: var(--white); border: 1px solid var(--black);
             box-shadow: 6px 6px 0 rgba(10,10,10,.85); display: flex; flex-direction: column; min-width: 180px; padding: 6px 0; }
  .navmenu[hidden] { display: none; }
  .navmenu a { padding: 9px 16px; text-decoration: none; font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--rock); white-space: nowrap; }
  .navmenu a:hover { background: var(--beige); color: var(--ink); }
  .court-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 16px; }
  #vf-tabs { margin-bottom: 18px; }
  .vf-next { align-items: center; margin-top: 14px; }
  .vf-next .field { max-width: 220px; margin: 0; }
  .vf-papers { margin: 6px 0 12px; }
  .vf-paper { display: flex; gap: 12px; align-items: baseline; padding: 8px 0; border-bottom: 1px solid var(--line); }
  .vf-paper span { color: var(--muted); font-size: 12.5px; flex: 1; }
  .vf-vault { display: block; margin: 8px 0; font-size: 13.5px; }
  .vf-add { cursor: pointer; }
  .vf-citerow { display: flex; gap: 8px; }
  .vf-citerow .field { margin: 0; }
  .vf-chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 14px; }
  .vf-chips .eyebrow { margin-right: 6px; }
  .vf-chips button { border: 1px solid var(--line); background: var(--white); padding: 4px 10px; font-size: 12.5px; }
  .vf-chips button:hover { border-color: var(--black); }
  .vf-cite-h { margin: 18px 0 0; }
  .fillbar { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 16px; }
  .fillbox { flex-basis: 100%; background: var(--white); border: 1px solid var(--line); padding: 12px; }
  .fillbox[hidden] { display: none; }
  .fillnote { font-size: 13px; color: var(--muted); align-self: center; }
  .vv-name { font-size: 13.5px; color: var(--rock); align-self: center; }
  .vv-strip { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; margin: 12px 0 14px; }
  .vv-strip:empty { display: none; }
  .vv-strip figure { margin: 0; background: var(--black); animation: arrive .4s ease-out both; }
  .vv-strip img { display: block; width: 100%; aspect-ratio: 16 / 10; object-fit: cover; }
  .vv-check { margin: 22px 0 6px; }
  .vv-verdict { border: 1px solid var(--black); border-left-width: 6px; padding: 12px 16px; margin: 0 0 18px; background: var(--white); }
  .vv-verdict span { display: block; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
  .vv-verdict b { display: block; font: 700 26px var(--display); letter-spacing: .03em; text-transform: uppercase; margin: 2px 0; }
  .vv-verdict p { margin: 0; font-size: 14px; color: var(--rock); }
  .vv-ok { border-left-color: var(--black); }
  .vv-part { border-left-color: var(--sand); }
  .vv-no { border-left-color: #a33a22; } .vv-no b { color: #a33a22; }
  .vv-row { display: grid; grid-template-columns: 150px 1fr 150px; gap: 14px; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--line); }
  .vv-row img { width: 150px; aspect-ratio: 16 / 10; object-fit: cover; display: block; background: var(--black); }
  .vv-row b { font: 700 15px var(--display); letter-spacing: .06em; }
  .vv-row p { margin: 2px 0 0; font-size: 14px; color: var(--rock); }
  .vv-row em { font-style: normal; font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; text-align: center; padding: 3px 6px; border: 1px solid var(--black); }
  .vv-seen em { background: var(--black); color: var(--white); }
  .vv-partly em { background: var(--sand); }
  .vv-not-seen em { background: #a33a22; color: var(--white); border-color: #a33a22; }
  @media (max-width: 640px) { .vv-row { grid-template-columns: 100px 1fr; } .vv-row img { width: 100px; } .vv-row em { grid-column: 2; justify-self: start; } }
  .vv-strip figcaption { color: var(--marker); font: 600 12px var(--display); letter-spacing: .08em; padding: 4px 6px; }
  @media (max-width: 640px) { .court-grid { grid-template-columns: 1fr; } .vf-citerow { flex-direction: column; } }

  .vid { position: fixed; inset: 0; z-index: 75; background: rgba(10,10,10,.96); color: var(--white); overflow-y: auto; }
  .vid-in { display: flex; gap: 40px; align-items: center; justify-content: center; min-height: 100%; padding: 50px 24px; }
  .vid-stage canvas, .vid-stage video { height: min(78vh, 720px); aspect-ratio: 9 / 16; display: block; background: #000; box-shadow: 0 0 0 1px #2b2b2b, 0 20px 60px rgba(0,0,0,.6); }
  .vid-stage [hidden] { display: none; }
  .vid-side { max-width: 380px; }
  .vid-side .eyebrow { color: var(--muted-dark); }
  .vid-side h3 { font-size: 22px; margin: 8px 0 16px; }
  .vid-steps { list-style: none; margin: 0 0 18px; padding: 0; font-size: 14px; }
  .vid-steps li { padding: 8px 0 8px 22px; border-bottom: 1px solid #1c1c1c; position: relative; color: var(--muted-dark); }
  .vid-steps li::before { content: ''; position: absolute; left: 0; top: 14px; width: 9px; height: 9px; border-radius: 50%; border: 1px solid var(--marker); }
  .vid-steps li.on { color: var(--white); } .vid-steps li.on::before { background: var(--marker); animation: caret 1s steps(1) infinite; }
  .vid-steps li.done { color: #dfe6ee; } .vid-steps li.done::before { background: var(--marker); }
  .vid-acts { display: flex; flex-direction: column; gap: 8px; }
  .vid-acts .btn { text-align: center; text-decoration: none; }
  .vid-acts .btn:not(.line) { background: var(--marker); color: var(--black); border-color: var(--marker); }
  .vid-acts .btn.line { color: var(--white); border-color: #2b2b2b; }
  .vid-note { font-size: 11.5px; color: var(--muted-dark); margin-top: 16px; }
  .rights-bar .rights-vid { border-left: 1px solid var(--marker); font-weight: 600; }
  .vidbtn { font-weight: 600; }
  @media (max-width: 760px) { .vid-in { flex-direction: column; gap: 18px; padding: 50px 14px 30px; } .vid-stage canvas, .vid-stage video { height: 62vh; } }

  /* ---------- home ---------- */
  .home { display: grid; grid-template-columns: 1.35fr 1fr; min-height: calc(100vh - 56px);
          background: var(--slate); padding: 32px; }
  .home-l { background: #e4d9c6; color: var(--ink); padding: 44px 48px; display: flex; flex-direction: column; }
  .home-l .eyebrow { color: #6b6358; }
  .home-l h1 { font-size: clamp(88px, 13vw, 200px); margin: auto 0 18px; }
  .home-l p { max-width: 520px; color: var(--rock); margin: 0 0 28px; font-size: 16px; }
  .home-l .court { display: flex; gap: 0; margin-top: 30px; border-top: 1px solid rgba(0,0,0,.18); }
  .home-l .court div { flex: 1; padding: 14px 16px 0 0; }
  .home-l .court span { display: block; font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: #6b6358; }
  .home-l .court b { font-family: var(--display); font-size: 22px; font-weight: 600; }
  .home-l .ctas { display: flex; gap: 12px; flex-wrap: wrap; }
  .home-l .btn { background: var(--black); color: var(--white); border-color: var(--black); }
  .home-l .btn.line { background: transparent; color: var(--black); }
  .home-l .btn.line { transition: background .25s ease, color .25s ease, transform .25s ease, box-shadow .25s ease; }
  .home-l .btn.line:hover { background: var(--black); color: var(--white); opacity: 1; transform: translateY(-2px);
                            box-shadow: 0 6px 14px rgba(0, 0, 0, .18); }
  .home-l .btn.line:active { transform: translateY(0); box-shadow: none; }
  @media (prefers-reduced-motion: reduce) { .home-l .btn.line { transition: none; } .home-l .btn.line:hover { transform: none; } }
  /* landing motion: typed headline, rotating tagline, scroll reveal, hover polish */
  .hl { opacity: 0; }
  .hl.on { opacity: 1; }
  .hero-caret { display: inline-block; width: .06em; height: .72em; margin-inline-start: .03em; background: currentColor;
                vertical-align: baseline; animation: caretblink 1.1s steps(2, start) infinite; }
  @keyframes caretblink { to { visibility: hidden; } }
  .home-l .tagline { min-height: 32px; margin: 0 0 18px; font-size: 21px; color: var(--ink); font-family: var(--display);
                     font-weight: 600; font-stretch: 85%; letter-spacing: .01em; }
  .home-l .tagline i { display: inline-block; width: 2px; height: 1em; margin-inline-start: 3px; background: var(--ink);
                       vertical-align: text-bottom; opacity: .7; animation: caretblink 1.1s steps(2, start) infinite; }
  .reveal { opacity: 0; transform: translateY(18px); transition: opacity .7s ease, transform .7s ease; transition-delay: var(--d, 0s); }
  .reveal.in { opacity: 1; transform: none; }
  .tasks a .t { transition: transform .25s ease; }
  .tasks a:hover .t { transform: translateX(6px); }
  .tasks a:hover .n { color: var(--white); }
  nav a { transition: color .2s ease, border-color .2s ease; }
  nav a:hover:not(.on) { border-bottom-color: var(--muted); }
  @media (prefers-reduced-motion: reduce) {
    .hero-caret, .home-l .tagline i { animation: none; }
    .reveal { opacity: 1; transform: none; transition: none; }
    .tasks a .t, nav a { transition: none; } .tasks a:hover .t { transform: none; }
  }
  .home-r { background: var(--black); color: var(--white); padding: 44px 48px; display: flex; flex-direction: column; gap: 34px; }
  .home-r .eyebrow { color: var(--muted-dark); }
  .bignums { display: grid; grid-template-columns: 1fr 1fr; gap: 28px 24px; }
  .bignums b { display: block; font-size: 60px; }
  .bignums span { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted-dark); }
  .tasks { border-top: 1px solid var(--line-dark); }
  .tasks a { display: grid; grid-template-columns: 44px 1fr auto; align-items: baseline; gap: 12px; padding: 16px 0;
             border-bottom: 1px solid var(--line-dark); text-decoration: none; }
  .tasks a:hover .t { text-decoration: underline; text-underline-offset: 4px; }
  .tasks .n { font-family: var(--display); color: var(--muted-dark); }
  .tasks .t { font-size: 26px; }
  .tasks .d { color: var(--muted-dark); font-size: 13px; text-align: right; }
  .specs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: auto; }
  .specs span { display: block; font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted-dark); }
  .specs b { font-family: var(--display); font-size: 26px; font-weight: 600; }
  .specs .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--marker);
                margin-right: 9px; vertical-align: 4px; }
  @media (max-width: 900px) { .home { grid-template-columns: 1fr; padding: 14px; } .home-l h1 { margin-top: 60px; } }
  .trust { background: var(--slate); padding: 0 32px 32px; }
  .trust-in { background: var(--beige); padding: 40px 48px 34px; }
  .trust h2 { font-size: 56px; margin: 6px 0 26px; }
  .trust-grid { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--ink); }
  .trust-grid > div { padding: 18px 22px 6px 0; }
  .trust-grid > div + div { border-left: 1px solid var(--line); padding-left: 22px; }
  .trust-grid b { display: block; font-family: var(--display); font-size: 52px; font-weight: 700; line-height: 1;
                  color: var(--deep); font-variant-numeric: tabular-nums; }
  .trust-grid span { display: block; margin-top: 10px; font-size: 14px; color: var(--rock); max-width: 250px; }
  .trust-note { margin: 22px 0 0; font-size: 13px; color: var(--muted); max-width: 820px; }
  @media (max-width: 900px) { .trust { padding: 0 14px 14px; } .trust-in { padding: 28px 22px; }
    .trust-grid { grid-template-columns: 1fr 1fr; } .trust-grid > div:nth-child(3) { border-left: 0; padding-left: 0; } }

  /* ---------- page titles: the same slate band on every page ---------- */
  main > section { --band: var(--slate); --band-ink: var(--white); --band-soft: #e6eded; }
  .ptitle { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; flex-wrap: wrap;
            background: var(--band); color: var(--band-ink); padding: 34px 36px 30px; margin-bottom: 32px; }
  .ptitle .eyebrow { color: var(--band-soft); }
  .ptitle h1 { font-size: 72px; margin: 8px 0 0; }
  .ptitle p { margin: 0; max-width: 400px; color: var(--band-soft); font-size: 14px; }

  /* ---------- ask ---------- */
  .examples { list-style: none; margin: 0; padding: 0; border-top: 1px solid var(--line); }
  .examples li { border-bottom: 1px solid var(--line); }
  .examples button { width: 100%; text-align: start; display: grid; grid-template-columns: 44px 1fr; gap: 10px;
                     padding: 14px 0; align-items: baseline; }
  .examples button:hover .q { text-decoration: underline; text-underline-offset: 4px; }
  .examples .n { font-family: var(--display); color: var(--deep); }
  .examples button:hover { background: var(--paper); }
  .examples .tag { display: block; font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
  .turn { padding: 30px 0; border-bottom: 1px solid var(--line); }
  .turn:first-child { padding-top: 0; }
  .turn .q { font-size: 21px; font-weight: 600; line-height: 1.35; margin: 0 0 10px; }
  .meta { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted);
          display: flex; flex-wrap: wrap; gap: 6px 14px; margin-bottom: 14px; }
  .meta b { color: var(--ink); font-weight: 600; }
  .trace { margin: 0 0 18px; font-size: 13px; }
  .trace > summary { cursor: pointer; list-style: none; font-size: 11px; letter-spacing: .12em;
                     text-transform: uppercase; color: var(--muted); }
  .trace > summary::-webkit-details-marker { display: none; }
  .trace > summary::after { content: "  +"; }
  .trace[open] > summary::after { content: "  –"; }
  .steps { list-style: none; margin: 10px 0 0; padding: 0; border-top: 1px solid var(--line); }
  .step { display: grid; grid-template-columns: 34px 110px 1fr; gap: 10px; padding: 8px 0;
          border-bottom: 1px solid var(--line); align-items: baseline; }
  .step .n { font-family: var(--display); color: var(--deep); }
  .step .k { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; }
  .step .v { color: var(--muted); }
  .step .ms { display: none; }
  .thinking { animation: breathe 2.6s ease-in-out infinite; }
  @keyframes breathe { 0%, 100% { opacity: .4; } 50% { opacity: 1; } }
  @media (prefers-reduced-motion: reduce) { .thinking { animation: none; } }
  .step.pending .k::after { content: "_"; animation: blink 1s steps(1) infinite; }
  .step.planned .k, .step.planned .v, .step.planned .n { color: var(--muted); opacity: .55; }
  .step.planned .ms { font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); opacity: .7; }
  .step.fail .k { color: #b42318; }
  @keyframes blink { 50% { opacity: 0; } }
  @media (max-width: 640px) { .step { grid-template-columns: 28px 1fr; } .step .v { grid-column: 2 / 3; } }

  .composer { position: fixed; left: 0; right: 0; bottom: 0; background: var(--beige); border-top: 1px solid var(--line);
              padding: 14px 28px; display: none; }
  body.p-ask .composer { display: block; }
  .composer form { max-width: 860px; margin: 0 auto; display: flex; gap: 0; }
  .composer textarea { flex: 1; resize: none; border: 1px solid var(--ink); border-right: 0; padding: 11px 14px;
                       min-height: 46px; max-height: 160px; outline: none; background: var(--white); border-radius: 0; }
  .composer .btn { height: auto; }
  .scope { max-width: 860px; margin: 0 auto 10px; font-size: 13px; display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }
  .scope b { font-weight: 600; }
  .scope button { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); text-decoration: underline; }
  .scope button:hover { color: var(--ink); }
  /* voice input: shown only where the browser can recognise speech */
  .mic { display: inline-flex; align-items: center; justify-content: center; gap: 8px; min-width: 46px; padding: 0 12px;
         border: 1px solid var(--ink); background: var(--white); color: var(--ink); cursor: pointer; font: inherit;
         font-size: 12px; letter-spacing: .08em; text-transform: uppercase; }
  .composer .mic { border-right: 0; }
  .mic svg { width: 16px; height: 16px; flex: none; }
  .mic.rec { background: var(--deep); border-color: var(--deep); color: var(--beige); }
  .mic.rec svg { animation: blink 1.1s step-start infinite; }
  @media (prefers-reduced-motion: reduce) { .mic.rec svg { animation: none; } }
  .mic[hidden] { display: none; }
  .microw { display: flex; gap: 10px; align-items: center; margin-top: 8px; }
  .microw .mic { height: 34px; }
  .micnote { font-size: 12px; color: var(--muted); }

  /* ---------- answers ---------- */
  .rv-wait { display: none !important; }
  .rv-caret { display: inline-block; width: 7px; height: 1em; margin-inline-start: 2px; vertical-align: text-bottom;
              background: var(--ink); opacity: .55; animation: rvblink 1s steps(2, start) infinite; }
  @keyframes rvblink { to { visibility: hidden; } }
  @media (prefers-reduced-motion: reduce) { .rv-caret { animation: none; } }
  .md { font-size: 15.5px; }
  .md > :first-child { margin-top: 0; }
  .md p { margin: 0 0 12px; }
  .md h2, .md h3, .md h4 { font-family: var(--display); font-weight: 700; font-stretch: 75%;
                           font-variation-settings: "wdth" 75, "wght" 700; text-transform: uppercase;
                           font-size: 22px; line-height: 1; margin: 26px 0 10px; }
  .md ul, .md ol { margin: 0 0 12px; padding-inline-start: 20px; }
  .md li { margin: 4px 0; }
  .md li::marker { color: var(--muted); }
  .md strong { font-weight: 650; }
  .md code { font-size: 13px; background: var(--paper); padding: 0 4px; }
  .tablewrap { overflow-x: auto; margin: 6px 0 16px; border-top: 1px solid var(--ink); }
  .md table { border-collapse: collapse; width: 100%; font-size: 14px; }
  .md th, .md td { text-align: start; vertical-align: top; padding: 10px 12px 10px 0; border-bottom: 1px solid var(--line); }
  .md th { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; font-weight: 600; color: var(--muted); }
  .md td:first-child { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); width: 22%; }
  .numbered { counter-reset: sec; }
  .numbered h3 { counter-increment: sec; }
  .numbered h3::before { content: counter(sec, decimal-leading-zero); color: var(--slate); margin-right: 14px; }
  .md h3 { color: var(--deep); }
  .tablewrap { border-top-color: var(--band, var(--deep)); border-top-width: 2px; }
  .md th { background: var(--paper); padding-left: 10px; }
  .md td:first-child { padding-left: 10px; }
  .cite { font-size: 10.5px; font-weight: 700; vertical-align: super; line-height: 0; color: var(--accent);
          padding: 0 1px; margin-left: 1px; }
  .cite:hover, .cite.on { color: var(--ink); text-decoration: underline; }
  .error-text { color: #b42318; }

  .ground { margin-top: 16px; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted);
            display: flex; gap: 14px; flex-wrap: wrap; align-items: center; }
  .ground b { color: var(--ink); font-weight: 600; }
  .ground .bad { color: #b42318; }
  .ground .warn { color: #9a6512; }
  .md mark.unverified { background: none; color: inherit; border-bottom: 2px dotted #9a6512; cursor: help; }
  .factnote { margin-top: 10px; border-left: 2px solid #9a6512; padding: 6px 0 6px 14px; font-size: 13.5px; color: var(--muted); }
  .factnote p { margin: 0 0 6px; }
  .factnote ul { margin: 0; padding-left: 18px; }
  .factnote button { background: none; border: 0; padding: 0; font: inherit; color: var(--ink); text-decoration: underline;
                     text-underline-offset: 2px; cursor: pointer; }
  .ground .copy { margin-left: auto; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
  .ground .copy:hover { color: var(--ink); }
  .srclist { list-style: none; margin: 14px 0 0; padding: 0; border-top: 1px solid var(--line); }
  .srclist li { display: grid; grid-template-columns: 1fr auto; gap: 4px 16px; padding: 10px 0; border-bottom: 1px solid var(--line);
                font-size: 13.5px; color: var(--muted); }
  .srclist li.cited { color: var(--ink); }
  .srclist .who { font-weight: 600; }
  .srclist li:not(.cited) .who { font-weight: 400; }
  .srclist .cno { font-size: 12px; text-align: right; white-space: nowrap; }
  .srclist .paras { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 12px; }
  .srclist .paras button { color: inherit; border-bottom: 1px solid var(--line); }
  .srclist .paras button:hover, .srclist .paras button.on { color: var(--accent); border-bottom-color: var(--accent); }
  .srclist .paras button span { color: var(--accent); margin-right: 3px; }

  /* ---------- summarize / compare / library ---------- */
  .split { display: grid; grid-template-columns: 360px 1fr; gap: 48px; }
  @media (max-width: 900px) { .split { grid-template-columns: 1fr; } }
  .jlist { list-style: none; margin: 12px 0 0; padding: 0; border-top: 1px solid var(--line); max-height: calc(100vh - 290px); overflow-y: auto; }
  .filelist { list-style: none; margin: 10px 0 0; padding: 0; }
  .filelist li { display: flex; gap: 12px; align-items: baseline; justify-content: space-between; padding: 7px 0;
                 border-bottom: 1px solid var(--line); font-size: 13.5px; }
  .filelist li span { overflow-wrap: anywhere; }
  .filelist li button { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
  .filelist li button:hover { color: var(--ink); }
  .filelist li.ferr { color: #9a6512; border-bottom: 0; }
  .jlist button { width: 100%; text-align: start; padding: 11px 10px; border-bottom: 1px solid var(--line); display: block; }
  .jlist button b { display: block; font-weight: 600; font-size: 14px; line-height: 1.35; }
  .jlist button span { font-size: 12px; color: var(--muted); }
  .jlist button:hover { background: var(--paper); }
  .jlist button.on { background: var(--band, var(--deep)); color: var(--white); }
  .jlist button.on span { color: var(--band-soft, var(--deep-ink)); }
  .chosen h2 { font-size: 44px; margin: 8px 0 6px; }
  .chosen .sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .status { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin: 18px 0; min-height: 18px; }
  .status.working::after { content: "_"; animation: blink 1s steps(1) infinite; }
  .result { margin-top: 26px; }
  .pair { display: grid; grid-template-columns: 1fr 64px 1fr; gap: 16px; align-items: end; margin-bottom: 20px; }
  .pair label { display: block; margin-bottom: 8px; }
  .pair .vs { font-size: 34px; text-align: center; color: var(--slate); padding-bottom: 4px; }
  @media (max-width: 700px) { .pair { grid-template-columns: 1fr; } .pair .vs { display: none; } }

  .drop { display: flex; flex-direction: column; align-items: flex-start; gap: 8px; padding: 30px 28px;
          border: 1px dashed var(--slate); background: var(--white); cursor: pointer; }
  .drop:hover, .drop.over { border-style: solid; background: var(--paper); }
  .drop .big { font-size: 34px; color: var(--slate); }
  .drop .fname { font-weight: 600; color: var(--deep); word-break: break-all; }
  textarea.field { height: auto; padding: 10px 12px; resize: vertical; font: inherit; }
  .privacy { font-size: 12.5px; color: var(--muted); margin-top: 14px; }
  .mitem { border-top: 1px solid var(--ink); padding: 18px 0 26px; }
  .mitem header { all: unset; display: flex; gap: 14px; align-items: baseline; justify-content: space-between; }
  .mitem h3 { margin: 4px 0 10px; font-size: 17px; }
  .mtools { display: flex; gap: 10px; flex-wrap: wrap; margin: 6px 0 22px; }
  .mhead h2 { font-size: 48px; margin: 0; }
  #panel .mlist button { width: 100%; text-align: start; padding: 10px 0; border-bottom: 1px solid var(--line-dark); color: var(--white); }
  #panel .mlist button span { color: var(--muted-dark); font-size: 12px; margin-left: 8px; }
  .fb { display: inline-flex; gap: 10px; align-items: center; }
  .fb button { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
  .fb button:hover { color: var(--ink); }
  .hist { margin-top: 34px; }
  .hist li button { width: 100%; text-align: start; display: grid; grid-template-columns: 1fr auto; gap: 14px; padding: 12px 0;
                    border-bottom: 1px solid var(--line); }
  .hist li button span:last-child { font-size: 12px; color: var(--muted); }
  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--ink); margin-bottom: 26px; grid-column: 1 / -1; }
  .kpis div { padding: 14px 18px 4px 0; }
  .kpis div + div { border-left: 1px solid var(--line); padding-left: 18px; }
  .kpis b { display: block; font-family: var(--display); font-size: 40px; color: var(--deep); line-height: 1.1; }
  .kpis span { font-size: 12.5px; color: var(--muted); }
  @media (max-width: 640px) { .kpis { grid-template-columns: 1fr 1fr; } .kpis div:nth-child(3) { border-left: 0; padding-left: 0; } }
  .pmap svg { width: 100%; height: auto; display: block; background: var(--white); border: 1px solid var(--line); }
  .pmap .cap { min-height: 44px; margin: 12px 0 4px; font-size: 14px; }
  .pmap .cap b { font-weight: 600; }
  .pmap .cap span { color: var(--muted); }
  .pmap .key { display: flex; gap: 22px; flex-wrap: wrap; font-size: 12.5px; color: var(--muted); margin-top: 6px; }
  .pmap .key i { display: inline-block; width: 11px; height: 11px; margin-right: 7px; vertical-align: -1px; }
  .g-edge { stroke: #b9b1a0; stroke-width: 1; opacity: .75; }
  .g-j { fill: var(--deep); stroke: var(--white); stroke-width: 1.5; cursor: pointer; }
  .g-p { fill: var(--sand); stroke: var(--slate); stroke-width: 1.5; cursor: pointer; }
  .g-lbl { font: 10.5px var(--text); fill: var(--rock); pointer-events: none; paint-order: stroke; stroke: #fff; stroke-width: 3px; }
  @media (max-width: 640px) { .g-lbl { display: none; } }
  .g-dim { opacity: .12; }
  .g-on { stroke: var(--ink); stroke-width: 2.5; }
  .g-edge.g-on { stroke: var(--deep); stroke-width: 2; opacity: 1; }
  .ground .copy.listening { color: var(--ink); text-decoration: underline; }
  .pitch h2 { font-size: 44px; margin: 44px 0 14px; padding-top: 18px; border-top: 1px solid var(--ink); }
  .pitch .plain { list-style: none; margin: 0; padding: 0; max-width: 860px; }
  .pitch .plain li { padding: 12px 0; border-bottom: 1px solid var(--line); font-size: 15.5px; }
  .pitch .plain b { font-weight: 600; }
  .pitch .tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; border-top: 1px solid var(--line); border-left: 1px solid var(--line); }
  .pitch .tiles a { display: block; padding: 18px 20px 20px; text-decoration: none; color: var(--ink); border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }
  .pitch .tiles a:hover { background: var(--paper); }
  .pitch .tiles b { display: block; font-family: var(--display); font-size: 26px; font-weight: 600; margin-bottom: 6px; color: var(--deep); }
  .pitch .tiles span { font-size: 14px; color: var(--rock); }
  .pitch .steps { margin: 0; padding-left: 26px; max-width: 860px; list-style: decimal; }
  .pitch .steps li { padding: 8px 0 8px 6px; font-size: 15.5px; }
  .pitch .steps b { font-weight: 600; }
  .pitch .team { font-size: 16px; margin: 0; }
  @media (max-width: 900px) { .pitch .tiles { grid-template-columns: 1fr 1fr; } .pitch h2 { font-size: 34px; } }
  @media (max-width: 560px) { .pitch .tiles { grid-template-columns: 1fr; } }
  /* ---------- agent room: the orchestrator and its agents, as a live graph ---------- */
  .case-in { max-width: 860px; }
  .case-try { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; margin-top: 14px; font-size: 13.5px; }
  .case-try button { color: var(--accent); text-decoration: underline; text-underline-offset: 3px; text-align: start; }
  /* the stage: a dark control room where the agents work */
  .stage { margin: 30px 0 0; background: var(--black); color: #dfe6ee; padding: 18px 18px 14px; }
  .route { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 12px; margin-bottom: 12px; }
  .route .rt-l { font-family: var(--display); letter-spacing: .12em; text-transform: uppercase; color: rgba(223,230,238,.66); margin-right: 6px; }
  .route span[data-kind] { border: 1px solid rgba(255,255,255,.16); padding: 4px 10px; color: rgba(223,230,238,.66); letter-spacing: .06em;
                           text-transform: uppercase; font-size: 11px; transition: all .4s; }
  .route.deciding span[data-kind] { animation: deciding 1.4s ease-in-out infinite; }
  .route.deciding span[data-kind]:nth-of-type(2) { animation-delay: .15s; }
  .route.deciding span[data-kind]:nth-of-type(3) { animation-delay: .3s; }
  .route.deciding span[data-kind]:nth-of-type(4) { animation-delay: .45s; }
  .route.deciding span[data-kind]:nth-of-type(5) { animation-delay: .6s; }
  @keyframes deciding { 50% { border-color: #d8cdb9; color: #dfe6ee; } }
  .route span[data-kind].on { background: #d8cdb9; color: #0a0a0a; border-color: #d8cdb9; font-weight: 600; }
  .route span[data-kind].off { opacity: .35; }
  .room-plan { padding: 0 0 12px; font-size: 15px; color: #dfe6ee; }
  .room-plan b { font-family: var(--display); letter-spacing: .06em; font-weight: 600; margin-right: 8px; color: #d8cdb9; }
  .rgraph { position: relative; background: #111111; border: 1px solid rgba(255,255,255,.07);
            background-image: radial-gradient(rgba(255,255,255,.07) 1px, transparent 1px); background-size: 22px 22px; overflow: hidden; }
  .rgraph svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  .redge { fill: none; stroke: rgba(255,255,255,.13); stroke-width: 1.6; }
  .redge.used { stroke: #5c7174; stroke-width: 2.2; }
  .redge.flow { stroke: #d8cdb9; stroke-width: 2.4; stroke-dasharray: 7 9; animation: flow .9s linear infinite; }
  @keyframes flow { to { stroke-dashoffset: -32; } }
  .rdot { fill: #f4f0e8; filter: drop-shadow(0 0 6px #d8cdb9); }
  .rtrail { fill: #d8cdb9; }
  .rlabel { font: 600 11px var(--text); fill: #f4f0e8; paint-order: stroke; stroke: #0a0a0a; stroke-width: 5px; }
  .rchip { font: 11px var(--text); fill: #dfe6ee; paint-order: stroke; stroke: #0a0a0a; stroke-width: 5px; }
  .rnode { position: absolute; background: #1c1c1c; border: 1px solid rgba(255,255,255,.1); padding: 10px 12px 10px; cursor: pointer;
           max-height: 168px; overflow: hidden;
           transition: border-color .4s, background .4s, opacity .4s, box-shadow .4s; }
  .rnode .rh { display: grid; grid-template-columns: 34px 1fr auto; gap: 9px; align-items: center; }
  .rnode .ring { width: 34px; height: 34px; position: relative; }
  .rnode .ring svg { position: absolute; inset: 0; width: 34px; height: 34px; }
  .rnode .ring circle { fill: none; stroke-width: 2.5; }
  .rnode .ring .bg { stroke: rgba(255,255,255,.12); }
  .rnode .ring .arc { stroke: #d8cdb9; stroke-dasharray: 26 100; stroke-linecap: round; transform-origin: 17px 17px; opacity: 0; }
  .rnode .ring span { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
                      font: 600 11px var(--display); letter-spacing: .04em; color: #dfe6ee; }
  .rnode b { font-size: 13.5px; font-weight: 600; line-height: 1.2; color: #ffffff; }
  .rnode .rk { font-size: 9.5px; letter-spacing: .12em; text-transform: uppercase; padding: 1px 5px; }
  .rnode .rk.ai { background: #d8cdb9; color: #0a0a0a; }
  .rnode .rk.rules { border: 1px solid rgba(223,230,238,.66); color: #dfe6ee; }
  .rnode .rnote { font-size: 12px; margin: 7px 0 0; color: #dfe6ee; display: -webkit-box; -webkit-line-clamp: 2;
                  -webkit-box-orient: vertical; overflow: hidden; }
  .rnode .rmini { list-style: none; margin: 6px 0 0; padding: 0; font-size: 11.5px; color: #dfe6ee; }
  .rnode .rmini li { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding: 1px 0 1px 10px; position: relative;
                     animation: arrive .5s ease-out both; }
  .rnode .rmini li::before { content: ''; position: absolute; left: 0; top: 8px; width: 4px; height: 4px; background: #d8cdb9; }
  @keyframes arrive { from { opacity: 0; transform: translateX(-6px); } }
  .rnode .rmeter { height: 4px; background: rgba(255,255,255,.1); margin-top: 7px; }
  .rnode .rmeter i { display: block; height: 100%; background: #d8cdb9; transition: width .8s; }
  .rnode .rt { font: 11px var(--text); color: rgba(223,230,238,.66); font-variant-numeric: tabular-nums; }
  .rnode[data-status="waiting"] { opacity: .42; }
  .rnode[data-status="working"] { border-color: #d8cdb9; box-shadow: 0 0 0 1px #d8cdb9, 0 0 22px rgba(216,205,185,.25); background: #1c1c1c; }
  .rnode[data-status="working"] .ring .arc { opacity: 1; animation: spin 1s linear infinite; }
  .rnode[data-status="working"] .rt { color: #d8cdb9; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .rnode[data-status="done"] { border-color: #5c7174; }
  .rnode[data-status="done"] .ring .bg { stroke: #5c7174; }
  .rnode[data-status="done"] .ring span { color: #0a0a0a; background: #d8cdb9; border-radius: 50%; margin: 5px; font-size: 10px; }
  .rnode[data-status="skipped"] { border-style: dashed; opacity: .4; }
  .rnode[data-status="failed"] { border-color: #e2725b; }
  .rnode.ping { animation: ping .7s ease-out; }
  @keyframes ping { 0% { box-shadow: 0 0 0 0 rgba(216,205,185,.6); } 100% { box-shadow: 0 0 0 14px rgba(216,205,185,0); } }
  .livetext { white-space: pre-wrap; font-size: 14.5px; line-height: 1.65; border-left: 2px solid var(--deep); padding-left: 14px;
              max-height: 320px; overflow: hidden; display: flex; flex-direction: column-reverse; }
  .rnode .rlive { font-size: 11.5px; line-height: 1.45; margin: 6px 0 0; color: #f4f0e8; max-height: 3.2em; overflow: hidden; }
  .rnode .rlive::after, .scard .sbody.live::after { content: ''; display: inline-block; width: 6px; height: 1em; margin-left: 2px;
                                                     background: #d8cdb9; vertical-align: -2px; animation: caret .8s steps(1) infinite; }
  @keyframes caret { 50% { opacity: 0; } }
  .streams { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 10px; margin-top: 12px; }
  .scard { background: #1c1c1c; border: 1px solid rgba(255,255,255,.08); display: flex; flex-direction: column; min-height: 168px;
           transition: border-color .4s; }
  .scard[data-status="working"] { border-color: #d8cdb9; }
  .scard[data-status="waiting"], .scard[data-status="skipped"] { opacity: .45; }
  .scard .sh { display: flex; align-items: center; gap: 8px; padding: 7px 10px; border-bottom: 1px solid rgba(255,255,255,.07);
                  font-size: 12px; color: #dfe6ee; font-weight: 600; }
  .scard .sh i { width: 7px; height: 7px; border-radius: 50%; background: #5c7174; flex: none; }
  .scard[data-status="working"] .sh i { background: #d8cdb9; animation: caret 1s steps(1) infinite; }
  .scard[data-status="done"] .sh i { background: #d8cdb9; }
  .scard[data-status="failed"] .sh i { background: #e2725b; }
  .scard .sh span { margin-left: auto; font-weight: 400; color: rgba(223,230,238,.66); font-variant-numeric: tabular-nums; }
  .scard .sbody { padding: 8px 10px; font-size: 12px; line-height: 1.55; color: #dfe6ee; white-space: pre-wrap; overflow-y: auto;
                  max-height: 170px; flex: 1; scrollbar-width: thin; }
  .scard .sbody.idle { color: rgba(223,230,238,.5); font-style: italic; }
  .gantt { margin-top: 12px; font-size: 11.5px; }
  .gantt .gr { display: grid; grid-template-columns: 150px 1fr 54px; gap: 10px; align-items: center; padding: 2px 0; color: #dfe6ee; }
  .gantt .gt { position: relative; height: 9px; background: rgba(255,255,255,.05); }
  .gantt .gt i { position: absolute; top: 0; bottom: 0; background: #d8cdb9; min-width: 2px; }
  .gantt .gt i.rules { background: transparent; border: 1px solid rgba(223,230,238,.66); }
  .gantt .gt i.working { background-image: repeating-linear-gradient(45deg, rgba(13,23,34,.45) 0 4px, transparent 4px 8px); }
  .gantt .gr span:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  .room-key { display: flex; gap: 20px; flex-wrap: wrap; font-size: 12px; color: rgba(223,230,238,.66); margin: 12px 0 0; align-items: center; }
  .room-key i { display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; }
  .room-key .k-ai { background: #d8cdb9; }
  .room-key .k-rules { border: 1px solid rgba(223,230,238,.66); }
  .room-key .k-live { height: 3px; background: #d8cdb9; vertical-align: 3px; }
  .room-key .clock { margin-left: auto; font-family: var(--display); font-size: 16px; color: #dfe6ee; letter-spacing: .04em; }
  /* the pipeline: the agents behind any answer, as they work */
  .pipe { background: var(--black); color: #dfe6ee; padding: 12px 14px 12px; margin: 14px 0 18px; }
  .pipe .pl { display: flex; align-items: center; gap: 14px; font: 11px var(--display); letter-spacing: .12em; text-transform: uppercase;
              color: rgba(223,230,238,.66); margin-bottom: 10px; }
  .pipe .pl .pt { margin-left: auto; font-size: 14px; color: #dfe6ee; letter-spacing: .04em; font-variant-numeric: tabular-nums; }
  .pipe .pl button { font: inherit; letter-spacing: inherit; text-transform: inherit; color: #d8cdb9; background: none; border: 0;
                     border-bottom: 1px solid #5c7174; padding: 0; cursor: pointer; }
  .prow { display: flex; flex-wrap: wrap; align-items: center; row-gap: 10px; }
  .pcol { display: flex; flex-direction: column; gap: 6px; flex: 1 1 0; min-width: 140px; max-width: 240px; }
  .pconn { width: 24px; height: 2px; background: rgba(255,255,255,.16); flex: none; }
  .pconn.flow { background: repeating-linear-gradient(90deg, #d8cdb9 0 6px, transparent 6px 12px); background-size: 24px 2px;
                animation: pflow .6s linear infinite; }
  .pconn.used { background: #5c7174; }
  @keyframes pflow { to { background-position: 24px 0; } }
  .pnode { background: #1c1c1c; border: 1px solid rgba(255,255,255,.1); padding: 8px 10px; font-size: 12px;
           animation: arrive .4s ease-out both; transition: border-color .4s, box-shadow .4s, background .4s; }
  .pnode .ph { display: flex; align-items: center; gap: 7px; }
  .pnode .ph i { width: 8px; height: 8px; border-radius: 50%; background: #5c7174; flex: none; }
  .pnode .ph b { font-size: 12.5px; color: #ffffff; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .pnode .ph em { margin-left: auto; font-style: normal; font-size: 9px; letter-spacing: .12em; text-transform: uppercase; padding: 1px 5px; flex: none; }
  .pnode .ph em.ai { background: #d8cdb9; color: #0a0a0a; }
  .pnode .ph em.rules { border: 1px solid rgba(223,230,238,.66); color: #dfe6ee; }
  .pnode .pn { color: #dfe6ee; margin: 5px 0 0; line-height: 1.4; }
  .pnode .pm { list-style: none; margin: 4px 0 0; padding: 0; color: #dfe6ee; font-size: 11.5px; }
  .pnode .pm li { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; animation: arrive .5s ease-out both; }
  .pnode .pm li::before { content: '\00b7  '; color: #d8cdb9; }
  .pnode .plive { color: #f4f0e8; margin: 5px 0 0; line-height: 1.4; max-height: 2.9em; overflow: hidden; font-size: 11.5px; }
  .pnode .plive::after { content: ''; display: inline-block; width: 6px; height: 1em; margin-left: 2px; background: #d8cdb9;
                         vertical-align: -2px; animation: caret .8s steps(1) infinite; }
  .pnode .ptime { color: rgba(223,230,238,.66); font-size: 11px; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .pnode[data-status="working"] { border-color: #d8cdb9; box-shadow: 0 0 0 1px #d8cdb9, 0 0 18px rgba(216,205,185,.22); background: #1c1c1c; }
  .pnode[data-status="working"] .ph i { background: #d8cdb9; animation: caret 1s steps(1) infinite; }
  .pnode[data-status="done"] .ph i { background: #d8cdb9; }
  .pnode[data-status="failed"] { border-color: #e2725b; }
  .pnode[data-status="failed"] .ph i { background: #e2725b; }
  .pipe .streams { margin-top: 10px; }
  .streams[hidden] { display: none; }
  @media (max-width: 760px) { .prow { flex-direction: column; align-items: stretch; } .pconn { width: 2px; height: 14px; margin-left: 20px; }
                              .pconn.flow { background: repeating-linear-gradient(180deg, #d8cdb9 0 6px, transparent 6px 12px); background-size: 2px 24px; }
                              .pcol { max-width: none; } }
  @media (prefers-reduced-motion: reduce) { .stage *, .stage *::before, .pipe *, .pipe *::after { animation: none !important; } }
  @media (max-width: 760px) { .stage { padding: 12px 10px; } .gantt .gr { grid-template-columns: 100px 1fr 44px; } }
  .case-log { margin: 18px 0 8px; border-top: 1px solid var(--line); padding-top: 12px; }
  .case-log ol { list-style: none; margin: 8px 0 0; padding: 0; max-height: 200px; overflow-y: auto; font-size: 13px; }
  .case-log li { padding: 4px 0; border-bottom: 1px dotted var(--line); display: grid; grid-template-columns: 52px 1fr; gap: 10px; }
  .case-log li span { color: var(--muted); font-variant-numeric: tabular-nums; }
  .case-log li b { font-weight: 600; }
  .cfile { margin-top: 26px; }
  .cfile > h2 { font-size: 48px; margin: 0 0 4px; }
  .csec { border-top: 1px solid var(--ink); padding: 16px 0 24px; scroll-margin-top: 70px; }
  .csec > .eyebrow { margin-bottom: 8px; }
  .csec .wait { color: var(--muted); font-size: 14px; }
  .brief dl { display: grid; grid-template-columns: 130px 1fr; gap: 6px 16px; margin: 0; font-size: 14.5px; }
  .brief dt { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); padding-top: 3px; }
  .brief dd { margin: 0; }
  .slist { list-style: none; margin: 0; padding: 0; }
  .slist li { border-bottom: 1px solid var(--line); }
  .slist button { width: 100%; text-align: start; display: grid; grid-template-columns: 170px 1fr; gap: 12px; padding: 9px 0; font-size: 14px; }
  .slist button span:first-child { font-family: var(--display); color: var(--deep); }
  .slist button:hover { background: var(--paper); }
  @media (max-width: 760px) { .slist button { grid-template-columns: 1fr; gap: 2px; } .brief dl { grid-template-columns: 1fr; } }
  .stylebox { margin-top: 18px; border-top: 1px solid var(--line); padding-top: 12px; }
  .stylebox summary { cursor: pointer; font-size: 12px; letter-spacing: .1em; text-transform: uppercase; color: var(--ink); }
  .stylebox summary span { color: var(--muted); margin-left: 6px; }
  .stylebox label { display: block; margin: 12px 0 8px; }

  /* forms, guides, calculators */
  .form { display: grid; gap: 6px; margin-top: 14px; }
  .form label.eyebrow { margin-top: 8px; }
  .form.two { grid-template-columns: minmax(150px, 1fr) 1.4fr; column-gap: 16px; row-gap: 8px; align-items: center; }
  .form.two label.eyebrow { margin-top: 0; }
  #guide h2, #calc h2 { font-size: 46px; margin: 8px 0 16px; }
  #guide h3 { font-family: var(--display); font-weight: 700; font-stretch: 75%; font-variation-settings: "wdth" 75, "wght" 700;
              text-transform: uppercase; font-size: 18px; color: var(--deep); margin: 26px 0 10px; }
  .gsteps { margin: 0; padding-inline-start: 22px; color: var(--rock); }
  .gsteps li { margin: 7px 0; }
  .glaws { margin: 0; padding-inline-start: 18px; color: var(--rock); font-size: 14px; }
  .glaws li { margin: 4px 0; }
  #guide .ctas { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 26px; }
  .srclist button.who { text-align: start; color: inherit; }
  .srclist button.who:hover { color: var(--accent); text-decoration: underline; text-underline-offset: 3px; }
  .calcout { margin-top: 24px; }
  .bigline { display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap; padding: 16px 0 12px; margin-bottom: 14px;
             border-top: 2px solid var(--slate); border-bottom: 1px solid var(--line); }
  .bigline .display { font-size: 52px; color: var(--deep); }
  .bigline em { font-style: normal; font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
  .kvl { display: grid; grid-template-columns: 120px 1fr; gap: 6px 14px; font-size: 14px; margin: 0 0 16px; }
  .kvl dt { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); padding-top: 3px; }
  .kvl dd { margin: 0; }
  .calcout td.num { font-family: var(--display); font-size: 20px; color: var(--deep); }
  .md li.sev { list-style: none; margin-inline-start: -20px; padding: 8px 0 8px 12px; border-left: 3px solid var(--sand); }
  .md li.sev-high { border-left-color: #8a2b1e; }
  .md li.sev-medium { border-left-color: #9a6512; }
  .md li.sev-low { border-left-color: var(--slate); }
  .sevtag { display: inline-block; min-width: 62px; margin-right: 8px; font-size: 10.5px; font-weight: 700; letter-spacing: .12em;
            text-transform: uppercase; color: var(--muted); }
  .sev-high .sevtag { color: #8a2b1e; }
  .sev-medium .sevtag { color: #9a6512; }
  pre.code { background: var(--black); color: #e9e9e6; padding: 16px 18px; overflow-x: auto; font: 13px/1.6 Consolas, "Cascadia Mono", Menlo, monospace; margin: 0 0 8px; }
  .apih { font-family: var(--display); font-weight: 700; font-stretch: 75%; font-variation-settings: "wdth" 75, "wght" 700;
          text-transform: uppercase; font-size: 18px; color: var(--deep); margin: 30px 0 10px; }
  .apitable code, .privacy code { font: 12.5px Consolas, "Cascadia Mono", Menlo, monospace; background: var(--paper); padding: 0 4px; }
  .apitable td:first-child { font-family: var(--display); color: var(--deep); }
  .devlink { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted-dark); text-decoration: none; }
  .devlink:hover { color: var(--white); }
  .insights { display: grid; grid-template-columns: 1fr 1fr; gap: 26px 48px; }
  @media (max-width: 900px) { .insights { grid-template-columns: 1fr; } }
  .ipanel h3 { font-family: var(--display); font-weight: 700; font-stretch: 75%; font-variation-settings: "wdth" 75, "wght" 700;
               text-transform: uppercase; font-size: 20px; color: var(--deep); margin: 0 0 4px; }
  .inote { font-size: 12.5px; color: var(--muted); margin: 0 0 10px; }
  .brow { display: grid; grid-template-columns: minmax(120px, 44%) 1fr 30px; gap: 10px; align-items: center; width: 100%;
          padding: 5px 0; border-bottom: 1px solid var(--line); text-align: start; font-size: 13px; color: var(--rock); }
  button.brow:hover { background: var(--paper); }
  .brow .bl { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .brow .bt { height: 10px; background: transparent; }
  .brow .bt i { display: block; height: 100%; background: var(--slate); border-radius: 0 4px 4px 0; }
  button.brow:hover .bt i { background: var(--deep); }
  .brow .bv { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink); }
  .colchips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 4px; }
  .colchips button { font-size: 13px; padding: 6px 12px; border: 1px solid var(--sand); background: var(--white); }
  .colchips button.on { background: var(--slate); border-color: var(--slate); color: var(--white); }
  table.rtable { border-collapse: collapse; width: 100%; font-size: 13.5px; }
  .rtable th { text-align: start; font-size: 11px; letter-spacing: .1em; text-transform: uppercase; font-weight: 600;
               background: var(--paper); padding: 10px; border-bottom: 1px solid var(--sand); vertical-align: bottom; }
  .rtable td { padding: 10px; border-bottom: 1px solid var(--line); vertical-align: top; color: var(--rock); }
  .rtable td:first-child { min-width: 180px; }
  .rtable td:first-child b { display: block; color: var(--ink); font-weight: 600; }
  .rtable td:first-child span { font-size: 12px; color: var(--muted); }
  .rtable td.ns { color: var(--muted); font-style: italic; }
  .statref { color: inherit; font: inherit; border-bottom: 1px dotted var(--accent); display: inline; text-align: inherit;
             line-height: inherit; white-space: normal; }
  .statref:hover { color: var(--accent); border-bottom-style: solid; }
  #panel .statref { border-bottom-color: var(--marker); }
  .libtabs { display: flex; gap: 22px; margin-right: 10px; }
  .libtabs button { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted);
                    padding: 6px 0; border-bottom: 2px solid transparent; }
  .libtabs button.on { color: var(--ink); border-bottom-color: var(--deep); }
  .md.editable { outline: 1px dashed var(--sand); outline-offset: 10px; }
  .md.editable:focus { outline: 1px solid var(--slate); }
  .followup { margin-top: 30px; padding-top: 18px; border-top: 2px solid var(--slate); }
  .fform { display: flex; margin: 10px 0 8px; }
  .fform .field { border-right: 0; }
  .fitem { padding: 14px 0; border-bottom: 1px solid var(--line); }
  .fitem .q { font-weight: 600; margin: 0 0 8px; }

  /* Save as PDF: print only the chosen result */
  #print-area { display: none; }
  @media print {
    body.printing > *:not(#print-area) { display: none !important; }
    body.printing { background: #fff; }
    body.printing #print-area { display: block; }
    #print-area h1 { font-size: 34px; margin: 6px 0 18px; }
    #print-area .copy { display: none; }
  }

  table.lib { width: 100%; border-collapse: collapse; font-size: 14px; }
  table.lib th { text-align: start; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--ink);
                 font-weight: 600; padding: 10px 14px 10px 10px; background: var(--paper); border-bottom: 1px solid var(--sand); }
  table.lib td:first-child { padding-left: 10px; }
  table.lib td.num { color: var(--deep); }
  table.lib td { padding: 12px 14px 12px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
  table.lib tbody tr { cursor: pointer; }
  table.lib tbody tr:hover td { background: var(--paper); }
  table.lib td.num { font-family: var(--display); font-size: 20px; font-variant-numeric: tabular-nums; }
  table.lib .who { font-weight: 600; }
  table.lib .small { font-size: 12.5px; color: var(--muted); }
  .libtools { display: flex; gap: 16px; align-items: center; margin-bottom: 18px; }
  .libtools .field { max-width: 420px; }
  @media (max-width: 800px) { table.lib .hide-sm { display: none; } }

  /* ---------- black side panel (passage and judgment detail) ---------- */
  #panel { position: fixed; top: 0; right: 0; bottom: 0; width: min(520px, 100vw); z-index: 40; overflow-y: auto;
           background: var(--black); color: var(--white); padding: 34px 36px 48px;
           transform: translateX(100%); transition: transform .22s ease; }
  body.panel-open #panel { transform: none; }
  #panel .eyebrow { color: var(--muted-dark); }
  #panel .close { position: absolute; top: 22px; right: 26px; font-size: 11px; letter-spacing: .14em;
                  text-transform: uppercase; color: var(--muted-dark); }
  #panel .close:hover { color: var(--white); }
  #panel h2.big { font-size: 96px; margin: 18px 0 6px; }
  #panel h3 { font-size: 20px; font-weight: 600; line-height: 1.3; margin: 0 0 4px; }
  #panel .sub { color: var(--muted-dark); font-size: 13px; }
  #panel .kv { display: grid; grid-template-columns: 120px 1fr; gap: 8px 14px; font-size: 13.5px; margin: 24px 0;
               padding-top: 18px; border-top: 1px solid var(--line-dark); }
  #panel .kv dt { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted-dark); padding-top: 2px; }
  #panel .kv dd { margin: 0; }
  #panel .why { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; padding: 18px 0;
                border-top: 1px solid var(--line-dark); border-bottom: 1px solid var(--line-dark); margin-bottom: 22px; }
  #panel .why span { display: block; font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted-dark); }
  #panel .why b { font-family: var(--display); font-size: 40px; font-weight: 700; font-variant-numeric: tabular-nums; }
  #panel .note a { overflow-wrap: anywhere; }
  #panel .note { color: var(--muted-dark); font-size: 13px; border-top: 1px solid var(--line-dark);
                 border-bottom: 1px solid var(--line-dark); padding: 14px 0; margin-bottom: 22px; }
  #panel .passage { white-space: pre-wrap; font-size: 14px; line-height: 1.75; color: #e9e9e6; }
  #panel .passage mark { background: none; color: var(--white); border-bottom: 2px solid var(--marker); }
  #panel h2.big::after { content: ""; display: inline-block; width: 12px; height: 12px; border-radius: 50%;
                         background: var(--marker); margin-left: 14px; vertical-align: top; margin-top: 10px; }
  #panel .plist { list-style: none; margin: 10px 0 0; padding: 0; border-top: 1px solid var(--line-dark); }
  #panel .plist li { display: grid; grid-template-columns: 64px 1fr; gap: 10px; padding: 10px 0; border-bottom: 1px solid var(--line-dark);
                     font-size: 13px; color: #cfcfcb; }
  #panel .plist li b { font-family: var(--display); color: var(--white); font-size: 16px; }
  .citeref { display: inline; padding: 0; border: 0; background: none; font: inherit; color: inherit; cursor: pointer;
             border-bottom: 1px dotted currentColor; }
  .citeref:hover { color: var(--accent); }
  #panel .citeref:hover, #panel .jlink:hover { color: var(--marker); }
  #panel .cchips { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 10px 0 24px; font-size: 13.5px; color: #cfcfcb; }
  #panel .cchips i { font-style: normal; font-size: 11px; color: var(--black); background: var(--marker); padding: 0 5px; margin-left: 3px; }
  #panel .jlink { padding: 0; border: 0; background: none; font: inherit; color: var(--white); cursor: pointer; text-align: start;
                  border-bottom: 1px solid var(--line-dark); }
  #panel .plist small { color: var(--muted-dark); font-size: 12px; }
  #panel .acts { display: flex; gap: 10px; margin: 22px 0 26px; flex-wrap: wrap; }
  #panel .acts .btn { background: var(--white); color: var(--black); border-color: var(--white); }
  #panel .acts .btn.line { background: transparent; color: var(--white); }
  .tour { position: fixed; right: 22px; bottom: 22px; z-index: 60; width: min(390px, calc(100vw - 28px)); background: var(--deep);
          color: var(--beige); padding: 20px 22px 18px; box-shadow: 0 10px 40px rgba(0,0,0,.28); }
  body.p-ask .tour { bottom: 104px; }
  .tour .eyebrow { color: #aebccb; }
  .tour h3 { margin: 6px 0 8px; font-family: var(--display); font-size: 24px; font-weight: 600; letter-spacing: .01em; }
  .tour p { margin: 0 0 16px; font-size: 14px; line-height: 1.55; color: #dfe6ee; }
  .tour-b { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .tour .btn { height: 38px; padding: 0 18px; background: var(--beige); color: var(--deep); border-color: var(--beige); }
  .tour .btn.line { background: transparent; color: var(--beige); border-color: #7f93a8; }
  .tour .btn:disabled { opacity: .4; }
  .tour-x { margin-left: auto; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: #aebccb; text-decoration: underline; }
  .tour-hl { outline: 3px solid #d8cdb9; outline-offset: 5px; box-shadow: 0 0 0 9999px rgba(20, 30, 42, .28); position: relative; z-index: 5; }
  @media (max-width: 640px) { .tour { right: 14px; left: 14px; width: auto; bottom: 14px; } body.p-ask .tour { bottom: 96px; } }
  .scrim { position: fixed; inset: 0; z-index: 35; background: rgba(0,0,0,.25); display: none; }
  body.panel-open .scrim { display: block; }
</style>
</head>
<body>
<header>
  <a href="/" data-link class="mark display">Wakeel AI</a>
  <nav>
    <a href="/workspace" data-link data-nav="workspace">Workspace</a>
    <a href="/ask" data-link data-nav="ask">Ask</a>
    <a href="/case" data-link data-nav="case">Agent room</a>
    <a href="/court" data-link data-nav="court">Courtroom</a>
    <a href="/verify" data-link data-nav="verify">Verify</a>
    <a href="/explain" data-link data-nav="explain">Explain</a>
    <a href="/draft" data-link data-nav="draft">Draft</a>
    <a href="/guides" data-link data-nav="guides">Guides</a>
    <a href="/library" data-link data-nav="library">Library</a>
  </nav>
  <div class="navmore" id="navmore"><button type="button" aria-expanded="false" aria-haspopup="true">More</button>
    <div class="navmenu" hidden>
      <a href="/summarize" data-link data-more="summarize">Summarize</a><a href="/compare" data-link data-more="compare">Compare</a>
      <a href="/calculate" data-link data-more="calculate">Calculate</a><a href="/matters" data-link data-more="matters">Matters</a>
      <a href="/pitch" data-link data-more="pitch">The pitch</a></div></div>
  <span class="grow"></span>
  <div class="toggle" id="mode" role="radiogroup" aria-label="Who the answer is written for">
    <span class="lbl">For</span>
    <button type="button" data-value="citizen" class="on">Citizen</button><span class="sep">/</span>
    <button type="button" data-value="lawyer">Lawyer</button>
  </div>
  <div class="toggle" id="language" role="radiogroup" aria-label="Language of the answer">
    <span class="lbl">In</span>
    <button type="button" data-value="en" class="on">English</button><span class="sep">/</span>
    <button type="button" data-value="ur" lang="ur">اردو</button><span class="sep">/</span>
    <button type="button" data-value="roman_ur">Roman Urdu</button>
  </div>
</header>

<main>
  <!-- HOME -->
  <section data-page="home">
    <div class="home">
      <div class="home-l">
        <div class="eyebrow">Lahore High Court · Pakistani case law</div>
        <h1 class="display" id="hero-title">Wakeel<br>AI</h1>
        <div class="tagline" aria-hidden="true"><span id="hero-tag"></span><i></i></div>
        <p>Legal research on real court judgments. Every answer is written from the judgments
           themselves and linked to the paragraph it came from, for citizens and lawyers, in
           English, Urdu or Roman Urdu.</p>
        <form class="live-ask" id="live-form">
          <label class="eyebrow" for="live-text">Describe what happened, and watch the agents work</label>
          <textarea id="live-text" rows="2" dir="auto" placeholder="In English, اردو or Roman Urdu: my cheque bounced, my bail was cancelled, my landlord…"></textarea>
          <div class="live-bar"><div class="live-try" id="live-try"></div><button class="mic" id="live-mic" type="button" hidden></button>
            <button class="btn" type="submit">Start the agents</button></div>
        </form>
        <div class="ctas">
          <a href="/workspace?who=lawyer" data-link class="btn line">Lawyer workspace</a>
          <a href="/workspace?who=people" data-link class="btn line">My case workspace</a>
          <a href="/case" data-link class="btn line">Agent room</a>
          <button type="button" class="btn line" id="tour-start">Take the 90-second tour</button>
          <button type="button" class="btn line" id="rights-open">Arrested or stopped? Rights now</button>
          <button type="button" class="btn line" id="voice-open">Voice mode · بول کر پوچھیں</button>
        </div>
        <div class="court">
          <div><span>Court</span><b>Lahore High Court</b></div>
          <div><span>Year</span><b>2023</b></div>
          <div><span>Answers in</span><b>EN · اردو · Roman</b></div>
        </div>
      </div>
      <div class="home-r">
        <div class="eyebrow">Knowledge base</div>
        <div class="netblock">
          <canvas class="netcanvas" id="home-net" aria-hidden="true"></canvas>
          <div class="bignums">
          <div><b class="display" data-stat="judgments">–</b><span>Judgments</span></div>
          <div><b class="display" data-stat="passages">–</b><span>Indexed passages</span></div>
          <div><b class="display" data-stat="citations">–</b><span>Case citations</span></div>
          <div><b class="display" data-stat="provisions">–</b><span>Statutory provisions</span></div>
        </div>
        </div>
        <ol class="live-feed" id="live-feed" aria-live="off"></ol>
        <div class="tasks">
          <a href="/case" data-link><span class="n">00</span><span class="t display">Agent room</span><span class="d">An orchestrator and a team of agents for any request</span></a>
          <a href="/ask" data-link><span class="n">01</span><span class="t display">Ask</span><span class="d">A question, answered with citations</span></a>
          <a href="/summarize" data-link><span class="n">02</span><span class="t display">Summarize</span><span class="d">One judgment, section by section</span></a>
          <a href="/compare" data-link><span class="n">03</span><span class="t display">Compare</span><span class="d">Two judgments, side by side</span></a>
          <a href="/explain" data-link><span class="n">04</span><span class="t display">Explain</span><span class="d">Your FIR, notice or court order, in plain words</span></a>
          <a href="/draft" data-link><span class="n">05</span><span class="t display">Draft</span><span class="d">Notices and applications; hearing prep, timelines, client letters</span></a>
          <a href="/calculate" data-link><span class="n">06</span><span class="t display">Calculate</span><span class="d">Limitation periods and inheritance shares</span></a>
        </div>
        <a href="/pitch" data-link class="devlink">For judges: the pitch →</a>
        <a href="/developers" data-link class="devlink">For developers: the API →</a>
        <div class="specs">
          <div><span>Search</span><b>Hybrid</b></div>
          <div><span>Languages</span><b>3</b></div>
          <div><span>Model</span><b><i class="dot"></i><span id="model-short" style="display:inline;font-size:inherit;letter-spacing:0;text-transform:none;color:inherit">–</span></b></div>
        </div>
      </div>
    </div>
    <div class="solve" id="solve">
      <div class="solve-in">
        <div class="eyebrow">What it solves</div>
        <h2 class="display">The problems people bring to a lawyer, and the hours lawyers lose.<span>Each one has an agent on it. Press any card to watch it solved, live.</span></h2>
        <div class="solve-tabs" id="solve-tabs" role="tablist"></div>
        <div class="solve-grid" id="solve-grid"></div>
      </div>
    </div>
    <div class="trust">
      <div class="trust-in">
        <div class="eyebrow">Measured, not claimed</div>
        <h2 class="display">Why you can trust it</h2>
        <div class="trust-grid">
          <div><b data-count="99.5" data-dec="1" data-suffix="%">99.5%</b><span>of test questions find the right judgment in the top ten results</span></div>
          <div><b data-count="0.97" data-dec="2">0.97</b><span>mean reciprocal rank: the right judgment almost always comes first</span></div>
          <div><b>1 click</b><span>from any claim in an answer to the paragraph it came from</span></div>
          <div><b>No AI</b><span>in the fact check: every name, section and figure is looked for in its cited paragraph</span></div>
        </div>
        <p class="trust-note">Search measured on 50 test questions across the indexed judgments. When the
           judgments do not answer a question, Wakeel AI says so instead of guessing. Statutes are
           shown word for word with their source. Uploaded documents are read for one answer and
           never stored.</p>
      </div>
    </div>
  </section>

  <!-- ASK -->
  <section data-page="ask">
    <div class="wrap">
      <div class="ptitle">
        <div><div class="eyebrow">01 · Research agent</div><h1 class="display">Ask</h1></div>
        <p>Ask in plain words. A router reads the question, sends it to the right agent, and the
           answer cites the paragraphs it came from.</p>
      </div>
      <div id="thread"></div>
    </div>
  </section>

  <!-- SUMMARIZE -->
  <section data-page="summarize">
    <div class="wrap wide">
      <div class="ptitle">
        <div><div class="eyebrow">02 · Summary agent</div><h1 class="display">Summarize</h1></div>
        <p>Choose a judgment. The summary agent reads all of it and returns the facts, issues,
           arguments, decision and why it matters, each linked to its paragraph.</p>
      </div>
      <div class="split">
        <div>
          <label class="btn line" for="sum-file" style="margin-bottom:18px;cursor:pointer">Upload a judgment (PDF or photo)</label>
          <input type="file" id="sum-file" accept=".pdf,image/png,image/jpeg,image/webp" hidden>
          <label class="eyebrow" for="sum-search" style="display:block">Or choose from the library</label>
          <input class="field" id="sum-search" placeholder="Search parties, case number or provision" style="margin-top:8px">
          <ul class="jlist" id="sum-list"></ul>
        </div>
        <div class="chosen">
          <div class="eyebrow" id="sum-court">No judgment selected</div>
          <h2 class="display" id="sum-title">Choose from the list</h2>
          <div class="sub" id="sum-sub"></div>
          <button class="btn" id="sum-go" disabled>Summarize</button>
          <div class="status" id="sum-status"></div>
          <div class="result" id="sum-result"></div>
        </div>
      </div>
    </div>
  </section>

  <!-- COMPARE -->
  <section data-page="compare">
    <div class="wrap wide">
      <div class="ptitle">
        <div><div class="eyebrow">03 · Compare agent</div><h1 class="display">Compare</h1></div>
        <p>Choose two judgments. The compare agent sets them side by side: court, parties,
           legal question, provisions and outcome, then the similarities and differences.</p>
      </div>
      <div class="libtabs" id="cmp-tabs" style="margin-bottom:22px">
        <button type="button" class="on" data-tab="two">Two judgments</button>
        <button type="button" data-tab="table">Review table · many documents</button>
      </div>
      <div id="cmp-two">
        <div class="pair">
          <div><label class="eyebrow" for="cmp-a">Judgment A</label><select class="field" id="cmp-a"></select></div>
          <div class="vs display">vs</div>
          <div><label class="eyebrow" for="cmp-b">Judgment B</label><select class="field" id="cmp-b"></select></div>
        </div>
        <button class="btn" id="cmp-go">Compare</button>
        <div class="status" id="cmp-status"></div>
        <div class="result" id="cmp-result"></div>
      </div>
      <div id="cmp-table" hidden>
        <div class="split">
          <div>
            <div class="eyebrow" id="rt-count">Judgments · choose up to 8</div>
            <input class="field" id="rt-search" placeholder="Search parties, case number or provision" style="margin-top:8px">
            <ul class="jlist" id="rt-list" style="max-height:440px"></ul>
            <label class="btn line" for="rt-file" style="margin-top:16px;cursor:pointer">Add your own documents</label>
            <input type="file" id="rt-file" accept=".pdf,image/png,image/jpeg,image/webp" multiple hidden>
            <ul class="filelist" id="rt-files"></ul>
            <p class="privacy">Contracts, FIRs, notices or orders: PDF or photo, 10 MB in all. Read for this
               table only, never saved.</p>
          </div>
          <div>
            <div class="eyebrow">Columns · up to 6</div>
            <div class="colchips" id="rt-cols"></div>
            <form class="fform" id="rt-addcol"><input class="field" id="rt-newcol" placeholder="Add your own column, e.g. Was bail granted?" maxlength="80">
              <button class="btn" type="submit">Add</button></form>
            <p class="privacy">Each judgment or document is read in full and every cell cites the paragraph it came from.</p>
            <button class="btn" id="rt-go" disabled>Build table</button>
            <div class="status" id="rt-status"></div>
          </div>
        </div>
        <div class="result" id="rt-result"></div>
      </div>
    </div>
  </section>

  <!-- EXPLAIN -->
  <section data-page="explain">
    <div class="wrap wide">
      <div class="ptitle">
        <div><div class="eyebrow">04 · Document agent</div><h1 class="display">Explain</h1></div>
        <p>Upload an FIR, legal notice, summons or court order. The document agent explains it
           in plain words, lists the laws it mentions and finds similar judgments.</p>
      </div>
      <div class="split">
        <div>
          <label class="drop" id="drop">
            <input type="file" id="doc-file" accept=".pdf,image/png,image/jpeg,image/webp" hidden>
            <span class="display big">Choose a file</span>
            <span class="eyebrow">or drop it here · PDF or photo · up to 10 MB</span>
            <span class="fname" id="doc-name"></span>
          </label>
          <label class="eyebrow" for="doc-text" style="display:block;margin:22px 0 8px">Or paste the text</label>
          <textarea class="field" id="doc-text" rows="7" dir="auto"></textarea>
          <div class="microw"><button class="mic" id="doc-mic" type="button" hidden></button>
            <span class="micnote" id="doc-micnote"></span></div>
          <button class="btn" id="doc-go" style="margin-top:14px">Explain</button>
          <p class="privacy">Your document is read for this answer only. It is not saved and is
             never added to the library.</p>
        </div>
        <div>
          <div class="status" id="doc-status" style="margin-top:0"></div>
          <div class="result" id="doc-result" style="margin-top:0"></div>
        </div>
      </div>
    </div>
  </section>

  <!-- DRAFT -->
  <section data-page="draft">
    <div class="wrap wide">
      <div class="ptitle">
        <div><div class="eyebrow">05 · Drafting agent</div><h1 class="display">Draft</h1></div>
        <p>Choose a document or a workflow (strategy memo, hearing preparation, chronology, client
           letter), fill in what you know, and review the result with your advocate. Anything you
           leave out stays as a [placeholder]; nothing is invented.</p>
      </div>
      <div class="split">
        <div>
          <div class="eyebrow">Document or workflow</div>
          <ul class="jlist" id="draft-kinds" style="max-height:none"></ul>
          <div id="draft-form" class="form"></div>
          <details class="stylebox" id="draft-style" hidden>
            <summary>Write it in my own style <span>optional</span></summary>
            <label class="eyebrow" for="draft-sample">Paste a notice or petition you have used before</label>
            <textarea class="field" id="draft-sample" rows="6" dir="auto"></textarea>
            <p class="privacy">The draft follows its layout, headings and tone. No names, dates or amounts
               are taken from it, and it is not saved.</p>
          </details>
          <button class="btn" id="draft-go" disabled style="margin-top:16px">Draft</button>
        </div>
        <div>
          <div class="status" id="draft-status" style="margin-top:0"></div>
          <div class="result" id="draft-result" style="margin-top:0"></div>
        </div>
      </div>
    </div>
  </section>

  <!-- GUIDES -->
  <section data-page="guides">
    <div class="wrap wide">
      <div class="ptitle">
        <div><div class="eyebrow">Know your rights</div><h1 class="display">Guides</h1></div>
        <p>Common legal problems, step by step: what to do, the laws involved, and real
           judgments from the library. General information, not legal advice.</p>
      </div>
      <div class="ctas" style="margin:-10px 0 24px"><button type="button" class="btn" id="rights-open-2">Arrested or stopped? Your rights right now</button></div>
      <div class="split">
        <ul class="jlist" id="guide-list" style="max-height:none;margin-top:0"></ul>
        <div class="guide" id="guide"></div>
      </div>
    </div>
  </section>

  <!-- CALCULATE -->
  <section data-page="calculate">
    <div class="wrap wide">
      <div class="ptitle">
        <div><div class="eyebrow">06 · No AI, just the rules</div><h1 class="display">Calculate</h1></div>
        <p>Deterministic calculators: the same inputs always give the same answer, with the rule
           that produced it shown underneath.</p>
      </div>
      <div class="split">
        <ul class="jlist" id="calc-list" style="max-height:none;margin-top:0"></ul>
        <div id="calc"></div>
      </div>
    </div>
  </section>

  <!-- DEVELOPERS -->
  <section data-page="developers">
    <div class="wrap wide">
      <div class="ptitle">
        <div><div class="eyebrow">For developers</div><h1 class="display">API</h1></div>
        <p>Everything in the app is available over plain HTTP and JSON: the same endpoints the
           pages use. Send mode (citizen / lawyer) and language (en / ur / roman_ur) with any request.</p>
      </div>
      <table class="lib apitable">
        <thead><tr><th>Method</th><th>Path</th><th>What it does</th><th>Body</th></tr></thead>
        <tbody>
          <tr><td>POST</td><td><code>/api/chat</code></td><td>Ask a question; the router picks the research, summary or compare agent</td><td><code>message, thread_id?</code></td></tr>
          <tr><td>POST</td><td><code>/api/chat/stream</code></td><td>The same, streamed as JSON lines: one <code>step</code> event per agent step, then a <code>result</code></td><td><code>message, thread_id?</code></td></tr>
          <tr><td>POST</td><td><code>/api/summarize</code></td><td>Brief a judgment from the library, or an uploaded one</td><td><code>doc_id</code> or <code>data, mime, filename</code> or <code>text</code></td></tr>
          <tr><td>POST</td><td><code>/api/compare</code></td><td>Compare two judgments side by side</td><td><code>doc_a, doc_b</code></td></tr>
          <tr><td>POST</td><td><code>/api/explain</code></td><td>Explain a user's document with red flags, laws and similar judgments; never stored</td><td><code>data, mime, filename</code> or <code>text</code></td></tr>
          <tr><td>POST</td><td><code>/api/explain/ask</code></td><td>A follow-up question about that document</td><td><code>text, question</code></td></tr>
          <tr><td>POST</td><td><code>/api/draft</code></td><td>Draft a document from facts, with supporting judgments</td><td><code>kind, facts{}</code></td></tr>
          <tr><td>GET</td><td><code>/api/judgments</code></td><td>Every indexed judgment</td><td>—</td></tr>
          <tr><td>GET</td><td><code>/api/judgments/&lt;doc_id&gt;</code></td><td>One judgment with all its passages</td><td>—</td></tr>
          <tr><td>GET</td><td><code>/api/citations</code></td><td>Every case citation in the library, with the judgment and paragraph citing it</td><td>—</td></tr>
          <tr><td>GET</td><td><code>/api/statutes</code></td><td>The Constitution, PPC, CrPC, CPC and family laws, word for word, with sources</td><td>—</td></tr>
          <tr><td>GET</td><td><code>/api/stats</code></td><td>Corpus counts and the model in use</td><td>—</td></tr>
        </tbody>
      </table>
      <p class="privacy">Every answer carries <code>sources</code>: the passages it was written from, each with case,
         court, paragraph, provisions and its search scores. <code>[S1]</code> in the answer text refers to <code>sources[0]</code>.</p>

      <h3 class="apih">Ask a question</h3>
<pre class="code">curl -s http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Can a mother get interim custody of a young child while the case is pending?",
       "mode": "citizen", "language": "roman_ur"}'</pre>

      <h3 class="apih">Stream the agent's steps (Python)</h3>
<pre class="code">import json, requests

r = requests.post("http://127.0.0.1:8000/api/chat/stream", stream=True,
                  json={"message": "Find bail cases under Section 497 CrPC", "mode": "lawyer"})
for line in r.iter_lines():
    event = json.loads(line)
    if event["type"] == "step":
        print("step:", event["node"])          # classify, retrieve, answer ...
    else:
        print(event["answer"])
        for s in event["sources"]:
            print(s["n"], s["case_no"], "para", s["paragraph"])</pre>

      <h3 class="apih">Explain a document and draft a reply</h3>
<pre class="code">import base64, requests

pdf = base64.b64encode(open("notice.pdf", "rb").read()).decode()
explained = requests.post("http://127.0.0.1:8000/api/explain",
    json={"filename": "notice.pdf", "mime": "application/pdf", "data": pdf}).json()
print(explained["answer"])                       # includes "Red flags and deadlines"

draft = requests.post("http://127.0.0.1:8000/api/draft", json={
    "kind": "reply_to_notice",
    "facts": {"Your name": "Kashif Javed", "What the notice demands": "Rs. 135,000 and vacating by 31.10.2026"}}).json()
print(draft["answer"])</pre>
      <p class="privacy">Draft kinds: <code>reply_to_notice</code>, <code>bail_application</code>, <code>cheque_notice</code>,
         <code>fir_application</code>, <code>superdari</code>. The server runs locally; there are no keys or accounts.</p>
    </div>
  </section>

  <!-- MATTERS -->
  <div class="voice" id="voice" hidden role="dialog" aria-label="Voice mode">
    <button type="button" class="voice-x" id="voice-x">Close</button>
    <div class="voice-in">
      <div class="eyebrow">Voice mode · speak your problem, hear the answer</div>
      <div class="voice-lang" id="voice-lang"><button type="button" data-l="ur" class="on">اردو</button><button type="button" data-l="en">English</button></div>
      <button type="button" class="voice-mic" id="voice-mic" aria-label="Press and speak"></button>
      <p class="voice-state" id="voice-state"></p>
      <p class="voice-heard" id="voice-heard" dir="auto"></p>
      <form class="voice-type" id="voice-type" hidden><input class="field" id="voice-text" dir="auto" placeholder="Type your question"><button class="btn" type="submit">Ask</button></form>
      <div id="voice-out"></div>
    </div>
  </div>

  <div class="auto" id="auto" hidden role="status" aria-live="polite">
    <div class="auto-bar"><i id="auto-prog"></i></div>
    <div class="auto-in"><span class="auto-n" id="auto-n"></span><p id="auto-cap"></p>
      <div class="auto-keys"><button type="button" id="auto-pause">Pause</button><button type="button" id="auto-next">Next</button><button type="button" id="auto-stop">Stop</button></div></div>
  </div>

  <div class="vid" id="vid" hidden role="dialog" aria-label="Answer video">
    <button type="button" class="voice-x" id="vid-x">Close</button>
    <div class="vid-in">
      <div class="vid-stage"><canvas id="vid-canvas" width="720" height="1280"></canvas><video id="vid-video" controls playsinline hidden></video></div>
      <div class="vid-side">
        <div class="eyebrow">Answer video · for WhatsApp</div>
        <h3 id="vid-title">Making your video</h3>
        <ol class="vid-steps" id="vid-steps"></ol>
        <div class="vid-acts" id="vid-acts" hidden></div>
        <p class="vid-note">The script is taken from the answer, word for word; the voice is OpenAI’s speech model. General information, not legal advice.</p>
      </div>
    </div>
  </div>

  <section data-page="workspace">
    <div class="ws">
      <aside class="ws-side">
        <div class="ws-who" id="ws-who" role="radiogroup" aria-label="Who is using the workspace">
          <button type="button" data-p="lawyer">I'm a lawyer</button><button type="button" data-p="people">I have a legal problem</button></div>
        <label class="ws-l" for="ws-matter" id="ws-matter-l">Matter</label>
        <select id="ws-matter" class="ws-sel"></select>
        <button type="button" class="ws-link" id="ws-new">+ New matter</button>
        <div class="ws-nav" id="ws-nav">
          <button type="button" data-v="assistant">Assistant</button>
          <button type="button" data-v="vault">Vault <span id="ws-vn"></span></button>
          <button type="button" data-v="workflows">Workflows <span id="ws-wn"></span></button>
          <button type="button" data-v="history">History <span id="ws-hn"></span></button>
        </div>
        <label class="ws-l" for="ws-facts" id="ws-facts-l">Matter facts</label>
        <textarea id="ws-facts" class="ws-facts" rows="9" dir="auto"></textarea>
        <p class="ws-priv">Kept in this browser only. Your files and facts are never stored on the server.</p>
        <div class="ws-help" id="ws-help"><span class="ws-l">Talk to a person, free</span><a href="tel:1818"><b>1818</b> Supreme Court legal call centre</a>
          <a href="tel:080070806"><b>0800-70806</b> Legal Aid Society, 24 hours</a><a href="tel:1099"><b>1099</b> Ministry of Human Rights</a></div>
      </aside>
      <div class="ws-main">
        <div class="ws-view" data-v="assistant">
          <div class="eyebrow" id="ws-kicker">Assistant</div>
          <h1 class="display ws-h" id="ws-hello"></h1>
          <form class="ws-ask" id="ws-ask">
            <textarea id="ws-q" rows="3" dir="auto"></textarea>
            <div class="ws-bar"><div class="ws-src" id="ws-src"><button type="button" data-s="law" class="on">Case law and statutes</button><button type="button" data-s="vault">This matter's documents</button></div>
              <span class="grow"></span><button class="mic" id="ws-mic" type="button" hidden></button><button class="btn" type="submit">Ask</button></div>
          </form>
          <div class="ws-sugg" id="ws-sugg"></div>
          <div id="ws-thread"></div>
        </div>
        <div class="ws-view" data-v="vault" hidden>
          <div class="eyebrow">Vault</div>
          <h1 class="display ws-h">The papers in this <span class="ws-word">matter</span></h1>
          <label class="ws-drop" id="ws-drop"><input type="file" id="ws-files" multiple accept=".pdf,.txt,.md,image/*" hidden>
            <b>Add files</b><span>Drop PDFs, photos or text here: FIRs, notices, orders, agreements. Read in memory, kept in this browser.</span></label>
          <div class="ws-docs" id="ws-docs"></div>
          <div class="ws-acts" id="ws-acts"></div>
          <div id="ws-vout"></div>
        </div>
        <div class="ws-view" data-v="workflows" hidden>
          <div class="eyebrow">Workflows</div>
          <h1 class="display ws-h" id="ws-wh"></h1>
          <div class="ws-flows" id="ws-flows"></div>
          <div id="ws-form"></div>
          <div id="ws-wout"></div>
        </div>
        <div class="ws-view" data-v="history" hidden>
          <div class="eyebrow">History</div>
          <h1 class="display ws-h">Everything done in this <span class="ws-word">matter</span></h1>
          <div id="ws-hist"></div>
        </div>
      </div>
    </div>
  </section>

  <section data-page="court">
    <div class="wrap wide">
      <div class="ptitle">
        <div><div class="eyebrow">Moot court · rehearse before you appear</div><h1 class="display">Courtroom</h1></div>
        <p>A Lahore High Court judge questions you one point at a time, citing the judgments, and reacts to each answer.
           Answer by typing or speaking, in English or Urdu. After five questions you get a score and feedback on every answer.</p>
      </div>
      <div class="ws-formbox court-setup">
        <div class="court-grid">
          <div><label class="eyebrow" for="court-hearing">The hearing</label><input class="field" id="court-hearing" value="a post-arrest bail application"></div>
          <div><label class="eyebrow" for="court-side">Your side</label><input class="field" id="court-side" value="accused (petitioner)"></div>
        </div>
        <label class="eyebrow" for="court-facts">The facts</label>
        <textarea class="field" id="court-facts" rows="5" dir="auto" placeholder="Who you act for, what happened and when, what the FIR or the other side says"></textarea>
        <label class="mt-speak"><input type="checkbox" id="court-voice" checked> The judge speaks aloud (best in Microsoft Edge)</label>
        <div class="ctas"><button class="btn" type="button" id="court-go">Enter the courtroom</button>
          <button class="btn line" type="button" id="court-sample">Try a sample: bail in a bounced-cheque case</button></div>
      </div>
      <div id="court-out"></div>
    </div>
  </section>

  <section data-page="verify">
    <div class="wrap wide">
      <div class="ptitle">
        <div><div class="eyebrow">Before you file</div><h1 class="display">Verify</h1></div>
        <p>Check a draft before it reaches the court: every section and case it cites, every fact against your papers,
           and whether a case is still followed. The law check uses no AI.</p>
      </div>
      <div class="libtabs" id="vf-tabs"><button type="button" class="on" data-tab="law">Check the law</button>
        <button type="button" data-tab="facts">Check the facts</button><button type="button" data-tab="case">Is a case still followed?</button>
        <button type="button" data-tab="video">Check a video</button></div>
      <div class="vf-pane" data-pane="law">
        <div class="ws-formbox">
          <p class="ws-empty">Every section is looked up in the official text, every reported case in the library, every year against today.
             An invented or repealed section and an impossible citation are flagged; a case we do not hold is marked to check.</p>
          <label class="eyebrow" for="vf-text">The draft</label>
          <textarea class="field" id="vf-text" rows="9" dir="auto" placeholder="Paste a petition, a notice, an application or any draft"></textarea>
          <div class="ctas"><button class="btn" type="button" id="vf-go">Check the law</button>
            <button class="btn line" type="button" id="vf-sample">Try a sample petition</button></div>
        </div>
        <div id="vf-out"></div>
      </div>
      <div class="vf-pane" data-pane="facts" hidden>
        <div class="ws-formbox">
          <p class="ws-empty">Each fact in the draft is marked supported, contradicted or not in the papers, with the words that show it.
             Every quote is checked here, in your browser, against the paper itself. Papers are read for this check only, never stored.</p>
          <label class="eyebrow" for="vf-draft">The draft</label>
          <textarea class="field" id="vf-draft" rows="7" dir="auto" placeholder="Paste the petition, plaint, reply or statement"></textarea>
          <div class="eyebrow">The papers</div>
          <div class="vf-papers" id="vf-papers"></div>
          <div class="ctas"><label class="btn line vf-add">Add papers (PDF, photo or text)<input type="file" id="vf-files" multiple accept=".pdf,.txt,.md,image/*" hidden></label>
            <button class="btn line" type="button" id="vf-fir">Use the sample FIR</button>
            <button class="btn" type="button" id="vf-check">Check the facts</button></div>
        </div>
        <div id="vf-fout"></div>
      </div>
      <div class="vf-pane" data-pane="video" hidden>
        <div class="ws-formbox">
          <p class="ws-empty">A phone video of a police stop, a seizure or an incident. The video stays on this device: the browser takes
             still frames from it and only those are read, then a timeline of what they show is checked against the judgments.
             The sound is not used, and nothing is kept.</p>
          <div class="ctas"><label class="btn line vf-add">Choose a video<input type="file" id="vv-file" accept="video/*" hidden></label>
            <span class="vv-name" id="vv-name"></span></div>
          <video id="vv-video" muted playsinline preload="auto" hidden></video>
          <div class="vv-strip" id="vv-strip"></div>
          <label class="eyebrow" for="vv-context">What it is, in your words (optional)</label>
          <textarea class="field" id="vv-context" rows="2" dir="auto" placeholder="e.g. Police took my motorbike outside my shop in Gulberg on 2 October"></textarea>
          <div class="ctas"><button class="btn" type="button" id="vv-go" disabled>Read the video</button></div>
        </div>
        <div id="vv-out"></div>
      </div>
      <div class="vf-pane" data-pane="case" hidden>
        <div class="ws-formbox">
          <p class="ws-empty">The citator agent reads each judgment in the library that cites the case and labels it: followed, distinguished,
             doubted or only mentioned, with the words that show it. It knows only how this library treats the case.</p>
          <label class="eyebrow" for="vf-cite">A reported case</label>
          <div class="vf-citerow"><input class="field" id="vf-cite" placeholder="e.g. 2013 SCMR 1205 or PLD 2016 SC 808"><button class="btn" type="button" id="vf-cite-go">Run the citator</button></div>
          <div class="vf-chips" id="vf-chips"></div>
        </div>
        <div id="vf-cout"></div>
      </div>
    </div>
  </section>

  <section data-page="matters">
    <div class="wrap wide">
      <div class="ptitle">
        <div><div class="eyebrow">Your workspace</div><h1 class="display">Matters</h1></div>
        <p>Keep the answers, tables and drafts for one case together, then export them as one PDF.
           Saved in this browser only; nothing is sent to the server.</p>
      </div>
      <div class="split">
        <div>
          <div class="eyebrow">Matters</div>
          <ul class="jlist" id="m-list" style="max-height:none"></ul>
          <form class="fform" id="m-new"><input class="field" id="m-name" placeholder="New matter, e.g. Aslam v. Tariq (cheque)" maxlength="80">
            <button class="btn" type="submit">Create</button></form>
          <p class="privacy">Press <b>Save to matter</b> under any answer, table or draft to add it here.</p>
        </div>
        <div id="m-view"></div>
      </div>
    </div>
  </section>

  <!-- AGENT ROOM (route /case) -->
  <section data-page="case">
    <div class="wrap wide">
      <div class="ptitle">
        <div><div class="eyebrow">Orchestrator and agents</div><h1 class="display">Agent room</h1></div>
        <p>Ask anything, describe your situation or add a document. An orchestrator reads it, picks a team of agents for
           the job and runs them. Watch each one work and what it hands to the next.</p>
      </div>
      <div class="case-in">
        <label class="eyebrow" for="case-text">Your request, in English, اردو or Roman Urdu</label>
        <textarea class="field" id="case-text" rows="3" dir="auto"
          placeholder="A question, your situation, or a judgment to read or compare"></textarea>
        <div class="microw">
          <button class="mic" id="case-mic" type="button" hidden></button>
          <label class="btn line" for="case-file" style="cursor:pointer;height:34px">Add a document</label>
          <input type="file" id="case-file" accept=".pdf,image/png,image/jpeg,image/webp" hidden>
          <span class="micnote" id="case-fname"></span>
        </div>
        <div class="case-try" id="case-try"></div>
        <button class="btn" id="case-go" style="margin-top:16px">Start the agents</button>
        <p class="privacy">A document you add is read for this request only and never stored.</p>
      </div>
      <div class="stage" id="room-stage" hidden>
        <div class="route" id="room-route"></div>
        <div class="room-plan" id="room-plan" hidden></div>
        <div class="netbox"><canvas class="netcanvas" id="room-net" aria-hidden="true"></canvas><ol class="netcap" id="room-netcap"></ol>
          <span class="netlabel">The law the agents are touching</span></div>
        <div class="rgraph" id="case-team"></div>
        <div class="streams" id="room-streams" aria-live="off"></div>
        <div class="gantt" id="room-gantt" aria-label="When each agent worked"></div>
        <div class="room-key" id="room-key"><span><i class="k-ai"></i>Uses the language model</span>
          <span><i class="k-rules"></i>Fixed rules, no AI</span><span><i class="k-live"></i>Data moving between agents</span>
          <span class="clock" id="room-clock"></span></div>
      </div>
      <div class="case-log" id="case-log" hidden><div class="eyebrow">What the agents are doing</div><ol id="case-feed"></ol></div>
      <div id="case-out"></div>
    </div>
  </section>

  <!-- PITCH -->
  <section data-page="pitch">
    <div class="wrap wide pitch">
      <div class="ptitle">
        <div><div class="eyebrow">For the judges</div><h1 class="display">The pitch</h1></div>
        <p>A legal research assistant for Pakistani case law that answers from real judgments, shows where every
           claim came from, and checks itself. Built for the AIEF 2026 championship.</p>
      </div>

      <h2 class="display">The problem</h2>
      <ul class="plain">
        <li><b>Case law is hard to use.</b> Judgments are long, mostly in English, and written for lawyers. Finding
            the paragraph that decides a question takes real effort, and most citizens cannot read them.</li>
        <li><b>Generic AI is risky here.</b> A chatbot can invent a judgment or a section number that sounds right. In
            law, an invented citation is worse than no answer.</li>
        <li><b>The law is scattered.</b> A judgment cites sections and earlier cases, and checking each one means
            finding it somewhere else.</li>
      </ul>

      <h2 class="display">What it does</h2>
      <div class="tiles">
        <a href="/case" data-link><b>Agent room</b><span>Any request goes to an orchestrator, which deploys a team of six agents for it: research, statutes, precedents, arguments, drafting, and a verifier that checks them.</span></a>
        <a href="/ask" data-link><b>Ask</b><span>Plain-language questions in English, Urdu or Roman Urdu, answered from the judgments, each claim linked to its paragraph.</span></a>
        <a href="/compare" data-link><b>Compare</b><span>Two judgments side by side, or a table of many (including your own documents) with every cell cited.</span></a>
        <a href="/explain" data-link><b>Explain</b><span>Upload an FIR, notice or order: plain words, red flags, deadlines, similar judgments. Nothing is stored.</span></a>
        <a href="/draft" data-link><b>Draft</b><span>Notices and applications, hearing preparation, a dated chronology, a client letter. Gaps stay as placeholders.</span></a>
        <a href="/library" data-link><b>Library</b><span>The judgments, 1,687 statute provisions word for word, a precedent map, and usage insights.</span></a>
        <a href="/calculate" data-link><b>Calculate</b><span>Limitation periods and inheritance shares by fixed rules, with the rule shown.</span></a>
        <a href="/matters" data-link><b>Matters</b><span>Keep a case's answers, tables and drafts together and export them as one PDF.</span></a>
        <a href="/guides" data-link><b>Guides</b><span>Common problems for citizens, step by step, with the laws and judgments behind them.</span></a>
      </div>

      <h2 class="display">Why you can trust it</h2>
      <div class="trust-grid" style="margin-bottom:8px">
        <div><b>99.5%</b><span>of our 50 test questions find the right judgment in the top ten results (MRR 0.97)</span></div>
        <div><b>1 click</b><span>from any claim to the exact paragraph it came from</span></div>
        <div><b>No AI</b><span>in the fact check: every name, section, number and quotation is looked for in the paragraph it cites</span></div>
        <div><b>0 stored</b><span>uploaded documents are read for one answer and never saved or indexed</span></div>
      </div>
      <p class="privacy" style="margin-top:6px">In a 30-question test on the demo model every question was answered and 29 answers cited only real
         passages. On a cheaper model the fact check caught a child's name the model made up and a date it worked out itself, neither in any judgment.
         The method and every number are in EVALUATION.md in the project repository.</p>

      <h2 class="display">Built to Pakistan's judicial AI guidelines</h2>
      <p class="privacy" style="margin-top:0">In 2026 the National Judicial (Policy Making) Committee issued guidelines for AI in Pakistan's courts:
         AI may support research, case management and drafting, and people stay in charge of every decision. The Supreme Court had
         already called AI an auxiliary tool (Ishfaq Ahmed v. Mushtaq Ahmed, 2025). Wakeel is built that way.</p>
      <div class="guide-map">
        <div><b>People decide</b><span>Wakeel never decides a case. Answers are research a lawyer can check claim by claim; drafts open for editing; every step of the agents is on screen.</span></div>
        <div><b>Transparency</b><span>Each agent is shown as it works, marked AI or No AI, with its words as it writes them. Every citation opens the paragraph it came from.</span></div>
        <div><b>Verification</b><span>A fact check and a brief checker with no AI; statute text word for word from the official texts; a wrong section or an impossible citation is flagged.</span></div>
        <div><b>Data protection</b><span>Uploaded papers are read for one request and never stored. The workspace keeps files in the user's own browser; the usage log counts requests, never content.</span></div>
        <div><b>Honest limits</b><span>When the judgments do not answer a question it says so instead of guessing. It sends people to free human help: the Supreme Court's 1818 line and legal aid.</span></div>
      </div>

      <h2 class="display">How it works</h2>
      <ol class="steps">
        <li><b>Read the judgments the way a lawyer does.</b> They are split on numbered paragraphs, not at fixed lengths, and every paragraph keeps its case, court and citations.</li>
        <li><b>Search by meaning and by keyword together.</b> Dense embeddings and BM25 are combined, then the neighbouring and closing paragraphs are added, because the outcome is usually in the last ones.</li>
        <li><b>Route the question.</b> An agent decides whether it is a question, a summary or a comparison, and shows its plan before it acts.</li>
        <li><b>Write only from the passages.</b> If they do not answer it, the app says so instead of guessing, and a second model gets one try before it does.</li>
        <li><b>Check the result.</b> Citations must point at real passages, and the fact check looks for each detail in the passage it cites.</li>
      </ol>

      <h2 class="display">What we measured, and what we did not</h2>
      <ul class="plain">
        <li><b>Measured:</b> search quality on 50 questions (<span data-stat="judgments">–</span> judgments, <span data-stat="passages">–</span> indexed passages); answer quality on 30 of them; a bug that dropped a quarter of every judgment, which we found by measuring and fixed.</li>
        <li><b>Not yet:</b> the 50 questions are our own and have not been reviewed by a lawyer; the library is Lahore High Court only; Urdu answers and the legal templates still need a lawyer's review; handwritten Urdu scans are untested.</li>
        <li><b>Not legal advice.</b> It is a research aid. Every answer links to its sources so a lawyer can verify it.</li>
      </ul>

      <h2 class="display">Team</h2>
      <p class="team"><b>Agha Ali Hassan</b> and <b>Warda Fatima</b>, FAST-NUCES Lahore, BS Computer Science.</p>
      <div class="ctas" style="margin:22px 0 10px">
        <button type="button" class="btn" id="pitch-auto">Play the demo (autopilot)</button>
        <button type="button" class="btn line" id="pitch-tour">Take the 90-second tour</button>
        <a href="/ask" data-link class="btn line">Ask a question</a>
        <a href="/developers" data-link class="btn line">For developers</a>
      </div>
    </div>
  </section>

  <!-- LIBRARY -->
  <section data-page="library">
    <div class="wrap wide">
      <div class="ptitle">
        <div><div class="eyebrow">Corpus</div><h1 class="display">Library</h1></div>
        <p id="lib-sum">Every judgment Wakeel AI can answer from. Select one to read its passages.</p>
      </div>
      <div class="libtools">
        <div class="libtabs" id="lib-tabs"><button type="button" class="on" data-tab="judgments">Judgments</button>
          <button type="button" data-tab="statutes">Statutes</button>
          <button type="button" data-tab="insights">Insights</button>
          <button type="button" data-tab="map">Precedent map</button>
          <button type="button" data-tab="activity">Activity</button></div>
        <input class="field" id="lib-search" placeholder="Filter by party, case number, court or provision">
      </div>
      <table class="lib" id="lib-judgments">
        <thead><tr><th>Case</th><th>Parties</th><th class="hide-sm">Court</th><th class="hide-sm">Key provisions</th><th>Passages</th></tr></thead>
        <tbody id="lib-body"></tbody>
      </table>
      <table class="lib" id="lib-statutes" hidden>
        <thead><tr><th>Law</th><th>Provision</th><th>Title</th></tr></thead>
        <tbody id="stat-body"></tbody>
      </table>
      <div class="insights" id="lib-insights" hidden></div>
      <div class="pmap" id="lib-map" hidden></div>
      <div class="insights" id="lib-activity" hidden></div>
    </div>
  </section>
</main>

<div class="composer">
  <div class="scope" id="ask-scope" hidden></div>
  <form id="form">
    <textarea id="input" rows="1" placeholder="Ask about Pakistani case law, in English, اردو or Roman Urdu"></textarea>
    <button class="mic" id="ask-mic" type="button" hidden></button>
    <button class="btn" id="send" type="submit">Ask</button>
  </form>
</div>

<div class="tour" id="tour" role="dialog" aria-label="Guided tour" hidden>
  <div class="eyebrow" id="tour-n"></div>
  <h3 id="tour-t"></h3>
  <p id="tour-p"></p>
  <div class="tour-b">
    <button type="button" class="btn line" id="tour-back">Back</button>
    <button type="button" class="btn" id="tour-next">Next</button>
    <button type="button" class="tour-x" id="tour-x">End tour</button>
  </div>
</div>
<div class="scrim" id="scrim"></div>
<aside id="panel" aria-label="Detail"></aside>
<div id="print-area"></div>

<script>
// ===========================================================================
// state and helpers
// ===========================================================================
const EXAMPLES = [
  ['Research', 'Can a mother get interim custody of a young child while the case is still pending?'],
  ['Summary','Summarize the judgment on interim custody of a minor under the Guardians and Wards Act'],
  ['Compare', 'Compare the two murder references where the death sentence was not confirmed'],
  ['Urdu', 'اگر پولیس نے میری گاڑی قبضے میں لے لی ہو تو میں اسے واپس کیسے حاصل کر سکتا ہوں؟', 'ur'],
  ['Roman Urdu', 'Kya maa ko chotay bachay ki interim custody mil sakti hai jab case chal raha ho?', 'roman_ur'],
];
const AGENT = {GQ: 'Research agent', SQ: 'Summary agent', CQ: 'Compare agent', REJECT: 'Router', ERROR: 'Router'};
const MODE_LABEL = {citizen: 'Citizen', lawyer: 'Lawyer'};
const LANG_LABEL = {en: 'English', ur: 'اردو', roman_ur: 'Roman Urdu'};
const STOP = new Set(('the a an and or of to in on for with by is are was were be can could would should under ' +
  'from that this these those what which when where who how does do did has have had any find show case cases ' +
  'judgment judgments court law about into their there than then also between both summarize compare').split(' '));

let thread = newThreadId();
let busy = false;
let stats = null;
let judgments = null;          // [{doc_id, title, ...}] from /api/judgments
const results = {};            // result key -> {sources, query}
let seq = 0;

function newThreadId() { return crypto.randomUUID ? crypto.randomUUID() : String(Math.random()); }
function $(id) { return document.getElementById(id); }
function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}
// While the model works the page says what it is doing, in the stage it has reached, instead of a stopwatch.
const THINKING = {
  start: ['Understanding your question', 'Working out what you are asking'],
  searching: ['Looking into the judgments', 'Searching for the closest passages', 'Going through them paragraph by paragraph'],
  writing: ['Reading the passages', 'Weighing what the court held', 'Writing a cited answer'],
  '/api/summarize': ['Reading the judgment', 'Going through it section by section', 'Tying each point to its paragraph', 'Writing the summary'],
  '/api/compare': ['Reading both judgments', 'Setting the points side by side', 'Checking where they agree and differ', 'Writing the comparison'],
  '/api/explain': ['Reading your document', 'Looking for the laws it relies on', 'Checking for red flags and deadlines', 'Finding similar judgments', 'Writing the explanation'],
  '/api/draft': ['Looking into similar judgments', 'Choosing the wording', 'Drafting from your facts', 'Leaving placeholders where facts are missing'],
  '/api/workflow': ['Looking into similar judgments', 'Thinking through the facts', 'Weighing both sides', 'Writing it up'],
  '/api/strategy': ['Looking into similar judgments', 'Thinking through the facts', 'Weighing both sides', 'Writing it up'],
  '/api/table': ['Reading each document', 'Filling in each column', 'Checking each cell against its paragraph'],
};
// The pipeline view: every agent a request passes through, drawn as it reports, with its words as it writes them.
// Agents that start while another is still working (the review table's reviewers) stand in one column.
function pipeView(before) {
  const box = el('div', 'pipe');
  box.innerHTML = '<div class="pl"><span>Behind this answer</span><span class="pc"></span><button type="button" hidden>Show what each agent wrote</button><span class="pt"></span></div>' +
                  '<div class="prow"></div><div class="streams"></div>';
  before.parentNode.insertBefore(box, before);
  const row = box.querySelector('.prow'), wall = box.querySelector('.streams'), t0 = performance.now();
  const P = {box, nodes: {}, cols: [], live: {}, start: {}, end: {}, done: false};
  const toggle = box.querySelector('.pl button');
  toggle.addEventListener('click', () => { wall.hidden = !wall.hidden; toggle.textContent = wall.hidden ? 'Show what each agent wrote' : 'Hide what each agent wrote'; });
  function family(id) { return id.split('-')[0]; }
  function node(ev) {
    if (P.nodes[ev.id]) return P.nodes[ev.id];
    const last = P.cols[P.cols.length - 1];
    const join = last && last.ids.some(i => P.nodes[i].dataset.status === 'working' && family(i) === family(ev.id) && i !== ev.id ||
                                       family(i) === family(ev.id) && family(ev.id) !== ev.id);
    const n = el('div', 'pnode');
    n.innerHTML = '<div class="ph"><i></i><b></b><em class="' + (ev.ai ? 'ai">AI' : 'rules">No AI') + '</em></div><p class="pn"></p><p class="plive" hidden></p><ul class="pm"></ul><div class="ptime"></div>';
    n.querySelector('b').textContent = ev.name || ev.id;
    n.querySelector('b').title = ev.name || ev.id;
    if (join) { last.el.appendChild(n); last.ids.push(ev.id); }
    else {
      if (last) { const c = el('div', 'pconn'); row.appendChild(c); last.conn = c; }
      const col = el('div', 'pcol'); col.appendChild(n); row.appendChild(col);
      P.cols.push({el: col, ids: [ev.id], conn: null});
    }
    P.nodes[ev.id] = n;
    box.querySelector('.pc').textContent = plural(Object.keys(P.nodes).length, 'agent');
    return n;
  }
  function conns() {
    P.cols.forEach((c, i) => {
      const next = P.cols[i + 1];
      if (!c.conn || !next) return;
      const working = next.ids.some(id => P.nodes[id].dataset.status === 'working');
      c.conn.className = 'pconn ' + (working ? 'flow' : 'used');
    });
  }
  function card(id) {
    let c = wall.querySelector('[data-card="' + id + '"]');
    if (!c) {
      c = el('div', 'scard');
      c.dataset.card = id;
      c.innerHTML = '<div class="sh"><i></i><b></b><span></span></div><div class="sbody"></div>';
      c.querySelector('b').textContent = P.nodes[id] ? P.nodes[id].querySelector('b').textContent : id;
      wall.appendChild(c);
    }
    return c;
  }
  P.event = ev => {
    if (ev.type === 'stage') {
      const n = node(ev), now = performance.now() - t0;
      if (ev.status === 'working' && P.start[ev.id] === undefined) P.start[ev.id] = now;
      if (ev.status !== 'working') { if (P.start[ev.id] === undefined) P.start[ev.id] = now; P.end[ev.id] = now; }
      if (ev.status === 'working') delete P.end[ev.id];
      n.dataset.status = ev.status;
      n.querySelector('.pn').textContent = ev.note || '';
      const d = ev.data || {};
      if (d.provisions && d.provisions.length) P.mini(ev.id, d.provisions);
      if (d.doc_ids && d.doc_ids.length && ev.id === 'search')
        loadJudgments().then(() => P.mini(ev.id, d.doc_ids.map(x => { const j = (judgments || []).find(y => y.doc_id === x); return j ? caseTitle(j) : x; })));
      const c = wall.querySelector('[data-card="' + ev.id + '"]');
      if (c) c.dataset.status = ev.status;
      if (c && ev.status !== 'working') c.querySelector('.sbody').classList.remove('live');
      if (ev.status !== 'working') n.querySelector('.plive').hidden = true;
      conns();
    } else if (ev.type === 'chunk') {
      P.live[ev.id] = (ev.new ? '' : P.live[ev.id] || '') + ev.text;
      const n = P.nodes[ev.id] || node({id: ev.id, name: 'Writer agent', ai: true}), live = n.querySelector('.plive');
      live.hidden = false; live.textContent = roomTail(P.live[ev.id], 90);
      const c = card(ev.id), body = c.querySelector('.sbody'), atEnd = body.scrollHeight - body.scrollTop - body.clientHeight < 30;
      c.dataset.status = n.dataset.status || 'working';
      body.textContent = plainWords(P.live[ev.id]);
      body.className = 'sbody live';
      if (atEnd) body.scrollTop = body.scrollHeight;
      c.querySelector('.sh span').textContent = wordCount(P.live[ev.id]) + ' words';
    }
  };
  P.agents = () => Object.values(P.nodes).map(n => ({name: n.querySelector('b').textContent, ai: !!n.querySelector('em.ai')}));
  P.mini = (id, items) => {
    const ul = P.nodes[id] && P.nodes[id].querySelector('.pm');
    if (!ul) return;
    ul.innerHTML = '';
    items.slice(0, 3).forEach(t => { const li = el('li'); li.textContent = t; ul.appendChild(li); });
  };
  // At the end: the fact checker (no AI) runs over the answer in the browser, and the words fold away under a toggle.
  P.finish = (data, userText) => {
    if (P.done) return;
    if (data && data.route !== 'ERROR' && data.route !== 'TABLE' && data.answer && (data.sources || []).length && Object.keys(P.nodes).length) {
      const f = checkFacts(data.answer, data.sources, userText || '');
      P.event({type: 'stage', id: 'verify', name: 'Fact checker', ai: false, status: 'done',
               note: 'Checked ' + plural(f.checked, 'fact') + ': ' + (f.checked - f.missing.length) + ' found in the passage cited'});
    }
    P.done = true;
    tick();
    if (wall.children.length) { wall.hidden = true; toggle.hidden = false; }
    if (!Object.keys(P.nodes).length) box.remove();
  };
  function tick() {
    const now = performance.now() - t0;
    Object.entries(P.nodes).forEach(([id, n]) => {
      if (P.start[id] === undefined) return;
      const e = P.end[id] !== undefined ? P.end[id] : now, w = wordCount(P.live[id]);
      n.querySelector('.ptime').textContent = (P.end[id] === undefined ? 'working ' : '') + caseClock(e - P.start[id]) + (w ? ' \u00b7 ' + w + ' words' : '');
    });
    box.querySelector('.pt').textContent = caseClock(now);
    if (!P.done) setTimeout(tick, 100);
  }
  tick();
  return P;
}
// POST to a page's endpoint, asking to watch: what each agent does goes to the view, and the answer is returned.
async function watchedPost(url, payload, before, userText) {
  const view = pipeView(before);
  let data = null;
  try {
    const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({...payload, watch: true})});
    if (!(r.headers.get('Content-Type') || '').includes('ndjson')) data = await r.json();
    else {
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const {value, done} = await reader.read();
        if (done) break;
        buf += dec.decode(value, {stream: true});
        let nl;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          const ev = JSON.parse(line);
          if (ev.type === 'result') data = ev; else view.event(ev);
        }
      }
    }
  } catch (e) { data = {route: 'ERROR', answer: 'Could not reach the server: ' + e.message, sources: []}; }
  data = data || {route: 'ERROR', answer: 'The answer did not arrive.', sources: []};
  view.finish(data, userText);
  data._agents = view.agents();
  return data;
}
function startThinking(node, phrases) {
  let i = 0;
  node.classList.add('thinking');
  node.textContent = phrases[0];
  return setInterval(() => { i = (i + 1) % phrases.length; node.textContent = phrases[i]; }, 2600);
}
function stopThinking(node, id) { clearInterval(id); node.classList.remove('thinking'); }
function plural(n, w) { return n + ' ' + w + (n === 1 ? '' : 's'); }
function pad2(n) { return String(n).padStart(2, '0'); }
function caseTitle(s) {
  const p = s.parties || {};
  if (p.appellant && p.respondent) return p.appellant + ' v. ' + p.respondent;
  return s.title || caseNumber(s) || 'Judgment';
}
function hasParties(s) { const p = s.parties || {}; return !!((p.appellant && p.respondent) || s.title); }
function caseNumber(s) { return [s.judgment_type, s.case_no].filter(Boolean).join(' '); }

// ===========================================================================
// pages
// ===========================================================================
const PAGES = ['home', 'ask', 'summarize', 'compare', 'explain', 'draft', 'guides', 'calculate', 'library', 'matters', 'case', 'pitch', 'developers', 'workspace', 'court', 'verify'];
function pageFromPath(p) { const n = p.replace(/^\/+|\/+$/g, ''); return PAGES.includes(n) ? n : 'home'; }
function showPage(name) {
  document.querySelectorAll('main > section').forEach(s => s.classList.toggle('on', s.dataset.page === name));
  document.querySelectorAll('nav a').forEach(a => a.classList.toggle('on', a.dataset.nav === name));
  const onLink = document.querySelector('nav a.on');
  if (onLink && onLink.parentElement.scrollWidth > onLink.parentElement.clientWidth) onLink.scrollIntoView({block: 'nearest', inline: 'center'});
  document.body.className = document.body.className.replace(/\bp-\w+/g, '').trim() + ' p-' + name;
  closePanel();
  if ('speechSynthesis' in window) speechSynthesis.cancel();   // stop reading when the page changes
  window.scrollTo(0, 0);
  if (name === 'home') heroIntro();
  if (name === 'ask') { renderThreadIntro(); $('input').focus(); }
  if (name === 'summarize' || name === 'compare' || name === 'library') loadJudgments();
  if (name === 'draft' && !draftKind) renderPicker($('draft-kinds'), DRAFTS, null, selectDraft);
  if (name === 'guides' && !guideId) selectGuide(GUIDES[0].id);
  if (name === 'calculate' && !calcId) selectCalc(CALCS[0].id);
  if (name === 'matters') renderMatters(new URLSearchParams(location.search).get('m'));
  if (name === 'workspace') wsOpen();
  if (name === 'court') courtOpen();
  if (name === 'verify') verifyOpen();
  const more = $('navmore');
  more.querySelector('button').classList.toggle('on', !!more.querySelector('[data-more="' + name + '"]'));
  more.querySelector('.navmenu').hidden = true;
}
function go(path) {
  if (location.pathname + location.search !== path) history.pushState({}, '', path);
  showPage(pageFromPath(new URL(path, location.origin).pathname));
}
document.addEventListener('click', e => {
  const a = e.target.closest('a[data-link]');
  if (!a || e.metaKey || e.ctrlKey) return;
  e.preventDefault();
  go(a.getAttribute('href'));
});
window.addEventListener('popstate', () => showPage(pageFromPath(location.pathname)));

// ===========================================================================
// mode and language (remembered per browser)
// ===========================================================================
function segValue(id) { return document.querySelector('#' + id + ' button.on').dataset.value; }
function setSeg(id, value) {
  document.querySelectorAll('#' + id + ' button').forEach(b => b.classList.toggle('on', b.dataset.value === value));
}
['mode', 'language'].forEach(id => {
  try {
    const saved = localStorage.getItem('wakeel.' + id);
    if (saved && document.querySelector('#' + id + ' button[data-value="' + saved + '"]')) setSeg(id, saved);
  } catch (e) {}
  document.querySelectorAll('#' + id + ' button').forEach(b => b.addEventListener('click', () => {
    setSeg(id, b.dataset.value);
    try { localStorage.setItem('wakeel.' + id, b.dataset.value); } catch (e) {}
  }));
});

// ===========================================================================
// voice input: the browser's own speech recognition (Chrome and Edge; needs internet).
// Listens in Urdu when the answer language is Urdu or Roman Urdu, otherwise in English.
// ===========================================================================
const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
const MIC_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">' +
  '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>';
const MIC_ERRORS = {'not-allowed': 'Microphone access is blocked. Allow it in the address bar and try again.',
  'service-not-allowed': 'Microphone access is blocked. Allow it in the address bar and try again.',
  'network': 'Voice input needs an internet connection. Please type instead.',
  'no-speech': 'Nothing was heard. Press the microphone and speak.',
  'audio-capture': 'No microphone was found.'};
function attachMic(btn, field) {
  if (!SpeechRec || !btn || !field) return;
  const compact = btn.closest('.composer');
  const idle = () => { btn.innerHTML = MIC_SVG + (compact ? '' : '<span>Speak</span>'); btn.classList.remove('rec');
                       btn.setAttribute('aria-label', 'Speak instead of typing'); btn.title = 'Speak instead of typing'; };
  const placeholder = field.placeholder;
  const note = (text) => {
    const n = btn.parentElement.querySelector('.micnote');
    if (n) n.textContent = text || '';
    else field.placeholder = text || placeholder;   // the Ask bar has no room for a note
  };
  idle();
  btn.hidden = false;
  let rec = null, failed = false;
  btn.addEventListener('click', () => {
    if (rec) { rec.stop(); return; }
    failed = false;
    rec = new SpeechRec();
    rec.lang = segValue('language') === 'en' ? 'en-IN' : 'ur-PK';
    rec.interimResults = true;
    const before = field.value.trim() ? field.value.replace(/\s*$/, ' ') : '';
    rec.onresult = e => {
      let heard = '';
      for (const r of e.results) heard += r[0].transcript;
      field.value = before + heard;
      field.dispatchEvent(new Event('input'));
    };
    rec.onerror = e => { failed = e.error !== 'aborted'; if (failed) note(MIC_ERRORS[e.error] || 'Voice input stopped. Please type instead.'); };
    rec.onend = () => { rec = null; idle(); if (!failed) note(''); field.focus(); };
    note(rec.lang === 'ur-PK' ? 'Listening in Urdu. Press again to stop.' : 'Listening in English. Press again to stop.');
    btn.classList.add('rec'); btn.setAttribute('aria-label', 'Stop listening');
    btn.innerHTML = MIC_SVG + (compact ? '' : '<span>Stop</span>');
    try { rec.start(); } catch (err) { rec = null; idle(); }
  });
}
attachMic($('ask-mic'), $('input'));
attachMic($('doc-mic'), $('doc-text'));

// ===========================================================================
// corpus stats and judgment list
// ===========================================================================
function paintStats() {
  if (!stats) return;
  document.querySelectorAll('[data-stat]').forEach(b => {
    const v = stats[b.dataset.stat];
    if (v !== undefined) countUp(b, Number(v));
  });
  $('model-short').textContent = stats.provider;
  $('model-short').title = stats.model;
  $('lib-sum').textContent = plural(stats.judgments, 'judgment') + ' · ' + stats.passages.toLocaleString() +
    ' passages · ' + stats.provisions + ' distinct provisions. Select a row to read its passages.';
}
fetch('/api/stats').then(r => r.json()).then(d => { stats = d; paintStats(); }).catch(() => {});

let judgmentsPromise = null;
function loadJudgments() {
  if (!judgmentsPromise) {
    judgmentsPromise = fetch('/api/judgments').then(r => r.json()).then(d => {
      judgments = d.judgments || [];
      renderSumList(''); renderCompareSelects(); renderLibrary('');
      return judgments;
    }).catch(() => { judgmentsPromise = null; return []; });
  }
  return judgmentsPromise;
}
function matches(j, q) {
  if (!q) return true;
  const hay = [caseTitle(j), j.title, caseNumber(j), j.court, (j.provisions || []).join(' '), j.doc_id].join(' ').toLowerCase();
  return q.toLowerCase().split(/\s+/).every(w => hay.includes(w));
}

// ===========================================================================
// Markdown -> HTML (headings, lists, tables, bold, italics, [S1] citations)
// ===========================================================================
function inline(s, key, byN) {
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*\w])\*(?!\s)([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  // [S1], and also [S3, S8] / [S3; S8] / [S3,8], which models write despite being asked not to.
  // A number with no matching source is a model slip; it is dropped rather than shown.
  s = s.replace(/\[(S\d+(?:\s*[,;]\s*S?\d+)*)\]/g, (m, list) =>
    list.split(/\s*[,;]\s*/).map(x => x.replace(/^S/, '')).map(n => byN[n]
      ? '<button type="button" class="cite" data-key="' + key + '" data-n="' + n + '">' + n + '</button>'
      : '').join(''));
  return linkCitations(linkStatutes(s));
}

// ---------------------------------------------------------------------------
// statute linker: "Section 489-F PPC" and "Article 199 of the Constitution" open their text
// ---------------------------------------------------------------------------
let STATUTES = {};
fetch('/api/statutes').then(r => r.json()).then(d => { STATUTES = d.statutes || {}; }).catch(() => {});
const PPC_RE = /\b((?:[Ss]ection|[Ss]ec\.|u\/s)\s*(\d{1,3}(?:-[A-Za-z]{1,2})?)(?:\([0-9a-zA-Z]+\))*\s*(?:of\s+the\s+)?(?:PPC|P\.P\.C\.?|Pakistan Penal Code(?:,?\s*1860)?))/g;
const CONST_RE = /\b([Aa]rticle\s+(\d{1,3}[A-Za-z]{0,2})(?:\([0-9a-zA-Z]+\))*\s+of\s+the\s+Constitution(?:\s+of\s+(?:the\s+Islamic\s+Republic\s+of\s+)?Pakistan(?:,?\s*1973)?)?)/g;
// Sections of the procedure codes and family laws. Their keys drop the hyphen ("crpc:561A"):
// judgments write both "561-A" and "561A". "Cr.P.C." is tried before "C.P.C." so neither takes the other.
const LAW_RE = /\b((?:[Ss]ections?|[Ss]ec\.|u\/s)\s*(\d{1,3}(?:-?[A-Za-z]{1,2})?)(?:\([0-9a-zA-Z]+\))*,?\s*(?:of\s+the\s+)?(Cr\.?\s?P\.?\s?C\b\.?|Code of Criminal Procedure(?:,?\s*1898)?|C\.?\s?P\.?\s?C\b\.?|Code of Civil Procedure(?:,?\s*1908)?|Guardians? (?:and|&) Wards Act(?:,?\s*1890)?|Muslim Family Laws? Ordinance(?:,?\s*1961)?|Dissolution of Muslim Marriages? Act(?:,?\s*1939)?|(?:West Pakistan |Punjab )?Family Courts? Act(?:,?\s*1964)?))/g;
function lawPrefix(name) {
  return /^Cr|Criminal/.test(name) ? 'crpc:' : /^C\.?\s?P|Civil/.test(name) ? 'cpc:' : /Guardian/.test(name) ? 'gwa:'
    : /Muslim Family/.test(name) ? 'mflo:' : /Dissolution/.test(name) ? 'dmma:' : 'fca:';
}
function linkStatutes(s) {
  const button = (m, whole, k) =>
    STATUTES[k] ? '<button type="button" class="statref" data-stat="' + k + '">' + whole + '</button>' : m;
  const wrap = (prefix) => (m, whole, num) => button(m, whole, prefix + num.toUpperCase());
  return s.replace(PPC_RE, wrap('ppc:')).replace(CONST_RE, wrap('const:'))
          .replace(LAW_RE, (m, whole, num, name) => button(m, whole, lawPrefix(name) + num.toUpperCase().replace('-', '')));
}
// Short names for the Library's Law column, in the order the Library lists them.
const LAWS = {'const:': 'Constitution', 'ppc:': 'PPC', 'crpc:': 'CrPC', 'cpc:': 'CPC', 'gwa:': 'Guardians and Wards Act',
              'mflo:': 'Muslim Family Laws Ordinance', 'dmma:': 'Dissolution of Muslim Marriages Act', 'fca:': 'Family Courts Act'};
function statuteLaw(k) { return LAWS[k.slice(0, k.indexOf(':') + 1)] || ''; }
function statuteLabel(k, v) { return (v.kind === 'article' ? 'Article ' : 'Section ') + v.number + (v.kind === 'article' ? '' : ' ' + statuteLaw(k)); }
// ---------------------------------------------------------------------------
// case citations: "2016 SCMR 859" opens the judgments in the library that rely on it. No AI.
// ---------------------------------------------------------------------------
// Two shapes: reporter year court page ("PLD 1989 SC 166", "PLJ 2014 Lahore 193") and year
// reporter page ("2018 SCMR 2098"). "1989 SC 166" is the PLD form without its prefix.
const CITE_RE = /\b(?:(PLD|PLJ)\s+((?:19|20)\d{2})\s+(SC|FSC|Lahore|Lah\.?|Karachi|Kar\.?|Peshawar|Pesh\.?|Quetta|Islamabad)\s+(\d{1,4})|((?:19|20)\d{2})\s+(SCMR|CLC|MLD|YLR|P\.?\s?Cr\.?\s?L\.?\s?J|CLD|PTD|NLR|SCR|PLC|KLR|MLR|GBLR)\s+(\d{1,4})|((?:19|20)\d{2})\s+(SC|FSC|Lahore|Karachi|Peshawar|Quetta|Islamabad)\s+(\d{1,4}))\b/g;
const COURTS = {'Lah': 'Lahore', 'Lah.': 'Lahore', 'Kar': 'Karachi', 'Kar.': 'Karachi', 'Pesh': 'Peshawar', 'Pesh.': 'Peshawar'};
// One key per reported case. PLD and PLJ share the year-court-page key: the corpus writes both
// "PLD 1989 SC 166" and "1989 SC 166" for the same case.
function citeKey(m) {
  if (m[1]) return m[2] + ' ' + (COURTS[m[3]] || m[3]) + ' ' + m[4];
  if (m[5]) return m[5] + ' ' + m[6].replace(/[.\s]/g, '') + ' ' + m[7];
  return m[8] + ' ' + m[9] + ' ' + m[10];
}
let CITES = {};   // key -> {display, refs: {doc_id: [paragraphs]}}
function indexCitations(rows) {
  const out = {};
  rows.forEach(r => {
    for (const m of String(r.citation).matchAll(CITE_RE)) {
      const k = citeKey(m), c = out[k] || (out[k] = {display: m[0], refs: {}});
      if (m[0].length > c.display.length) c.display = m[0];
      const paras = c.refs[r.doc_id] || (c.refs[r.doc_id] = []);
      if (r.paragraph && !paras.includes(r.paragraph)) paras.push(r.paragraph);
    }
  });
  return out;
}
const citesReady = fetch('/api/citations').then(r => r.json()).then(d => { CITES = indexCitations(d.citations || []); }).catch(() => {});
function linkCitations(s) {
  return s.replace(CITE_RE, (...m) => {
    const k = citeKey(m);
    return CITES[k] ? '<button type="button" class="citeref" data-cite="' + escapeHtml(k) + '">' + m[0] + '</button>' : m[0];
  });
}
function paraList(paras) {
  const real = paras.filter(p => p && p !== '0');
  return real.length ? '¶' + real.sort((a, b) => parseInt(a, 10) - parseInt(b, 10)).join(', ¶') : 'Head';
}
async function openCitation(k) {
  const c = CITES[k];
  if (!c) return;
  await loadJudgments();
  const docs = Object.keys(c.refs);
  const name = id => { const j = (judgments || []).find(x => x.doc_id === id); return j ? caseTitle(j) : id; };
  openPanel(
    '<div class="eyebrow">Reported judgment</div>' +
    '<h2 class="big display" style="font-size:56px">' + escapeHtml(c.display) + '</h2>' +
    '<div class="note" style="margin-top:18px">This reported judgment is not itself in the library. ' +
      (docs.length === 1 ? 'One judgment in the library relies on it:' : plural(docs.length, 'judgment') + ' in the library rely on it:') + '</div>' +
    '<ul class="plist">' + docs.map(d =>
      '<li><b>' + escapeHtml(paraList(c.refs[d])) + '</b><span><button type="button" class="jlink" data-doc="' + escapeHtml(d) + '">' +
      escapeHtml(name(d)) + '</button></span></li>').join('') + '</ul>' +
    '<div class="ctas" style="margin-top:18px"><button type="button" class="btn" id="cite-treat">Is it still followed? Run the citator</button></div><div id="cite-out"></div>');
  $('cite-treat').addEventListener('click', () => runCitator(k));
}
const TREATMENT = {followed: 'Followed', distinguished: 'Distinguished', doubted: 'Doubted', mentioned: 'Mentioned'};
async function runCitator(k, out, btn) {
  const c = CITES[k];
  out = out || $('cite-out'); btn = btn || $('cite-treat');
  btn.disabled = true;
  out.innerHTML = '';
  const data = await watchedPost('/api/citator', {citation: c.display, refs: c.refs}, out.appendChild(el('div')));
  btn.disabled = false;
  if (data.route === 'ERROR') { out.insertAdjacentHTML('beforeend', '<p class="error-text"></p>'); out.lastChild.textContent = data.answer; return; }
  const name = id => { const j = (judgments || []).find(x => x.doc_id === id); return j ? caseTitle(j) : id; };
  const n = data.counts || {}, warn = (n.doubted || 0) > 0;
  const box = el('div', 'citator');
  box.innerHTML = '<div class="ct-sum ' + (warn ? 'warn' : 'ok') + '"><b></b><span>The citator agent read each paragraph that cites it. ' +
    'Each label comes with the words that show it; check the paragraph before relying on it. It knows only how this library treats the case.</span></div><ul></ul>';
  box.querySelector('b').textContent = warn ? 'Treated with caution: doubted in ' + plural(n.doubted, 'judgment')
    : (n.followed ? 'Followed in ' + plural(n.followed, 'judgment') : 'Not relied on to decide anything here');
  data.results.forEach(r => {
    const li = el('li', 'ct-' + r.treatment);
    li.innerHTML = '<span class="ct-tag"></span><div><button type="button" class="jlink"></button><em></em><p></p><q></q></div>';
    li.querySelector('.ct-tag').textContent = TREATMENT[r.treatment] || r.treatment;
    const j = li.querySelector('.jlink'); j.dataset.doc = r.doc_id; j.textContent = name(r.doc_id);
    li.querySelector('em').textContent = r.paragraph && r.paragraph !== '0' ? '\u00b6' + r.paragraph : '';
    li.querySelector('p').textContent = r.reason;
    const q = li.querySelector('q');
    if (r.quote) q.textContent = r.quote; else { q.remove(); li.querySelector('p').insertAdjacentHTML('afterend', '<p class="ct-noquote">No quote could be matched to the paragraph: read it before relying on this label.</p>'); }
    box.querySelector('ul').appendChild(li);
  });
  out.appendChild(box);
}
// For one judgment: the precedents it cites, and the other judgments that cite the same ones.
function precedentsOf(docId) {
  const mine = Object.keys(CITES).filter(k => CITES[k].refs[docId]);
  const shared = {};
  mine.forEach(k => Object.keys(CITES[k].refs).forEach(d => { if (d !== docId) (shared[d] = shared[d] || []).push(k); }));
  const related = Object.entries(shared).sort((a, b) => b[1].length - a[1].length);
  return {mine, related};
}
document.addEventListener('click', e => {
  const c = e.target.closest('.citeref');
  if (c) { openCitation(c.dataset.cite); return; }
  const j = e.target.closest('.jlink');
  if (j) openJudgment(j.dataset.doc);
});
function openStatute(k) {
  const v = STATUTES[k];
  if (!v) return;
  openPanel(
    '<div class="eyebrow">' + escapeHtml(v.act) + '</div>' +
    '<h2 class="big display">' + (v.kind === 'article' ? 'Art. ' : '§ ') + escapeHtml(v.number) + '</h2>' +
    '<h3>' + escapeHtml(v.title) + '</h3>' +
    '<div class="note" style="margin-top:18px">' + escapeHtml(v.note || '') + ' Source: <a href="' + escapeHtml(v.source) +
      '" target="_blank" rel="noopener">' + escapeHtml(v.source.replace(/^https?:\/\//, '')) + '</a>, retrieved ' + escapeHtml(v.retrieved) + '.</div>' +
    '<div class="passage" dir="auto">' + escapeHtml(v.text) + '</div>');
}
document.addEventListener('click', e => {
  const b = e.target.closest('.statref');
  if (b) openStatute(b.dataset.stat);
});
function tableCells(row) { return row.replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim()); }
function renderTable(rows, key, byN) {
  const body = rows.filter(r => !/^\|?\s*:?-{2,}/.test(r));
  if (!body.length) return '';
  let h = '<div class="tablewrap"><table><thead><tr>' +
          tableCells(body[0]).map(c => '<th dir="auto">' + inline(c, key, byN) + '</th>').join('') + '</tr></thead><tbody>';
  body.slice(1).forEach(r => {
    h += '<tr>' + tableCells(r).map(c => '<td dir="auto">' + inline(c, key, byN) + '</td>').join('') + '</tr>';
  });
  return h + '</tbody></table></div>';
}
function renderMarkdown(text, key, sources) {
  const byN = {};
  (sources || []).forEach(s => { byN[s.n] = s; });
  const lines = escapeHtml(text).split(/\r?\n/);
  let html = '', list = null, para = [], i = 0;
  const flush = () => { if (para.length) { html += '<p dir="auto">' + inline(para.join('<br>'), key, byN) + '</p>'; para = []; } };
  const close = () => { if (list) { html += '</' + list + '>'; list = null; } };
  while (i < lines.length) {
    const t = lines[i].trim();
    let m;
    if (!t) { flush(); close(); i++; continue; }
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(t)) { flush(); close(); html += '<hr class="rule" style="margin:20px 0">'; i++; continue; }
    if ((m = t.match(/^(#{1,4})\s+(.*)$/))) {
      flush(); close();
      const lvl = Math.min(m[1].length + 1, 4);
      html += '<h' + lvl + ' dir="auto">' + inline(m[2], key, byN) + '</h' + lvl + '>';
      i++; continue;
    }
    if (t.startsWith('|')) {
      flush(); close();
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) rows.push(lines[i++].trim());
      html += renderTable(rows, key, byN);
      continue;
    }
    if ((m = t.match(/^[-*•]\s+(.*)$/)) || (m = t.match(/^\d+[.)]\s+(.*)$/))) {
      flush();
      const kind = /^\d/.test(t) ? 'ol' : 'ul';
      // Numbered paragraphs separated by blank lines arrive as separate lists; keep their numbers.
      const num = kind === 'ol' ? parseInt(t, 10) : 1;
      if (list !== kind) { close(); html += '<' + kind + (num > 1 ? ' start="' + num + '"' : '') + '>'; list = kind; }
      html += '<li dir="auto">' + inline(m[1], key, byN) + '</li>';
      i++; continue;
    }
    close();
    para.push(t);
    i++;
  }
  flush(); close();
  return html;
}

// ===========================================================================
// a finished result: answer, grounding line, sources
// ===========================================================================
function citedNumbers(text) {
  const ns = new Set();
  (String(text || '').match(/\[S\d+(?:\s*[,;]\s*S?\d+)*\]/g) || []).forEach(m =>
    (m.match(/\d+/g) || []).forEach(n => ns.add(Number(n))));
  return ns;
}

// Fact check, with no AI: every name, section, number and quotation in a sentence that cites
// [S3] must appear in passage 3 itself or in its judgment's case details. A miss is shown,
// with where it was found instead if another passage has it.
const NUMBER_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
  'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'];
const NUMBER_TENS = {30: 'thirty', 40: 'forty', 50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety', 100: 'hundred'};
// Capitalised words that are legal vocabulary, not names. Anything else capitalised mid-sentence
// that never appears in lowercase in the sources is treated as a name and checked.
const NOT_NAMES = new Set(('court high supreme lahore islamabad sindh punjab balochistan khyber pakhtunkhwa pakistan pakistani ' +
  'federal provincial government state police magistrate judge justice bench section sections article articles constitution ' +
  'penal code criminal civil procedure act ordinance order rules rule schedule islamic republic sharia shariat qanun ' +
  'shahadat family guardians wards muslim law laws session sessions trial appellate appeal petition petitioner respondent ' +
  'appellant complainant accused prosecution defence state fir investigation officer station house mr mrs mst ms miss sir ' +
  'january february march april may june july august september october november december english urdu roman citizen lawyer ' +
  'however therefore thus also this that these those the there their they then when where which while what who in on at ' +
  'for from with without under after before during although because since if but and or not no yes it its he she his her ' +
  'as an a of to by be is are was were has have had do does did can could may might must shall should will would one two ' +
  'first second third key in-brief brief facts issues held holding outcome why matters similarities differences summary ' +
  'next steps note notes see source sources passage passages judgment judgments case cases ppc crpc cpc qso pld scmr plj ' +
  'ylr clc mld pcrlj nlr whether consequently additionally furthermore moreover accordingly finally firstly ' +
  'secondly thirdly meanwhile here both neither either unless until once given according regarding despite ' +
  'following overall instead otherwise yet still because ').split(/\s+/).filter(Boolean));
// A sentence ends at . ! or ? before a capital, a quote or a new item; not after "Rs." or "Mst.".
const SENTENCE_END = /(?<!\b(?:Rs|Mr|Mrs|Mst|Ms|Dr|No|Nos|Crl|Misc|Art|Arts|Sec|Ss|vs|v|etc|e\.g|i\.e|Hon|Pvt|Ltd|Co|St|PLD|SCMR|PLJ|YLR|CLC|MLD|PCrLJ)\.)(?<=[.!?])\s+(?=[A-Z0-9"“(*\-\[؀-ۿ])|\n+/;
// Letters and digits only, so a quotation matches whatever punctuation, spacing or brackets the model or the
// source put around the same words.
function squash(s) { return String(s || '').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ''); }
function factText(s) {
  return ' ' + String(s || '').toLowerCase().replace(/(\d),(?=\d{2,3}\b)/g, '$1').replace(/[‘’]/g, "'")
    .replace(/\s*-\s*/g, '-').replace(/\s+/g, ' ') + ' ';
}
function sourceFacts(s) {
  const p = s.parties || {};
  return factText([s.excerpt, s.title, s.case_no, s.court, s.judgment_type, (s.judges || []).join(' '),
                   Object.values(p).join(' '), (s.provisions || []).join(' ')].join(' \n '));
}
function hasNumber(hay, num, suffix) {
  const re = new RegExp('(^|[^0-9])' + num + (suffix ? '-?' + suffix : '') + '(?![0-9])', 'i');
  if (re.test(hay)) return true;
  const n = Number(num);
  const word = n <= 20 ? NUMBER_WORDS[n] : NUMBER_TENS[n];
  return !suffix && !!word && new RegExp('\\b' + word + '\\b').test(hay);
}
function factsIn(sentence, lowerAll) {
  const out = [];
  // A bold or plain label ("**Outcome:**", "Conclusion/Finding:") and a list marker are the model's
  // formatting, not facts: drop them before looking for names.
  let raw = sentence.replace(/\[S[^\]]*\]/g, ' ');
  raw = raw.replace(/^\s*(?:[-•]|\*(?=\s)|\d+[.)])?\s*/, '')
           .replace(/\*\*[^*]{1,80}?:\*\*|\*\*[^*]{1,80}\*\*\s*:/g, ' ')
           .replace(/^[A-Z][A-Za-z\/ -]{0,40}:\s+/, '');
  const text = raw.replace(/[*_`#>|]/g, ' ');
  // Quotations: pair the quote marks in order, so a closing mark never opens the next quotation.
  const quotes = [];
  for (let open = -1, m, re = /[“”"]/g; (m = re.exec(text));) {
    if (open < 0) open = m.index; else { quotes.push(text.slice(open + 1, m.index)); open = -1; }
  }
  quotes.filter(q => q.trim().length >= 12 && q.length <= 240).forEach(q =>
    out.push({kind: 'quotation', label: '“' + q.trim() + '”', test: hay => squash(hay).includes(squash(q))}));
  (text.match(/\b\d[\d,]*(?:\.\d+)?(?:\s*-\s*[A-Za-z]{1,2}\b)?/g) || []).forEach(raw => {
    const m = raw.replace(/,(?=\d)/g, '').match(/^(\d+(?:\.\d+)?)(?:\s*-\s*([A-Za-z]{1,2}))?$/);
    if (!m || /^(st|nd|rd|th)$/i.test(m[2] || '')) return;
    const num = m[1], suffix = m[2] || '';
    if (num.length === 1 && !suffix) return;   // "2 options", list counts: too noisy to check
    out.push({kind: suffix ? 'section' : 'number', label: raw.trim(), test: hay => hasNumber(hay, num.replace('.', '\\.'), suffix)});
  });
  const toks = [...text.matchAll(/[A-Za-z][A-Za-z'-]*/g)].map(m => ({w: m[0].replace(/['’]s$/, ''), at: m.index, end: m.index + m[0].length}));
  const words = toks.map(t => t.w);
  const common = x => NOT_NAMES.has(x) || lowerAll.includes(' ' + x + ' ');
  const isName = (w, i) => {
    if (!/^[A-Z][a-z]/.test(w) || w.length < 4) return false;
    const lw = w.toLowerCase();
    if (common(lw) || common(lw.replace(/s$/, ''))) return false;   // a common word, capitalised
    // The first word is capitalised anyway ("Consequently", "Citing", "Granted"); count it as a name
    // only when it is not such an opener and the next word is capitalised too ("Rashida Bibi").
    return i > 0 || (!/(ing|ly|ed)$/.test(lw) && !!(words[1] && /^[A-Z][a-z]/.test(words[1])));
  };
  // Neighbouring name words make one name ("Kamran Shah"); every word of it must be in the passage.
  let run = [];
  const flush = () => {
    if (!run.length) return;
    const parts = run.map(w => w.toLowerCase());
    // A single capitalised word ("Commission", "Declaration") is usually a defined term or an
    // expanded abbreviation, so only names of two or more words are checked.
    if (parts.length > 1) out.push({kind: 'name', label: run.join(' '), test: hay => parts.every(p => hay.includes(p))});
    run = [];
  };
  words.forEach((w, i) => {
    if (!isName(w, i)) return flush();
    // A name run ends at punctuation: "Registrar, Imran Qureshi" is a title and a name, not one name.
    if (run.length && !/^\s+$/.test(text.slice(toks[i - 1].end, toks[i].at))) flush();
    run.push(w);
  });
  flush();
  const seen = new Set();
  return out.filter(f => !seen.has(f.kind + f.label.toLowerCase()) && seen.add(f.kind + f.label.toLowerCase()));
}
// userText: what the user told us (their situation or the facts they typed). A detail found there is the user's
// own fact, cited as such, not a citation error, so it is not flagged. A style sample never counts as facts.
function checkFacts(answer, sources, userText) {
  const user = userText ? factText(userText) : '';
  const byN = {};
  sources.forEach(s => { byN[s.n] = sourceFacts(s); });
  // Words that occur in lowercase anywhere in the sources are vocabulary, not names.
  const lowerAll = ' ' + sources.map(s => String(s.excerpt || '')).join(' ').replace(/[^a-z' -]+/g, ' ') + ' ';
  const result = {checked: 0, sentences: 0, missing: []};
  String(answer || '').split(SENTENCE_END).forEach(sentence => {
    const ns = [...citedNumbers(sentence)].filter(n => byN[n]);
    if (!ns.length) return;
    result.sentences++;
    const hay = ns.map(n => byN[n]).join(' ');
    factsIn(sentence, lowerAll).forEach(f => {
      result.checked++;
      if (f.test(hay) || (user && f.test(user))) return;
      const elsewhere = Object.keys(byN).map(Number).filter(n => f.test(byN[n]));
      result.missing.push({kind: f.kind, label: f.label, cited: ns, elsewhere});
    });
  });
  return result;
}

// ===========================================================================
// Listen: the answer read aloud by the browser's own speech voices (no server, no model).
// Shown only when the browser has a voice for the answer's language: English almost everywhere, Urdu
// where an Urdu voice is installed (Edge ships neural Urdu voices). Roman Urdu has no usable voice.
// ===========================================================================
function speechText(md) {
  return String(md || '')
    .replace(/\[S[^\]]*\]/g, '')                 // citation markers
    .replace(/^\s*\|.*\|\s*$/gm, '')             // table rows are not read aloud
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/[*_`>]/g, '')
    .replace(/^\s*(?:[-•]|\d+[.)])\s+/gm, '')
    .replace(/\s*\n+\s*/g, '. ')                 // each line becomes a sentence
    .replace(/([.!?۔؟])\s*\./g, '$1')
    .replace(/\s{2,}/g, ' ').trim();
}
function speechChunks(text, max) {
  const out = [];
  let cur = '';
  (text.match(/[^.!?۔؟]+[.!?۔؟]*\s*/g) || [text]).forEach(sentence => {
    if (cur && (cur + sentence).length > max) { out.push(cur.trim()); cur = ''; }
    cur += sentence;
  });
  if (cur.trim()) out.push(cur.trim());
  return out;
}
function pickVoice(voices, lang) {
  const pool = (voices || []).filter(v => String(v.lang).toLowerCase().startsWith(lang));
  return pool.find(v => /natural|online/i.test(v.name)) || pool.find(v => v.localService) || pool[0] || null;
}
// Listen: a human voice from the server's speech model (each passage is made once and then replayed from the cache),
// falling back to the browser's own voice if the server cannot make one.
function playHuman(chunks, language, tone, hooks) {
  let stopped = false, current = null;
  const urls = [];
  const clip = text => fetch('/api/speak', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                            body: JSON.stringify({text, language, tone})})
    .then(r => { if (!r.ok) throw new Error('voice'); return r.blob(); })
    .then(b => { const u = URL.createObjectURL(b); urls.push(u); return u; });
  (async () => {
    let next = clip(chunks[0]);
    for (let i = 0; i < chunks.length && !stopped; i++) {
      const url = await next;
      if (i + 1 < chunks.length) { next = clip(chunks[i + 1]); next.catch(() => {}); }   // the next passage is made while this one plays
      if (stopped) break;
      await new Promise((resolve, reject) => {
        const a = new Audio(url);
        current = a;
        a.onended = resolve;
        a.onerror = () => reject(new Error('play'));
        if (i === 0 && hooks.start) hooks.start();
        a.play().catch(reject);
      });
    }
    urls.forEach(u => URL.revokeObjectURL(u));
    hooks.end(null);
  })().catch(err => { urls.forEach(u => URL.revokeObjectURL(u)); hooks.end(stopped ? null : err); });
  return {stop() { stopped = true; if (current) current.pause(); }};
}
function listenButton(data) {
  const lang = ['en', 'ur', 'roman_ur'].includes(data.language) ? data.language : 'en';
  if (!speechText(data.answer)) return null;
  const btn = el('button', 'copy', 'Listen');
  btn.type = 'button';
  btn.style.marginLeft = '0';
  let human = null;
  const idle = msg => {
    btn.textContent = msg || 'Listen'; btn.classList.remove('listening'); human = null;
    if (msg) setTimeout(() => { if (btn.textContent === msg) btn.textContent = 'Listen'; }, 2500);
  };
  const browserVoice = () => {
    const code = lang === 'ur' ? 'ur' : 'en';
    const voice = lang !== 'roman_ur' && 'speechSynthesis' in window ? pickVoice(speechSynthesis.getVoices(), code) : null;
    if (!voice) { idle('Voice unavailable'); return; }
    speechSynthesis.cancel();
    const chunks = speechChunks(speechText(data.answer), 220);
    chunks.forEach((c, i) => {
      const u = new SpeechSynthesisUtterance(c);
      u.voice = voice; u.lang = voice.lang; u.rate = 0.95;
      if (i === chunks.length - 1) { u.onend = () => idle(); u.onerror = () => idle(); }
      speechSynthesis.speak(u);
    });
    btn.textContent = 'Stop'; btn.classList.add('listening');
  };
  btn.addEventListener('click', () => {
    if (human || btn.classList.contains('listening')) {
      if (human) human.stop();
      if ('speechSynthesis' in window) speechSynthesis.cancel();
      idle();
      return;
    }
    btn.textContent = 'Preparing the voice…'; btn.classList.add('listening');
    human = playHuman(speechChunks(speechText(data.answer), 600), lang, segValue('mode') === 'lawyer' ? 'formal' : 'plain', {
      start() { btn.textContent = 'Stop'; },
      end(err) { if (!err) { idle(); return; } human = null; btn.classList.remove('listening'); browserVoice(); },
    });
  });
  return btn;
}

// Red flags: a list item that starts with a bold High / Medium / Low gets a severity label.
function markSeverity(root) {
  root.querySelectorAll('li').forEach(li => {
    const first = li.firstElementChild;
    const m = first && first.tagName === 'STRONG' && li.textContent.trim().startsWith(first.textContent.trim()) &&
              first.textContent.trim().replace(/[:.\-\s]+$/, '').match(/^(High|Medium|Low)$/i);
    if (!m) return;
    const level = m[1].toLowerCase();
    li.classList.add('sev', 'sev-' + level);
    first.outerHTML = '<span class="sevtag">' + level + '</span>';
    li.innerHTML = li.innerHTML.replace(/(<\/span>)\s*[:\-–]\s*/, '$1 ');
  });
}

function renderSourceList(container, sources, key, cited) {
  if (!sources || !sources.length) return;
  const groups = [], byDoc = {};
  sources.forEach(s => {
    const id = s.doc_id || ('src' + s.n);
    if (!byDoc[id]) { byDoc[id] = {first: s, items: [], cited: false}; groups.push(byDoc[id]); }
    byDoc[id].items.push(s);
    if (cited.has(s.n)) byDoc[id].cited = true;
  });
  groups.sort((a, b) => b.cited - a.cited);
  const ul = el('ul', 'srclist');
  groups.forEach(g => {
    const li = el('li', g.cited ? 'cited' : '');
    li.innerHTML = '<span class="who"></span><span class="cno"></span><span class="paras"></span>';
    li.querySelector('.who').textContent = caseTitle(g.first);
    // Without extracted parties the title already is the case number; show the court instead.
    li.querySelector('.cno').textContent = hasParties(g.first) ? caseNumber(g.first) : (g.first.court || '');
    g.items.forEach(s => {
      const b = el('button', '', '<span>' + s.n + '</span>' + (s.paragraph && s.paragraph !== '0' ? '¶' + escapeHtml(s.paragraph) : 'Header'));
      b.type = 'button'; b.dataset.key = key; b.dataset.n = s.n;
      li.querySelector('.paras').appendChild(b);
    });
    ul.appendChild(li);
  });
  container.appendChild(ul);
}

// Word-by-word reveal of a finished answer, like a chat model typing. The answer is already built
// (citation chips, bold, lists, fact marks); its text is emptied and given back a few words at a
// time, and each element appears when its first word does. The footer waits until the end.
// Pure helpers first, so they can be tested without a browser.
function revealDuration(total) { return Math.min(6000, Math.max(1200, total * 6)); }
function revealCount(total, elapsed, duration) { return Math.min(total, Math.floor(total * Math.min(1, elapsed / duration))); }
function snapWord(text, shown) { if (shown <= 0) return 0; while (shown < text.length && /\S/.test(text.charAt(shown))) shown++; return Math.min(shown, text.length); }
function revealAnswer(ans, tail, done) {
  const walker = document.createTreeWalker(ans, NodeFilter.SHOW_TEXT);
  const nodes = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) if (/\S/.test(n.nodeValue)) nodes.push({node: n, text: n.nodeValue});
  const total = nodes.reduce((s, x) => s + x.text.length, 0);
  const full = nodes.map(x => x.text).join('');
  const waiting = [];
  nodes.forEach((x, i) => {
    x.start = i ? nodes[i - 1].start + nodes[i - 1].text.length : 0;
    for (let a = x.node.parentElement; a && a !== ans; a = a.parentElement) {
      if (a.classList.contains('rv-wait')) { if (!a._rv) a._rv = i; continue; }
      if (a._rv === undefined) { a._rv = i; a.classList.add('rv-wait'); waiting.push(a); }
    }
    x.node.nodeValue = '';
  });
  tail.forEach(t => t.classList.add('rv-wait'));
  const caret = el('span', 'rv-caret');
  let finished = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    nodes.forEach(x => { if (x.node.isConnected) x.node.nodeValue = x.text; });
    waiting.forEach(a => a.classList.remove('rv-wait'));
    tail.forEach(t => t.classList.remove('rv-wait'));
    caret.remove();
    ans.removeEventListener('click', finish);
    if (done) done();
  };
  if (!total) { finish(); return; }
  ans.addEventListener('click', finish);
  const duration = revealDuration(total);
  const t0 = performance.now();
  const step = () => {
    if (finished || !ans.isConnected) { finish(); return; }
    const shown = snapWord(full, revealCount(total, performance.now() - t0, duration));
    let last = null;
    nodes.forEach(x => {
      const n = Math.max(0, Math.min(x.text.length, shown - x.start));
      if (n && x.node.nodeValue.length !== n) x.node.nodeValue = x.text.slice(0, n);
      if (n) last = x;
    });
    waiting.slice().forEach(a => {
      if (last && a._rv <= nodes.indexOf(last)) { a.classList.remove('rv-wait'); waiting.splice(waiting.indexOf(a), 1); }
    });
    if (last) last.node.parentNode.insertBefore(caret, last.node.nextSibling);
    if (shown >= total) finish(); else requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
// ---- Check a brief: every section against the official text we hold, every reported case against the library, and
// every year against the calendar. No AI: a section we do not hold is flagged for checking, never called invented.
function checkBrief(text, statutes, cites, year) {
  const found = [], byId = {}, src = String(text || '');
  const add = (item) => { if (byId[item.id]) { byId[item.id].count++; return; } item.count = 1; byId[item.id] = item; found.push(item); };
  const stat = (re, keyOf) => { for (const m of src.matchAll(re)) { const k = keyOf(m), v = statutes[k];
    add({id: 's' + k, kind: 'statute', at: m.index, ref: m[1].replace(/\s+/g, ' ').trim(), key: k, status: v ? 'verified' : 'wrong', title: v ? v.title : ''}); } };
  stat(PPC_RE, m => 'ppc:' + m[2].toUpperCase());
  stat(CONST_RE, m => 'const:' + m[2].toUpperCase());
  stat(LAW_RE, m => lawPrefix(m[3]) + m[2].toUpperCase().replace('-', ''));
  for (const m of src.matchAll(CITE_RE)) {
    const y = +(m[2] || m[5] || m[8]), k = citeKey(m), c = cites[k];
    add({id: 'c' + k, kind: 'case', at: m.index, ref: m[0], key: k, year: y,
         status: y > year ? 'wrong' : c ? 'verified' : 'check', judgments: c ? Object.keys(c.refs).length : 0});
  }
  return found.sort((a, b) => a.at - b.at);
}
function briefNote(f, n) {
  if (f.kind === 'statute') return f.status === 'verified' ? 'In the official text: ' + f.title
    : 'Not in the official text we hold. It may be repealed, mistyped or invented: check it before filing.';
  if (f.status === 'wrong') return 'Reported in ' + f.year + ', a year that has not come yet. This report cannot exist.';
  return f.status === 'verified' ? 'In the library: relied on by ' + plural(f.judgments, 'judgment')
    : 'Not among the ' + n + ' judgments in the library. Look it up in the law report before relying on it.';
}
function briefReport(found, n) {
  const c = {verified: 0, check: 0, wrong: 0};
  found.forEach(f => { c[f.status]++; });
  const lines = ['**Checked ' + plural(found.filter(f => f.kind === 'statute').length, 'statute reference') + ' and ' +
                 plural(found.filter(f => f.kind === 'case').length, 'case citation') + ':** ' + c.verified + ' verified, ' + c.check +
                 ' to check, ' + c.wrong + ' wrong.', ''];
  found.forEach(f => lines.push('- **' + {verified: 'Verified', check: 'Check', wrong: 'Wrong'}[f.status] + '** ' + f.ref + ': ' + briefNote(f, n)));
  return lines.join('\n');
}
function renderBriefCheck(container, text) {
  const n = (judgments || []).length, found = checkBrief(text, STATUTES, CITES, new Date().getFullYear());
  const box = el('div', 'brief-check');
  if (!found.length) { box.innerHTML = '<p class="ws-empty">No section, article or reported case was found in this text.</p>'; container.appendChild(box); return {found, md: ''}; }
  const c = {verified: 0, check: 0, wrong: 0};
  found.forEach(f => { c[f.status]++; });
  box.innerHTML = '<div class="bc-sum"><div><b>' + c.wrong + '</b><span>Wrong</span></div><div><b>' + c.check + '</b><span>To check</span></div>' +
    '<div><b>' + c.verified + '</b><span>Verified</span></div><p>Every section is looked up in the official text, every reported case in the library, ' +
    'every year against today. No AI is used.</p></div><ul class="bc-list"></ul>';
  const ul = box.querySelector('.bc-list');
  found.forEach(f => {
    const li = el('li', 'bc-' + f.status);
    li.innerHTML = '<span class="bc-tag"></span><div><b dir="auto"></b><p></p></div>';
    li.querySelector('.bc-tag').textContent = {verified: 'Verified', check: 'Check', wrong: 'Wrong'}[f.status];
    li.querySelector('b').textContent = f.ref + (f.count > 1 ? '  \u00d7' + f.count : '');
    li.querySelector('p').textContent = briefNote(f, n);
    if (f.kind === 'statute' && f.status === 'verified') {
      const b = el('button', 'copy', 'Read the section'); b.type = 'button'; b.style.marginLeft = '0';
      b.addEventListener('click', () => openStatute(f.key)); li.lastChild.appendChild(b);
    }
    if (f.kind === 'case' && f.status === 'verified') li.lastChild.insertAdjacentHTML('beforeend', '<button type="button" class="citeref copy" data-cite="' + escapeHtml(f.key) + '">Which judgments rely on it</button>');
    ul.appendChild(li);
  });
  container.appendChild(box);
  return {found, md: briefReport(found, n)};
}
const SAMPLE_PETITION = 'IN THE COURT OF THE SESSIONS JUDGE, LAHORE\n\nPost-arrest bail petition under Section 497 Cr.P.C.\nTariq Mehmood v. The State, FIR No. 214/2024, Police Station Gulberg, Lahore, under Section 489-F PPC.\n\nGROUNDS\n1. The petitioner has been in custody since 5 March 2024. The offence under Section 489-F PPC does not fall within the prohibitory clause of Section 497 Cr.P.C.\n2. The cheque was handed over as security for a business loan; the dispute is civil. Reliance is placed on 2013 SCMR 1205 and PLD 2016 SC 808.\n3. Further reliance is placed on 2019 SCMR 1234 and PLD 2031 SC 77.\n4. Even otherwise the case is one under Section 489-Z PPC, a bailable offence, and Section 300 Cr.P.C. bars a second trial on the same facts.\n5. The right to a fair trial is guaranteed by Article 10A of the Constitution.\n\nPRAYER\nIt is respectfully prayed that the petitioner be released on bail.';
// ---- Rights right now: when someone is stopped or arrested. Each right is the law's own words, from the official text.
const RIGHTS = [
  {k: 'const:10', from: '(1)', to: '(2)', en: 'You must be told why you are being arrested, and you may consult a lawyer of your choice.',
   ur: 'آپ کو گرفتاری کی وجہ بتانا لازم ہے، اور آپ اپنی پسند کے وکیل سے مشورہ کر سکتے ہیں۔'},
  {k: 'const:10', from: '(2)', to: '(3)', en: 'You must be produced before a magistrate within 24 hours of your arrest.',
   ur: 'گرفتاری کے چوبیس گھنٹوں کے اندر آپ کو مجسٹریٹ کے سامنے پیش کرنا لازم ہے۔'},
  {k: 'crpc:61', en: 'The police may not keep you more than 24 hours without a magistrate’s order.',
   ur: 'مجسٹریٹ کے حکم کے بغیر پولیس آپ کو چوبیس گھنٹوں سے زیادہ نہیں رکھ سکتی۔'},
  {k: 'const:14', from: '(2)', en: 'No one may be tortured to get evidence.',
   ur: 'ثبوت حاصل کرنے کے لیے کسی پر تشدد نہیں کیا جا سکتا۔'},
  {k: 'crpc:50', en: 'You may not be restrained more than needed to stop you escaping.',
   ur: 'فرار روکنے سے زیادہ آپ پر سختی نہیں کی جا سکتی۔'},
  {k: 'crpc:52', en: 'A woman may be searched only by another woman, with decency.',
   ur: 'عورت کی تلاشی صرف دوسری عورت ہی لے سکتی ہے، شائستگی کے ساتھ۔'},
  {k: 'crpc:496', en: 'For a bailable offence, you must be released when you offer bail.',
   ur: 'قابلِ ضمانت جرم میں ضمانت دینے پر آپ کو رہا کرنا لازم ہے۔'},
  {k: 'crpc:164', en: 'A confession is recorded by a magistrate, who must not be a police officer.',
   ur: 'اعترافی بیان مجسٹریٹ ریکارڈ کرتا ہے، پولیس افسر نہیں۔'},
  {k: 'const:10A', en: 'You have the right to a fair trial and due process.',
   ur: 'آپ کو منصفانہ سماعت اور قانونی کارروائی کا حق حاصل ہے۔'},
];
const RIGHTS_DO = {en: ['Stay calm and do not resist.', 'Ask: why am I being arrested, and to which police station am I being taken?',
                        'Note the officers’ names and the police station. Tell a family member.', 'Call a lawyer, or the free helplines below.'],
                   ur: ['پرسکون رہیں اور مزاحمت نہ کریں۔', 'پوچھیں: مجھے کیوں گرفتار کیا جا رہا ہے، اور کس تھانے لے جایا جا رہا ہے؟',
                        'اہلکاروں کے نام اور تھانے کا نام یاد رکھیں۔ گھر والوں کو بتائیں۔', 'وکیل یا نیچے دیے گئے مفت ہیلپ لائن پر فون کریں۔']};
function lawWords(r) {
  const v = STATUTES[r.k];
  if (!v) return '';
  let t = v.text;
  if (r.from && t.includes(r.from)) t = t.slice(t.indexOf(r.from));
  if (r.to && t.includes(r.to)) t = t.slice(0, t.indexOf(r.to));
  t = t.trim();
  return t.length > 420 ? t.slice(0, 417).replace(/\s+\S*$/, '') + '…' : t;
}
function openRights(lang) {
  lang = lang || (segValue('language') === 'en' ? 'en' : 'ur');
  const L = lang === 'ur', items = RIGHTS.filter(r => STATUTES[r.k]);
  openPanel('<div class="rights" dir="' + (L ? 'rtl' : 'ltr') + '"><div class="eyebrow" dir="ltr">Rights right now · stopped or arrested</div>' +
    '<h2 class="display" dir="auto">' + (L ? 'گرفتاری کے وقت آپ کے حقوق' : 'Your rights if you are arrested') + '</h2>' +
    '<div class="rights-bar" dir="ltr"><button type="button" data-l="ur"' + (L ? ' class="on"' : '') + '>اردو</button><button type="button" data-l="en"' + (L ? '' : ' class="on"') +
    '>English</button><button type="button" class="rights-say">' + (L ? 'سنیں' : 'Read aloud') + '</button><button type="button" class="rights-vid">' + (L ? 'ویڈیو بنائیں' : 'Make a video') + '</button></div>' +
    '<ol class="rights-do">' + RIGHTS_DO[lang].map(t => '<li>' + escapeHtml(t) + '</li>').join('') + '</ol>' +
    '<ul class="rights-list"></ul>' +
    '<p class="privacy" dir="ltr">Each right above quotes the official text of the Constitution or the Code of Criminal Procedure. General information, not legal advice.</p></div>');
  const panel = $('panel'), ul = panel.querySelector('.rights-list');
  items.forEach(r => {
    const li = el('li');
    li.innerHTML = '<b dir="auto"></b><blockquote dir="ltr"></blockquote><button type="button" class="statref" dir="ltr"></button>';
    li.querySelector('b').textContent = L ? r.ur : r.en;
    li.querySelector('blockquote').textContent = lawWords(r);
    const b = li.querySelector('.statref'); b.dataset.stat = r.k; b.textContent = statuteLabel(r.k, STATUTES[r.k]) + ' →';
    ul.appendChild(li);
  });
  panel.querySelector('.rights').appendChild(helpStrip());
  panel.querySelectorAll('.rights-bar [data-l]').forEach(b => b.addEventListener('click', () => openRights(b.dataset.l)));
  panel.querySelector('.rights-vid').addEventListener('click', () => { closePanel(); makeVideo(rightsVideoScript(lang)); });
  const say = panel.querySelector('.rights-say');
  const voice = 'speechSynthesis' in window ? pickVoice(speechSynthesis.getVoices(), lang) : null;
  if (!voice) say.hidden = true;
  say.addEventListener('click', () => {
    if (speechSynthesis.speaking) { speechSynthesis.cancel(); return; }
    const text = RIGHTS_DO[lang].concat(items.map(r => L ? r.ur : r.en)).join(' ');
    speechChunks(text, 220).forEach(c => { const u = new SpeechSynthesisUtterance(c); u.voice = voice; u.lang = voice.lang; u.rate = 0.92; speechSynthesis.speak(u); });
  });
}
// ---- Never miss a date: the dates an answer gives (a hearing, a deadline, the events in an FIR), as a calendar file.
// Only dates written out in the answer are used; a relative one ("within 15 days") has no day to put in a calendar.
const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
function monthOf(name) { const n = name.toLowerCase().slice(0, 3); return MONTHS.findIndex(m => m.startsWith(n)) + 1; }
function datesIn(text, today) {
  const M = '(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)';
  const forms = [
    [new RegExp('\\b(\\d{1,2})(?:st|nd|rd|th)?(?:\\s+of)?[\\s.-]+' + M + '\\.?[,\\s.-]+(\\d{4})\\b', 'gi'), m => [+m[3], monthOf(m[2]), +m[1]]],
    [new RegExp('\\b' + M + '\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?,?\\s+(\\d{4})\\b', 'gi'), m => [+m[3], monthOf(m[1]), +m[2]]],
    [/\b(\d{4})-(\d{2})-(\d{2})\b/g, m => [+m[1], +m[2], +m[3]]],
    [/\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b/g, m => [+m[3], +m[2], +m[1]]],   // day first, as written in Pakistan
  ];
  const now = today || new Date(), start = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
  const out = [], seen = new Set();
  String(text || '').split('\n').forEach(raw => {
    const line = raw.replace(/^\s*\||\|\s*$/g, '').replace(/\[S[\d,\s\u2013-]*\]/g, '').replace(/^[\s#>*-]+|\d+\.\s+/, '').replace(/\*\*|__|`/g, '')
                    .replace(/\s*\|\s*/g, ' \u00b7 ').replace(/^\u00b7\s*|\s*\u00b7$/g, '').replace(/\s+/g, ' ').trim();
    const taken = [];
    forms.forEach(([re, parts]) => {
      for (const m of raw.matchAll(re)) {
        if (taken.some(([a, b]) => m.index < b && m.index + m[0].length > a)) continue;   // one date read once
        taken.push([m.index, m.index + m[0].length]);
        const [y, mo, d] = parts(m);
        if (y < 1900 || y > 2100 || mo < 1 || mo > 12) continue;
        const t = Date.UTC(y, mo - 1, d), check = new Date(t);
        if (check.getUTCDate() !== d) continue;   // 31 February and the like
        const iso = y + '-' + String(mo).padStart(2, '0') + '-' + String(d).padStart(2, '0');
        if (seen.has(iso + line)) continue;
        seen.add(iso + line);
        out.push({iso, time: t, upcoming: t >= start, line: line.length > 180 ? line.slice(0, 177) + '\u2026' : line});
      }
    });
  });
  return out.sort((a, b) => a.time - b.time);
}
function icsEscape(t) { return String(t).replace(/\\/g, '\\\\').replace(/;/g, '\\;').replace(/,/g, '\\,').replace(/\r?\n/g, '\\n'); }
function icsFold(line) {   // lines of at most 75 characters, continued with a space
  const out = [];
  for (let i = 0; i < line.length; i += i ? 74 : 75) out.push((i ? ' ' : '') + line.slice(i, i ? i + 74 : 75));
  return out.join('\r\n');
}
function icsOf(events, title, reminders, stamp) {
  const ymd = iso => iso.replace(/-/g, ''), now = (stamp || new Date()).toISOString().replace(/[-:]/g, '').replace(/\.\d+/, '');
  const lines = ['BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//Wakeel AI//Dates//EN', 'CALSCALE:GREGORIAN', 'METHOD:PUBLISH'];
  events.forEach((e, i) => {
    const next = new Date(e.time + 86400000).toISOString().slice(0, 10);
    lines.push('BEGIN:VEVENT', 'UID:' + ymd(e.iso) + '-' + i + '-' + now + '@wakeel.ai', 'DTSTAMP:' + now,
               'DTSTART;VALUE=DATE:' + ymd(e.iso), 'DTEND;VALUE=DATE:' + ymd(next),
               'SUMMARY:' + icsEscape('Wakeel: ' + (e.line.length > 70 ? e.line.slice(0, 67) + '\u2026' : e.line)),
               'DESCRIPTION:' + icsEscape(e.line + '\n\nFrom: ' + title + '\nCheck this date against your papers.'));
    (reminders || []).forEach(days => lines.push('BEGIN:VALARM', 'ACTION:DISPLAY', 'DESCRIPTION:' + icsEscape(e.line.slice(0, 70)),
                                                 'TRIGGER:-P' + days + 'D', 'END:VALARM'));
    lines.push('END:VEVENT');
  });
  lines.push('END:VCALENDAR');
  return lines.map(icsFold).join('\r\n') + '\r\n';
}
function calendarButton(data, title) {
  const dates = datesIn(data.answer);
  if (!dates.length || data.route === 'ERROR') return null;
  const up = dates.filter(d => d.upcoming).length;
  const b = el('button', 'copy', up ? 'Add ' + plural(up, 'upcoming date') + ' to calendar' : 'Save ' + plural(dates.length, 'date') + ' to a calendar');
  b.type = 'button'; b.style.marginLeft = '0';
  b.title = 'A calendar file with these dates, for Google Calendar, Outlook or your phone';
  b.addEventListener('click', () => openCalendar(dates, title));
  return b;
}
function openCalendar(dates, title) {
  const fmt = d => new Date(d.time).toLocaleDateString(undefined, {weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC'});
  openPanel('<div class="eyebrow">Never miss a date</div><h3>' + plural(dates.length, 'date') + ' in this answer</h3>' +
    '<p class="privacy">Upcoming dates are ticked. Each one is taken word for word from the answer, so check it against your papers.</p>' +
    '<ul class="calist">' + dates.map((d, i) => '<li><label><input type="checkbox" data-i="' + i + '"' + (d.upcoming ? ' checked' : '') + '><span><b>' + fmt(d) +
      '</b><em class="' + (d.upcoming ? 'up' : '') + '">' + (d.upcoming ? 'Upcoming' : 'Past') + '</em></span><span class="cl"></span></label></li>').join('') + '</ul>' +
    '<div class="calrem"><span class="eyebrow">Remind me</span><label><input type="checkbox" data-r="1" checked> a day before</label>' +
    '<label><input type="checkbox" data-r="7"' + (dates.some(d => d.upcoming) ? ' checked' : '') + '> a week before</label></div>' +
    '<button class="btn" type="button" id="cal-go">Download the calendar file</button><p class="privacy" id="cal-note">Opens in Google Calendar, Outlook, or the calendar on your phone. Made in this browser; nothing is sent anywhere.</p>');
  const panel = $('panel');
  panel.querySelectorAll('.calist .cl').forEach((x, i) => { x.textContent = dates[i].line; });
  const go = $('cal-go');
  const sync = () => { const n = panel.querySelectorAll('.calist input:checked').length; go.disabled = !n; go.textContent = n ? 'Download ' + plural(n, 'date') + ' (.ics)' : 'Tick a date'; };
  panel.querySelectorAll('.calist input').forEach(x => x.addEventListener('change', sync));
  sync();
  go.addEventListener('click', () => {
    const chosen = [...panel.querySelectorAll('.calist input:checked')].map(x => dates[+x.dataset.i]);
    const reminders = [...panel.querySelectorAll('.calrem input:checked')].map(x => +x.dataset.r);
    const blob = new Blob([icsOf(chosen, title || 'Wakeel AI', reminders)], {type: 'text/calendar;charset=utf-8'});
    const a = el('a');
    a.href = URL.createObjectURL(blob); a.download = 'wakeel-dates.ics';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    $('cal-note').textContent = 'Saved ' + plural(chosen.length, 'date') + '. Open wakeel-dates.ics to add them to your calendar.';
  });
}
// Free help from people, not the app. Numbers checked in October 2026 against the services' own pages and reports.
const HELPLINES = [
  ['1818', 'tel:1818', 'Supreme Court legal call centre', 'court information, free legal guidance, help applying for legal aid'],
  ['0800-70806', 'tel:080070806', 'Legal Aid Society', 'free legal advice, 24 hours, from anywhere in Pakistan'],
  ['1099', 'tel:1099', 'Ministry of Human Rights', 'free legal advice, counselling and referral'],
  ['15', 'tel:15', 'Police emergency', 'if you or someone is in danger now'],
];
function helpStrip() {
  const box = el('div', 'help');
  box.innerHTML = '<b>Talk to a person, free</b><div class="help-lines">' + HELPLINES.map(h =>
    '<a href="' + h[1] + '"><strong>' + h[0] + '</strong><span>' + h[2] + '</span><em>' + h[3] + '</em></a>').join('') +
    '</div><p>Checked October 2026. These lines are run by the services named, not by Wakeel.</p>';
  return box;
}
// ---- Legal words, explained: tap a marked word for its meaning in English and Urdu and the law behind it.
// The meanings are fixed text written for this app (not AI); the law is the official text we hold.
// cs: case-sensitive patterns (acronyms); ci: any case. The longer form comes first where two overlap.
const GLOSSARY = [
  {id: 'noncog', ci: ['non-cognizable offen[cs]e', 'non-cognizable case', 'non-cognizable'], law: 'crpc:155',
   en: 'An offence the police may not investigate, or arrest for, without a magistrate’s order.',
   ur: 'ایسا جرم جس میں پولیس مجسٹریٹ کے حکم کے بغیر نہ تفتیش کر سکتی ہے نہ گرفتاری۔'},
  {id: 'cog', ci: ['cognizable offen[cs]e', 'cognizable case'], urp: ['قابلِ? دست اندازی'], law: 'crpc:4', find: 'cognizable offence” means',
   en: 'An offence for which the police may arrest without a warrant.',
   ur: 'ایسا جرم جس میں پولیس وارنٹ کے بغیر گرفتار کر سکتی ہے۔'},
  {id: 'fir', cs: ['FIRs?', 'F\\.I\\.R\\.?'], ci: ['first information report'], urp: ['ایف آئی آر'], law: 'crpc:154',
   en: 'First Information Report: the police’s written record of a cognizable offence. It starts the investigation.',
   ur: 'ابتدائی اطلاعاتی رپورٹ: کسی قابلِ دست اندازی جرم کی پولیس میں درج تحریری رپورٹ، جس سے تفتیش شروع ہوتی ہے۔'},
  {id: 'superdari', ci: ['superdari', 'supurdari', 'supurdgi'], urp: ['سپرداری'], law: 'crpc:516A',
   en: 'Temporary custody of seized property, such as a vehicle, given to its owner by a court until the case is decided.',
   ur: 'ضبط شدہ مال، جیسے گاڑی، کی عارضی حوالگی جو عدالت مقدمے کے فیصلے تک مالک کو دیتی ہے۔'},
  {id: 'challan', ci: ['challans?'], urp: ['چالان'], law: 'crpc:173',
   en: 'The police report sent to court when the investigation is finished, so the trial can begin.',
   ur: 'تفتیش مکمل ہونے پر پولیس کی عدالت میں بھیجی گئی رپورٹ، تاکہ مقدمہ شروع ہو سکے۔'},
  {id: 'prearrest', ci: ['pre-arrest bail', 'bail before arrest', 'ad-interim pre-arrest bail'], urp: ['ضمانت قبل از گرفتاری'], law: 'crpc:498',
   en: 'A court order, made before any arrest, that you are to be released on bail if you are arrested.',
   ur: 'گرفتاری سے پہلے عدالت کا حکم کہ گرفتاری کی صورت میں آپ کو ضمانت پر رہا کیا جائے۔'},
  {id: 'postarrest', ci: ['post-arrest bail', 'bail after arrest'], urp: ['ضمانت بعد از گرفتاری'], law: 'crpc:497',
   en: 'Release on bail from custody after an arrest.',
   ur: 'گرفتاری کے بعد ضمانت پر حراست سے رہائی۔'},
  {id: 'nonbail', ci: ['non-bailable offen[cs]es?', 'non-bailable'], law: 'crpc:497',
   en: 'An offence in which bail is up to the court, not a right.',
   ur: 'ایسا جرم جس میں ضمانت حق نہیں بلکہ عدالت کی صوابدید ہے۔'},
  {id: 'bailable', ci: ['bailable offen[cs]es?'], law: 'crpc:496',
   en: 'An offence in which bail is a right: you must be released when you offer bail.',
   ur: 'ایسا جرم جس میں ضمانت حق ہے: ضمانت دینے پر رہائی لازم ہے۔'},
  {id: 'prohibitory', ci: ['prohibitory clause'], law: 'crpc:497',
   en: 'The part of section 497 CrPC under which bail is normally refused for offences punishable with death, life imprisonment or ten years.',
   ur: 'دفعہ 497 کا وہ حصہ جس کے تحت موت، عمر قید یا دس سال سزا والے جرائم میں عام طور پر ضمانت نہیں دی جاتی۔'},
  {id: 'remand', ci: ['physical remand', 'judicial remand', 'remand'], urp: ['ریمانڈ'], law: 'crpc:167',
   en: 'A magistrate’s order keeping an arrested person in police custody (physical remand) or in jail (judicial remand) beyond 24 hours.',
   ur: 'مجسٹریٹ کا حکم جس سے گرفتار شخص کو چوبیس گھنٹے سے زیادہ پولیس (جسمانی ریمانڈ) یا جیل (جوڈیشل ریمانڈ) میں رکھا جاتا ہے۔'},
  {id: 'warrant', ci: ['warrants? of arrest', 'arrest warrants?'], law: 'crpc:75',
   en: 'A written court order telling the police to arrest a named person.',
   ur: 'عدالت کا تحریری حکم جس میں پولیس کو کسی شخص کو گرفتار کرنے کا کہا جاتا ہے۔'},
  {id: 'summons', ci: ['summons'], law: 'crpc:68',
   en: 'A court order to appear before it on a fixed date.',
   ur: 'عدالت کا حکم کہ مقررہ تاریخ پر حاضر ہوں۔'},
  {id: 'writ', ci: ['writ petitions?', 'constitutional petitions?'], urp: ['رٹ پٹیشن'], law: 'const:199',
   en: 'A petition to the High Court under Article 199 of the Constitution, against an unlawful act or order of a public body.',
   ur: 'آئین کے آرٹیکل 199 کے تحت ہائی کورٹ میں درخواست، کسی سرکاری ادارے کے غیر قانونی اقدام یا حکم کے خلاف۔'},
  {id: 'quash', ci: ['quashment', 'quashing', 'quashed', 'quash'], law: 'crpc:561A',
   en: 'When the High Court ends criminal proceedings, or an FIR, using its inherent power.',
   ur: 'جب ہائی کورٹ اپنے موروثی اختیار سے فوجداری کارروائی یا ایف آئی آر ختم کر دے۔'},
  {id: 'compound', ci: ['compoundable', 'compounding', 'compounded'], law: 'crpc:345',
   en: 'Settling an offence between the parties so the case ends; allowed only for the offences listed in section 345 CrPC.',
   ur: 'فریقین کے درمیان صلح سے مقدمہ ختم کرنا؛ صرف دفعہ 345 میں درج جرائم میں جائز ہے۔'},
  {id: 'khula', ci: ['khula'], urp: ['خلع'], law: 'fca:10', find: 'khula',
   en: 'Dissolution of the marriage at the wife’s request, decreed by the Family Court.',
   ur: 'بیوی کی درخواست پر فیملی کورٹ کے ذریعے نکاح کی تنسیخ۔'},
  {id: 'maintenance', ci: ['maintenance', 'nafqa', 'nan nafqa'], urp: ['نان نفقہ'], law: 'mflo:9',
   en: 'Money a husband must pay to support his wife, and a father his children.',
   ur: 'وہ خرچہ جو شوہر بیوی کے، اور والد بچوں کے، گزارے کے لیے دینے کا پابند ہے۔'},
  {id: 'custody', ci: ['custody of (?:the |a )?minors?', 'custody of (?:the |a )?child(?:ren)?', 'hizanat'], urp: ['حضانت'], law: 'gwa:25',
   en: 'The right to keep and bring up a child. The Guardian Court decides it by what is best for the child.',
   ur: 'بچے کو اپنے پاس رکھنے اور پرورش کا حق۔ گارڈین کورٹ بچے کی بہتری دیکھ کر فیصلہ کرتی ہے۔'},
  {id: 'decree', ci: ['decree'], law: 'cpc:2', find: 'decree” means',
   en: 'The formal order that records a civil court’s final decision.',
   ur: 'دیوانی عدالت کے حتمی فیصلے کا باضابطہ حکم۔'},
  {id: 'vakalat', ci: ['vakalatnama', 'wakalatnama', 'power of attorney for the advocate'], urp: ['وکالت نامہ'],
   en: 'The signed paper that authorises an advocate to act for you in court.',
   ur: 'وہ دستخط شدہ کاغذ جس سے آپ وکیل کو عدالت میں اپنی نمائندگی کا اختیار دیتے ہیں۔'},
  {id: 'causelist', ci: ['cause lists?'],
   en: 'The court’s list of the cases it will hear on a given day.',
   ur: 'عدالت کی فہرست جس میں کسی دن سنے جانے والے مقدمات درج ہوتے ہیں۔'},
  {id: 'exparte', ci: ['ex[ -]parte'],
   en: 'Heard or decided in the absence of one side, usually because they did not appear.',
   ur: 'ایک فریق کی غیر حاضری میں سماعت یا فیصلہ۔'},
  {id: 'acquittal', ci: ['acquittal', 'acquitted'],
   en: 'A court’s finding that the accused is not guilty.',
   ur: 'عدالت کا فیصلہ کہ ملزم بے قصور ہے۔'},
  {id: 'mensrea', ci: ['mens rea'],
   en: 'The guilty intention most crimes require, such as dishonesty.',
   ur: 'مجرمانہ نیت، جیسے بددیانتی، جو اکثر جرائم کے لیے ضروری ہے۔'},
  {id: 'limitation', ci: ['period of limitation', 'barred by limitation', 'time-barred', 'time barred'],
   en: 'The time limit the law sets for filing a case. A case filed late can be dismissed for that alone.',
   ur: 'مقدمہ دائر کرنے کی قانونی مدت۔ دیر سے دائر مقدمہ صرف اسی وجہ سے خارج ہو سکتا ہے۔'},
];
let GLOSS_RES = null;
function glossRes() {
  if (GLOSS_RES) return GLOSS_RES;
  const L = '(?<![A-Za-z])', R = '(?![A-Za-z])';
  GLOSS_RES = GLOSSARY.map(g => {
    const parts = [];
    if (g.ci) parts.push(new RegExp(L + '(?:' + g.ci.join('|') + ')' + R, 'gi'));
    if (g.cs) parts.push(new RegExp(L + '(?:' + g.cs.join('|') + ')' + R, 'g'));
    if (g.urp) parts.push(new RegExp('(?:' + g.urp.join('|') + ')', 'g'));
    return {g, parts};
  });
  return GLOSS_RES;
}
// The earliest legal word in text not yet used: {id, index, length}. A longer match wins a tie.
function glossFind(text, used) {
  let best = null;
  glossRes().forEach(({g, parts}) => {
    if (used.has(g.id)) return;
    parts.forEach(re => {
      re.lastIndex = 0;
      const m = re.exec(text);
      if (m && (!best || m.index < best.index || (m.index === best.index && m[0].length > best.length))) best = {id: g.id, index: m.index, length: m[0].length};
    });
  });
  return best;
}
function glossMatches(text) {
  const used = new Set(), out = [];
  let rest = String(text || ''), offset = 0, f;
  while ((f = glossFind(rest, used))) {
    out.push({id: f.id, index: offset + f.index, text: rest.substr(f.index, f.length)});
    used.add(f.id);
    offset += f.index + f.length; rest = rest.slice(f.index + f.length);
  }
  return out;
}
// Mark the first time each legal word appears in an answer or a judgment, leaving citations and buttons alone.
function glossify(root) {
  if (!root) return;
  const used = new Set();
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {acceptNode: n =>
    n.parentElement.closest('button, a, code, sup, .cite, .gloss, .statref, .citeref, table thead') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT});
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach(node => {
    let cur = node, f;
    while (cur && (f = glossFind(cur.nodeValue, used))) {
      used.add(f.id);
      const word = cur.splitText(f.index), after = word.splitText(f.length);
      const b = el('button', 'gloss');
      b.type = 'button'; b.dataset.g = f.id; b.textContent = word.nodeValue;
      b.title = 'What this means';
      word.replaceWith(b);
      cur = after;
    }
  });
}
function glossLaw(g) {
  const v = g.law && STATUTES[g.law];
  if (!v) return '';
  let t = v.text, at = g.find ? t.toLowerCase().indexOf(g.find.toLowerCase()) : 0;
  if (at < 0) at = 0;
  t = t.slice(Math.max(0, at - (g.find ? 20 : 0)));
  return (at > 20 ? '…' : '') + (t.length > 260 ? t.slice(0, 257).replace(/\s+\S*$/, '') + '…' : t);
}
function openGloss(btn) {
  closeGloss();
  const g = GLOSSARY.find(x => x.id === btn.dataset.g);
  if (!g) return;
  const pop = el('div', 'gpop');
  pop.setAttribute('role', 'dialog');
  pop.innerHTML = '<b class="gw"></b><p class="gen"></p><p class="gur" dir="rtl" lang="ur"></p>' +
    (glossLaw(g) ? '<blockquote></blockquote><button type="button" class="statref"></button>' : '') +
    '<small>Plain meaning written by the Wakeel team. General information, not legal advice.</small>';
  pop.querySelector('.gw').textContent = btn.textContent;
  pop.querySelector('.gen').textContent = g.en;
  pop.querySelector('.gur').textContent = g.ur;
  if (glossLaw(g)) {
    pop.querySelector('blockquote').textContent = glossLaw(g);
    const s = pop.querySelector('.statref'); s.dataset.stat = g.law; s.textContent = statuteLabel(g.law, STATUTES[g.law]) + ' →';
  }
  document.body.appendChild(pop);
  const r = btn.getBoundingClientRect(), w = Math.min(360, window.innerWidth - 24);
  pop.style.width = w + 'px';
  pop.style.left = Math.max(12, Math.min(window.innerWidth - w - 12, r.left)) + 'px';
  const below = r.bottom + 8, h = pop.offsetHeight;
  pop.style.top = (below + h > window.innerHeight - 8 && r.top - h - 8 > 0 ? r.top - h - 8 : below) + 'px';
}
function closeGloss() { document.querySelectorAll('.gpop').forEach(p => p.remove()); }
document.addEventListener('click', e => {
  const b = e.target.closest('.gloss');
  if (b) { e.preventDefault(); openGloss(b); return; }
  if (!e.target.closest('.gpop')) closeGloss();
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeGloss(); });
window.addEventListener('scroll', closeGloss, {passive: true});
document.addEventListener('scroll', e => { if (!e.target.closest || !e.target.closest('.gpop')) closeGloss(); }, true);
function renderResult(container, data, key, query, opts) {
  opts = opts || {};
  if (!data.language) data.language = segValue('language');   // saved with the answer, so Listen knows its language later
  const sources = data.sources || [];
  const isError = data.route === 'ERROR';
  results[key] = {sources, query};
  const ans = el('div', 'md' + (data.route === 'SQ' ? ' numbered' : '') + (isError ? ' error-text' : ''));
  ans.innerHTML = renderMarkdown(data.answer || 'No answer produced.', key, sources);
  markSeverity(ans);
  if (!isError) glossify(ans);
  container.appendChild(ans);
  if (isError) return;

  const known = new Set(sources.map(s => s.n));
  const cited = new Set([...citedNumbers(data.answer)].filter(n => known.has(n)));
  const docs = new Set(sources.map(s => s.doc_id)).size;
  const citedDocs = new Set(sources.filter(s => cited.has(s.n)).map(s => s.doc_id)).size;
  const g = el('div', 'ground');
  if (sources.length) {
    g.innerHTML = !data.grounded
      ? '<span class="bad">Not found in the indexed judgments</span><span>' + plural(docs, 'judgment') + ' searched</span>'
      : citedDocs
        ? '<span><b>Grounded</b></span><span>Cites ' + plural(cited.size, 'passage') + ' from ' + plural(citedDocs, 'judgment') +
          '</span>' + (opts.whole ? '' : '<span>' + plural(docs, 'judgment') + ' searched</span>')
        : '<span><b>Written from ' + plural(docs, 'judgment') + '</b></span>';
  }
  const copy = el('button', 'copy', 'Copy');
  copy.type = 'button';
  copy.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(ans.innerText || data.answer || ''); copy.textContent = 'Copied'; }
    catch (e) { copy.textContent = 'Copy failed'; }
    setTimeout(() => { copy.textContent = 'Copy'; }, 1500);
  });
  const facts = data.grounded ? checkFacts(data.answer, sources, opts.userText) : {checked: 0, missing: []};
  if (facts.checked) {
    const f = el('span');
    f.innerHTML = facts.missing.length
      ? '<span class="warn">' + plural(facts.missing.length, 'detail') + ' to check</span>'
      : '<span><b>Facts checked</b> ' + facts.checked + ' of ' + facts.checked + '</span>';
    f.title = 'Every name, section, number and quotation in a cited sentence is looked for in the passage it cites. No AI.';
    g.appendChild(f);
  }
  g.appendChild(copy);
  const listen = listenButton(data);
  if (listen) g.appendChild(listen);
  if (!opts.noSave) g.appendChild(saveButton(() => ({kind: pageKind(), title: opts.print || query || 'Answer', query, data})));
  if (!opts.noSave) g.appendChild(feedbackButtons(data, facts));
  if (!opts.noSave && (pageKind() === 'ask' || data.document_text)) {
    const c = el('button', 'copy', 'Open as a case');
    c.type = 'button'; c.style.marginLeft = '0';
    c.title = 'Hand this to the agent room: an orchestrator and a team of agents';
    c.addEventListener('click', () => data.document_text ? openCase('', data.document_text) : openCase(query || data.restated_query || ''));
    g.appendChild(c);
  }
  const cal = calendarButton(data, opts.print || query || 'Wakeel AI');
  if (cal) g.appendChild(cal);
  if (!opts.noVideo && !['TABLE', 'CHECK', 'CITATOR', 'EVIDENCE', 'MOOT', 'ERROR', 'REJECT'].includes(data.route) && videoPoints(data.answer, 4).length >= 2) {
    const v = el('button', 'copy vidbtn', segValue('mode') === 'lawyer' ? 'Make a video brief' : 'Make a video');
    v.type = 'button'; v.style.marginLeft = '0';
    v.title = 'A 30-second video of this answer, with captions and a voice, to send on WhatsApp';
    v.addEventListener('click', () => makeVideo(videoScript(data, query || opts.print)));
    g.appendChild(v);
  }
  if (opts.print) {
    const pdf = el('button', 'copy', 'Save as PDF');
    pdf.type = 'button';
    pdf.style.marginLeft = '0';
    pdf.addEventListener('click', () => printResult(container, opts.print));
    g.appendChild(pdf);
  }
  container.appendChild(g);
  if (!opts.noHelp && data.route !== 'ERROR' && segValue('mode') === 'citizen') container.appendChild(helpStrip());
  if (facts.missing.length) renderFactNote(container, ans, facts.missing, sources, query);
  if (!opts.whole) renderSourceList(container, sources, key, cited);
  if (opts.reveal && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const tail = [];
    for (let n = ans.nextElementSibling; n; n = n.nextElementSibling) tail.push(n);
    revealAnswer(ans, tail);
  }
}
// The details the fact check could not find: marked in the answer and listed with their passages.
function renderFactNote(container, ans, missing, sources, query) {
  const note = el('div', 'factnote');
  note.innerHTML = '<p>Check these against the judgment. They were not found in the passage cited for them:</p><ul></ul>';
  const ul = note.querySelector('ul');
  const link = n => {
    const b = el('button', '', '[S' + n + ']');
    b.type = 'button';
    b.addEventListener('click', () => { const s = sources.find(x => x.n === n); if (s) openSource(s, query); });
    return b;
  };
  missing.forEach(m => {
    const li = el('li');
    li.appendChild(document.createTextNode('“' + m.label.replace(/^["“]|["”]$/g, '') + '”, cited to '));
    m.cited.forEach((n, i) => { if (i) li.appendChild(document.createTextNode(', ')); li.appendChild(link(n)); });
    if (m.elsewhere.length) {
      li.appendChild(document.createTextNode('; it is in '));
      m.elsewhere.slice(0, 3).forEach((n, i) => { if (i) li.appendChild(document.createTextNode(', ')); li.appendChild(link(n)); });
    }
    ul.appendChild(li);
    markText(ans, m.label.replace(/^["“]|["”]$/g, ''));
  });
  container.appendChild(note);
}
function markText(root, label) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: n => n.parentElement.closest('button, mark') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT});
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const i = n.nodeValue.indexOf(label);
    if (i < 0) continue;
    const mark = el('mark', 'unverified');
    mark.title = 'Not found in the passage cited for it';
    const rest = n.splitText(i);
    rest.splitText(label.length);
    rest.parentNode.replaceChild(mark, rest);
    mark.appendChild(rest);
    return;
  }
}

// ===========================================================================
// black panel: one passage, or one judgment
// ===========================================================================
function queryTerms(q) {
  return [...new Set((q || '').toLowerCase().match(/[a-z0-9-]{3,}/g) || [])].filter(w => !STOP.has(w)).slice(0, 12);
}
function highlight(text, terms) {
  let h = escapeHtml(text);
  terms.forEach(t => {
    const rx = new RegExp('\\b(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
    h = h.replace(rx, '<mark>$1</mark>');
  });
  return h;
}
function openPanel(html) {
  const p = $('panel');
  p.innerHTML = '<button class="close" type="button">Close</button>' + html;
  p.querySelector('.close').addEventListener('click', closePanel);
  document.body.classList.add('panel-open');
  p.scrollTop = 0;
}
function closePanel() {
  document.body.classList.remove('panel-open');
  document.querySelectorAll('.cite.on, .srclist .paras button.on').forEach(x => x.classList.remove('on'));
}
$('scrim').addEventListener('click', closePanel);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closePanel(); });

function uniq(list) {   // "Section 25" and "section 25" are the same provision
  const seen = new Set();
  return (list || []).filter(x => { const k = String(x).toLowerCase().replace(/\s+/g, ' '); return !seen.has(k) && seen.add(k); });
}
function openSource(s, query) {
  const rows = [['Case', caseNumber(s)], ['Judges', (s.judges || []).join(', ')], ['Section', s.section_type],
                ['Provisions', uniq(s.provisions).join(' · ')], ['Cases cited', uniq(s.citations).join(' · ')]]
    .filter(r => r[1]);
  const scored = s.semantic !== null && s.semantic !== undefined;
  const para = s.paragraph && s.paragraph !== '0' ? '¶' + escapeHtml(s.paragraph) : 'Head';
  openPanel(
    '<div class="eyebrow">Source ' + s.n + ' · ' + escapeHtml(s.court || '') + '</div>' +
    '<h2 class="big display">' + para + '</h2>' +
    '<h3>' + escapeHtml(caseTitle(s)) + '</h3>' +
    '<dl class="kv">' + rows.map(r => '<dt>' + r[0] + '</dt><dd>' +
      (r[0] === 'Provisions' ? linkStatutes(escapeHtml(r[1])) : r[0] === 'Cases cited' ? linkCitations(escapeHtml(r[1]))
        : escapeHtml(r[1])) + '</dd>').join('') + '</dl>' +
    (scored
      ? '<div class="why"><div><span>Meaning</span><b>' + s.semantic.toFixed(2) + '</b></div>' +
        '<div><span>Keyword</span><b>' + s.keyword.toFixed(2) + '</b></div>' +
        '<div><span>Hybrid</span><b>' + Number(s.score).toFixed(2) + '</b></div></div>'
      : '<div class="note">' + (s.score === 1 ? 'Part of the judgment you selected.'
          : 'Added for context: a neighbouring or closing paragraph of a top match.') + '</div>') +
    '<div class="eyebrow" style="margin-bottom:10px">Passage' + (query ? ' · search terms underlined' : '') + '</div>' +
    '<div class="passage" dir="auto">' + highlight(s.excerpt || '(no text)', queryTerms(query)) + '</div>');
}
document.addEventListener('click', e => {
  const b = e.target.closest('.cite, .srclist .paras button');
  if (!b) return;
  const r = results[b.dataset.key];
  const s = r && r.sources.find(x => String(x.n) === b.dataset.n);
  if (!s) return;
  document.querySelectorAll('.cite.on, .srclist .paras button.on').forEach(x => x.classList.remove('on'));
  document.querySelectorAll('[data-key="' + b.dataset.key + '"][data-n="' + b.dataset.n + '"]').forEach(x => x.classList.add('on'));
  openSource(s, r.query);
});

async function openJudgment(docId) {
  openPanel('<div class="eyebrow">Loading</div>');
  let j;
  try { j = await (await fetch('/api/judgments/' + encodeURIComponent(docId))).json(); }
  catch (e) { openPanel('<div class="eyebrow">Could not load this judgment</div>'); return; }
  if (j.error) { openPanel('<div class="eyebrow">' + escapeHtml(j.error) + '</div>'); return; }
  const rows = [['Case', caseNumber(j)], ['Court', j.court], ['Judges', (j.judges || []).join(', ')],
                ['Provisions', (j.provisions || []).join(' · ')], ['Document', j.doc_id]].filter(r => r[1]);
  await loadJudgments();
  const prec = precedentsOf(docId);
  // The corpus holds some orders in the same case, so the case number and id tell them apart.
  const title = id => { const x = (judgments || []).find(y => y.doc_id === id);
                        return x ? caseTitle(x) + ' · ' + caseNumber(x) + ' (' + id + ')' : id; };
  const precHtml = !prec.mine.length ? '' :
    '<div class="eyebrow">Precedents relied on · ' + prec.mine.length + '</div><div class="cchips">' +
    prec.mine.map(k => '<button type="button" class="citeref" data-cite="' + escapeHtml(k) + '">' + escapeHtml(CITES[k].display) +
      (Object.keys(CITES[k].refs).length > 1 ? ' <i>' + Object.keys(CITES[k].refs).length + '</i>' : '') + '</button>').join('') + '</div>' +
    (prec.related.length ? '<div class="eyebrow">Shares precedents with</div><ul class="plist">' + prec.related.slice(0, 6).map(([d, ks]) =>
      '<li><b>' + ks.length + '</b><span><button type="button" class="jlink" data-doc="' + escapeHtml(d) + '">' + escapeHtml(title(d)) +
      '</button><br><small>' + ks.map(k => escapeHtml(CITES[k].display)).join(' · ') + '</small></span></li>').join('') + '</ul>' : '');
  openPanel(
    '<div class="eyebrow">' + escapeHtml(j.court || '') + '</div>' +
    '<h2 class="big display">' + (j.passages || []).length + '</h2>' +
    '<div class="sub" style="margin:-4px 0 18px">Indexed passages</div>' +
    '<h3>' + escapeHtml(caseTitle(j)) + '</h3>' +
    '<dl class="kv">' + rows.map(r => '<dt>' + r[0] + '</dt><dd>' + escapeHtml(r[1]) + '</dd>').join('') + '</dl>' +
    '<div class="acts"><button class="btn" data-act="sum">Summarize</button><button class="btn line" data-act="cmp">Compare with…</button>' +
    '<button class="btn line" data-act="time">Timeline</button><button class="btn line" data-act="ask">Ask this judgment</button>' +
    '<button class="btn line" data-act="team">Send to the agents</button></div>' +
    precHtml +
    '<div class="eyebrow">Passages</div>' +
    '<ul class="plist">' + (j.passages || []).map(p =>
      '<li><b>' + (p.paragraph && p.paragraph !== '0' ? '¶' + escapeHtml(p.paragraph) : 'Head') + '</b><span dir="auto">' +
      escapeHtml((p.text || '').slice(0, 220)) + ((p.text || '').length > 220 ? '…' : '') + '</span></li>').join('') + '</ul>');
  glossify($('panel').querySelector('.plist:last-of-type'));
  $('panel').querySelector('[data-act="sum"]').addEventListener('click', () => { go('/summarize'); selectForSummary(docId); });
  $('panel').querySelector('[data-act="cmp"]').addEventListener('click', () => { go('/compare'); loadJudgments().then(() => { $('cmp-a').value = docId; }); });
  $('panel').querySelector('[data-act="time"]').addEventListener('click', () => timelineOf(docId, caseTitle(j)));
  $('panel').querySelector('[data-act="ask"]').addEventListener('click', () => setAskScope({doc_id: docId, title: caseTitle(j), no: caseNumber(j)}));
  $('panel').querySelector('[data-act="team"]').addEventListener('click', () => openCase('Read ' + caseTitle(j), '', [docId]));
}

// ===========================================================================
// ASK page: routed agent with a live trace
// ===========================================================================
function renderThreadIntro() {
  const t = $('thread');
  if (t.children.length) return;
  t.innerHTML =
    '<div class="eyebrow" style="margin-bottom:10px">Try</div><ul class="examples">' +
    EXAMPLES.map((e, i) => '<li><button type="button" data-ex="' + i + '"><span class="n">' + pad2(i + 1) + '</span>' +
      '<span><span class="tag">' + e[0] + '</span><span class="q" dir="auto">' + escapeHtml(e[1]) + '</span></span></button></li>').join('') +
    '</ul>' + historyHtml();
}
$('thread').addEventListener('click', e => {
  const h = e.target.closest('[data-hist]');
  if (h) { reopen(Number(h.dataset.hist)); return; }
  const b = e.target.closest('[data-ex]');
  if (!b) return;
  const ex = EXAMPLES[Number(b.dataset.ex)];
  if (ex[2]) setSeg('language', ex[2]);
  ask(ex[1]);
});

function startOver() {
  thread = newThreadId();
  $('thread').innerHTML = '';
  renderThreadIntro();
  $('input').focus();
}

function makeTurn(text) {
  const t = $('thread');
  if (t.querySelector('.examples')) {
    t.innerHTML = '<div class="ground" style="margin:0 0 28px"><span>Conversation · follow-up questions remember the earlier ones</span>' +
                  '<button class="copy" type="button" id="start-over">Start over</button></div>';
    $('start-over').addEventListener('click', startOver);
  }
  const key = 'a' + (++seq);
  const turn = el('div', 'turn');
  turn.innerHTML = '<p class="q" dir="auto"></p><div class="meta"><b class="who">Router</b><span>' +
    MODE_LABEL[segValue('mode')] + '</span><span>' + LANG_LABEL[segValue('language')] + '</span></div>' +
    '<details class="trace" open><summary>Working</summary><ol class="steps"></ol></details><div class="body"></div>';
  turn.querySelector('.q').textContent = text;
  t.appendChild(turn);
  turn.scrollIntoView({block: 'start', behavior: 'smooth'});
  const steps = turn.querySelector('.steps');
  const t0 = performance.now();
  let last = t0, pending = null, n = 0;
  const sum = turn.querySelector('summary');
  let phase = THINKING.start, pi = 0;
  sum.classList.add('thinking');
  sum.textContent = phase[0];
  const timer = setInterval(() => { pi = (pi + 1) % phase.length; sum.textContent = phase[pi]; }, 2600);
  function row(cls, k, v, ms) {
    const li = el('li', 'step ' + cls, '<span class="n"></span><span class="k"></span><span class="v" dir="auto"></span><span class="ms"></span>');
    li.querySelector('.k').textContent = k;
    li.querySelector('.v').textContent = v || '';
    li.querySelector('.ms').textContent = ms || '';
    return li;
  }
  const planned = [];
  return {
    key, text, el: turn, t0, timer, route: '', restated: '',
    setPhase(list) { phase = list; pi = 0; sum.textContent = list[0]; },
    // The router's plan: the steps it will take, shown greyed out until each one runs.
    plan(items) {
      items.forEach((it, i) => {
        const li = row('planned', it[0], it[1], 'planned');
        li.querySelector('.n').textContent = pad2(n + 1 + i);
        steps.appendChild(li); planned.push(li);
      });
    },
    pending(k, v) {
      const next = planned.shift();
      pending = next || row('pending', k, v);
      pending.className = 'step pending';
      pending.querySelector('.k').textContent = k;
      pending.querySelector('.v').textContent = v || '';
      pending.querySelector('.ms').textContent = '';
      pending.querySelector('.n').textContent = pad2(n + 1);
      if (!next) steps.appendChild(pending);
    },
    done(k, v, fail) {
      const now = performance.now();
      const li = row(fail ? 'fail' : '', k, v);
      li.querySelector('.n').textContent = pad2(++n);
      if (pending) { pending.replaceWith(li); pending = null; } else steps.appendChild(li);
      last = now;
    },
    clearPending() { if (pending) { pending.remove(); pending = null; } planned.splice(0).forEach(p => p.remove()); },
    setAgent(name) { turn.querySelector('.who').textContent = name; },
    count() { return n; },
  };
}

function onStep(turn, ev) {
  if (ev.node === 'classify') {
    turn.route = ev.route;
    turn.restated = ev.restated_query || '';
    if (ev.route === 'ERROR') { turn.done('Understood', 'The language model did not respond', true); return; }
    const changed = turn.restated && turn.restated.trim().toLowerCase() !== turn.text.trim().toLowerCase();
    turn.done('Understood', changed ? 'Searching as: ' + turn.restated : 'Searching as asked');
    turn.setAgent(AGENT[ev.route] || 'Router');
    turn.done('Routed', ev.route === 'CQ' && ev.compare_a ? 'Compare agent · ' + ev.compare_a + ' vs ' + ev.compare_b
      : ev.route === 'REJECT' ? 'Not a question the judgments can answer' : AGENT[ev.route]);
    const style = MODE_LABEL[segValue('mode')] + ' · ' + LANG_LABEL[segValue('language')];
    turn.setPhase(THINKING.searching);
    if (ev.route === 'GQ' || ev.route === 'SQ') {
      turn.plan([['Search', 'Meaning and keywords' + (stats ? ', ' + stats.passages + ' passages' : '') + ', then neighbouring and closing paragraphs'],
                 ['Write', (ev.route === 'SQ' ? 'Summary by section' : 'Answer citing each paragraph') + ' · ' + style]]);
      turn.pending('Searching', 'Meaning and keywords' + (stats ? ', ' + stats.passages + ' passages' : ''));
    } else if (ev.route === 'CQ') {
      turn.plan([['Search', 'Each side separately'], ['Write', 'Side-by-side comparison · ' + style]]);
      turn.pending('Searching', 'Both judgments');
    }
  } else if (ev.node === 'retrieve' || ev.node === 'retrieve_pair') {
    turn.done('Searched', plural(ev.passages, 'passage') + ' from ' + plural(ev.judgments, 'judgment'));
    turn.setPhase(THINKING.writing);
    turn.pending('Writing', MODE_LABEL[segValue('mode')] + ' · ' + LANG_LABEL[segValue('language')]);
  } else if (ev.node === 'answer' || ev.node === 'summarize' || ev.node === 'compare') {
    if (ev.route === 'ERROR') turn.done('Stopped', 'The language model did not respond', true);
    else turn.done('Written', {answer: 'Answer with citations', summarize: 'Summary by section', compare: 'Side-by-side comparison'}[ev.node]);
  }
}

function onResult(turn, data) {
  stopThinking(turn.el.querySelector('summary'), turn.timer);
  turn.clearPending();
  if (data.route === 'ERROR' && !turn.el.querySelector('.step.fail')) turn.done('Stopped', 'See the message below', true);
  const tr = turn.el.querySelector('.trace');
  if (turn.count()) { tr.querySelector('summary').textContent = plural(turn.count(), 'step'); tr.open = false; }
  else tr.remove();
  renderResult(turn.el.querySelector('.body'), data, turn.key, data.restated_query || turn.restated || turn.text, {reveal: true});
  if (data.route !== 'ERROR') remember(turn.text, data);
}

// ===========================================================================
// AGENT ROOM (route /case): an orchestrator reads the request, picks a team for the job and runs it. The server
// runs the model agents; the statute, precedent and verifier agents finish here from what the others hand them.
// The page draws the team as a graph: a node per agent, an edge per real hand-off of data.
// ===========================================================================
const CASE_TRIES = [
  ['A situation: bounced cheque', 'I lent Rs. 900,000 to Tariq Mehmood in January 2024. He gave me cheque No. 00771245 on Bank Alfalah, which bounced on 1 March 2024 for insufficient funds. He now refuses to pay. What can I do?'],
  ['A question', 'Can the High Court cancel pre-arrest bail that was granted on the basis of a compromise?'],
  ['Read a judgment', 'Explain the judgment in Azeem-ud-Din v. Feroze Khan'],
  ['Compare two', 'Compare Azeem-ud-Din v. Feroze Khan with Muhammad Manzoor @ Dani v. The State'],
  ['A document: an FIR', 'Explain this FIR and tell me what to do next.', 'fir'],
  ['In Urdu: custody', 'میری طلاق ہو گئی ہے اور میرا تین سال کا بیٹا اپنے والد کے پاس ہے۔ کیا میں مقدمہ چلنے کے دوران عدالت سے عارضی تحویل لے سکتی ہوں؟'],
];
const ROOM_SECTIONS = {
  orchestrator: 'Orchestrator · the brief', research: 'Research agent · what the judgments say',
  statutes: 'Statute agent · the law, word for word', precedents: 'Precedent agent · the cases relied on',
  strategy: 'Strategy agent · arguments for and against', draft: 'Drafting agent · the document',
  counter: 'Counter-argument agent · where the rule has limits', explainer: 'Explainer agent · in plain words',
  reader: 'Reader agent · the judgment in brief', timeline: 'Timeline agent · the dates, in order',
  similar: 'Similar-judgment agent · other judgments on the same points', reader_a: 'Reader A · the first judgment',
  reader_b: 'Reader B · the second judgment', comparator: 'Comparator agent · side by side', verify: 'Verifier · what was checked'};
let caseFile = null;   // a document added in the agent room: {filename, mime, data} or {filename, text}
let caseDocs = [];     // judgments chosen elsewhere (the Library): one to read, two to compare
let caseRun = null;    // the run on screen
$('case-try').innerHTML = '<span class="eyebrow">Try</span>' + CASE_TRIES.map((t, i) => '<button type="button" data-i="' + i + '">' + escapeHtml(t[0]) + '</button>').join('');
$('case-try').addEventListener('click', e => {
  const b = e.target.closest('[data-i]');
  if (!b) return;
  const t = CASE_TRIES[b.dataset.i];
  $('case-text').value = t[1];
  caseDocs = [];
  caseFile = t[2] === 'fir' ? {filename: 'sample FIR', text: TOUR_FIR} : null;
  $('case-fname').textContent = caseFile ? 'A sample FIR is attached' : '';
  $('case-text').focus();
});
attachMic($('case-mic'), $('case-text'));
// The home page's live map: it lights a judgment every few seconds and says, from the data, what it relies on.
const homeNet = makeLawNet($('home-net'), {aspect: 0.5, labels: 0, ambientMs: 90});
const roomNet = makeLawNet($('room-net'), {aspect: 0.28, cites: 200, labels: 6, ambientMs: 110});
function liveFeed(html) {
  const ol = $('live-feed'), li = el('li', '', html);
  ol.appendChild(li);
  while (ol.children.length > 5) ol.firstChild.remove();
}
homeNet.ready.then(() => {
  const js = (judgments || []).filter(j => homeNet.byId['j:' + j.doc_id] !== undefined && homeNet.adj[homeNet.byId['j:' + j.doc_id]].length > 1);
  let k = Math.floor(Math.random() * js.length);
  const step = () => {
    if (pageFromPath(location.pathname) === 'home' && js.length) {
      const j = js[k++ % js.length], mine = precedentsOf(j.doc_id).mine;
      homeNet.light(['j:' + j.doc_id], {ttl: 3800});
      const provs = [...new Set((j.provisions || []).map(provisionKey).filter(Boolean))].slice(0, 2);
      liveFeed('<b>' + escapeHtml(caseTitle(j)) + '</b> · ' + escapeHtml([mine.length ? 'cites ' + plural(mine.length, 'reported case') : '',
        provs.length ? 'applies ' + provs.join(', ') : ''].filter(Boolean).join(' · ') || plural(j.passages, 'passage')));
    }
    setTimeout(step, 2600);
  };
  step();
});
CASE_TRIES.forEach((t, i) => { if (t[2] || /^(Read|Compare)/.test(t[0])) return;
  const b = el('button'); b.type = 'button'; b.textContent = t[0]; b.dataset.i = i; $('live-try').appendChild(b); });
$('live-try').addEventListener('click', e => { const b = e.target.closest('[data-i]'); if (b) { $('live-text').value = CASE_TRIES[b.dataset.i][1]; $('live-text').dispatchEvent(new Event('input')); $('live-form').requestSubmit(); } });
$('live-text').addEventListener('input', () => { const t = $('live-text'); t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight, 220) + 'px'; });
$('live-text').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('live-form').requestSubmit(); } });
attachMic($('live-mic'), $('live-text'));
$('live-form').addEventListener('submit', e => {
  e.preventDefault();
  const text = $('live-text').value.trim();
  if (!text) { $('live-text').focus(); return; }
  go('/case');
  $('case-text').value = text; caseDocs = []; caseFile = null; $('case-fname').textContent = '';
  $('case-go').click();
});
// What it solves: each problem in the words people and lawyers use, the agent on it, and a live example one press away.
function solveRoom(i) { go('/case'); $('case-text').value = CASE_TRIES[i][1]; caseDocs = []; caseFile = null; $('case-fname').textContent = ''; $('case-go').click(); }
function solveDraft(id, values) {
  go('/draft'); selectDraft(id);
  values.forEach((v, i) => { const f = $('df' + i); if (f) f.value = v; });
  $('draft-go').click();
}
async function solveTable() {
  go('/compare');
  const b = document.querySelector('#cmp-tabs [data-tab="table"]'); if (b) b.click();
  await loadJudgments();
  rtSelected.length = 0;
  [...judgments].sort((a, b) => (b.passages || 0) - (a.passages || 0)).slice(1, 5).forEach(j => rtSelected.push(j.doc_id));
  $('rt-go').disabled = false; $('rt-go').click();
}
async function solveSummary() {
  await loadJudgments();
  const j = [...judgments].sort((a, b) => (b.passages || 0) - (a.passages || 0))[0];
  go('/summarize'); selectForSummary(j.doc_id);
  setTimeout(() => $('sum-go').click(), 300);
}
const SOLVES = {
  people: [
    {q: 'The police registered an FIR. I can’t make sense of it.', was: 'Wait for a lawyer to read it to you', agent: 'Explain agent',
     how: 'Reads it in plain words, flags red flags and deadlines, finds judgments like it.',
     go: () => { go('/explain'); docFile = null; $('doc-text').value = TOUR_FIR; $('doc-go').disabled = false; $('doc-go').click(); }},
    {q: 'My cheque bounced and he refuses to pay.', was: 'Not knowing whether it is a crime, a civil claim, or both', agent: 'Agent room',
     how: 'Six agents: the law, Section 489-F word for word, precedents, a strategy and a legal notice, checked.', go: () => solveRoom(0)},
    {q: 'I can’t afford a lawyer just to send a legal notice.', was: 'Pay a fee for a two-page letter', agent: 'Drafting agent',
     how: 'Drafts the notice from your facts, citing the judgments, ready to sign.',
     go: () => solveDraft('cheque_notice', ['Muhammad Aslam', 'Tariq Mehmood', 'Rs. 900,000', 'Cheque No. 00771245, Bank Alfalah, dated 1 March 2024',
       'Loan given in January 2024, to be repaid by February 2024', '15 days'])},
    {q: 'Is it already too late to go to court?', was: 'Find out from the court, after the deadline', agent: 'Limitation calculator', rules: true,
     how: 'The Limitation Act’s periods, worked out to the day. Rules, no AI.', go: () => { go('/calculate'); selectCalc('limitation'); }},
    {q: 'میں صرف اردو سمجھتی ہوں۔ میرا بچہ اپنے والد کے پاس ہے۔', was: 'Legal help only in English',
     agent: 'Agent room · Urdu', how: 'Ask in Urdu, by voice or text; the answer comes back in Urdu, cited, and can be read aloud.', go: () => solveRoom(5)},
    {q: 'Someone in my family was arrested. What happens now?', was: 'Panic, and whoever answers the phone', agent: 'Rights right now', rules: true,
     how: 'One tap: the rights that apply this minute, in Urdu or English, each in the law\u2019s own words, read aloud, with free helplines.',
     go: () => openRights()},
    {q: 'How is my father’s property divided?', was: 'Family disputes over who gets what', agent: 'Inheritance calculator', rules: true,
     how: 'Shares under Hanafi law for each heir, exactly. Rules, no AI.', go: () => { go('/calculate'); selectCalc('inheritance'); }},
  ],
  lawyers: [
    {q: 'Hours in the law reports for one point of law.', was: 'Searching volume by volume', agent: 'Agent room · research team',
     how: 'Research, statutes, precedents and a counter-argument agent work in parallel, then a verifier checks them.', go: () => solveRoom(1)},
    {q: 'Find the holes before the other side does.', was: 'Learning them at the hearing', agent: 'Opposing counsel agent',
     how: 'It argues against your draft: the weakest points, the judgments against you, what the bench will press, how to fix it.',
     go: () => { WS.who = 'lawyer'; WS.prefill = SAMPLE_PETITION; go('/workspace?who=lawyer'); wsShow('workflows');
                 wsStartFlow(wsFlows().find(f => f.id === 'opponent')).then(() => { if ($('wsf1')) $('wsf1').value = 'Accused (petitioner)'; $('wsf-go').click(); }); }},
    {q: 'Did the AI invent a citation in my draft?', was: 'Invented cases in a filed brief', agent: 'Brief checker', rules: true,
     how: 'Every section is checked against the official text and every case against the library. A year still to come is caught too.',
     go: () => { go('/verify'); verifyTab('law'); $('vf-sample').click(); }},
    {q: 'A sixty-page judgment, and court in an hour.', was: 'Skim it and hope', agent: 'Summary agent',
     how: 'Facts, issues, holding and relief, section by section, every point tied to its paragraph.', go: solveSummary},
    {q: 'Comparing a stack of judgments for one brief.', was: 'A spreadsheet built by hand', agent: 'Review table',
     how: 'One reviewer agent per judgment, in parallel; every cell cites its paragraph.', go: solveTable},
    {q: 'Two leading judgments: where do they really differ?', was: 'Reading both end to end, twice', agent: 'Agent room · compare team',
     how: 'Two readers in parallel, a comparator, and the precedents the two share.', go: () => solveRoom(3)},
    {q: 'The same bail application, every week.', was: 'Retyping from the last file', agent: 'Drafting agent',
     how: 'Drafts it from the facts and the judgments, in your own office style if you give it one.',
     go: () => solveDraft('bail_application', ['Court of Sessions, Lahore', 'Tariq Mehmood', 'FIR No. 214/2024 dated 3 March 2024', 'Police Station Gulberg, Lahore',
       'Section 489-F PPC', 'The dispute is a civil one over a loan; the offence is not within the prohibitory clause; the accused has no previous record'])},
    {q: 'What will the bench ask tomorrow?', was: 'Guessing the night before', agent: 'Hearing preparation',
     how: 'The questions the bench may ask, with answers drawn from the judgments.',
     go: () => solveDraft('hearing', ['Accused', 'Post-arrest bail', 'FIR under Section 489-F PPC for a bounced cheque of Rs. 900,000; the accused says the cheque was security for a loan',
       'The complainant says the cheque was issued for repayment'])},
    {q: 'Which precedents does the whole library rely on?', was: 'No way to see it', agent: 'Precedent map', rules: true,
     how: 'Every reported case the judgments cite, and which judgments share them.',
     go: () => { go('/library'); const b = document.querySelector('#lib-tabs [data-tab="map"]'); if (b) b.click(); }},
  ],
};
function renderSolve(who) {
  $('solve-tabs').innerHTML = [['people', 'For people'], ['lawyers', 'For lawyers']].map(([k, t]) =>
    '<button type="button" role="tab" data-who="' + k + '" class="' + (k === who ? 'on' : '') + '">' + t + '<b>' + SOLVES[k].length + '</b></button>').join('');
  const grid = $('solve-grid');
  grid.innerHTML = '';
  SOLVES[who].forEach((x, i) => {
    const c = el('button', 'sv');
    c.type = 'button';
    c.style.setProperty('--d', (i * 0.06) + 's');
    c.innerHTML = '<span class="sv-n"></span><p class="sv-q" dir="auto"></p><p class="sv-was">Usually: <s></s></p>' +
      '<div class="sv-a"><span class="sv-agent' + (x.rules ? ' rules' : '') + '"></span><p class="sv-how"></p></div><span class="sv-go">Solve it live →</span>';
    c.querySelector('.sv-n').textContent = String(i + 1).padStart(2, '0');
    c.querySelector('.sv-q').textContent = '“' + x.q + '”';
    c.querySelector('s').textContent = x.was;
    c.querySelector('.sv-agent').textContent = x.agent;
    c.querySelector('.sv-how').textContent = x.how;
    c.addEventListener('click', () => { window.scrollTo(0, 0); x.go(); });
    grid.appendChild(c);
  });
}
$('solve-tabs').addEventListener('click', e => { const b = e.target.closest('[data-who]'); if (b) renderSolve(b.dataset.who); });
renderSolve('people');
// ===========================================================================
// WORKSPACE: one place per matter (a lawyer's) or per case (a person's): the assistant, the vault of papers, workflows
// and the history. Matters, facts and papers are kept in this browser; the server only reads what each request sends.
// ===========================================================================
const WS = {who: 'lawyer', view: 'assistant', src: 'law', flow: null, n: 0};
const WS_TEXT = {
  lawyer: {matter: 'Matter', facts: 'Matter facts', word: 'matter', newName: 'New matter, e.g. State v. Tariq Mehmood (bail)',
           factsHint: 'Client, other side, court, what happened and when, what you want. Every workflow starts from these.',
           hello: 'What are we working on?', ask: 'Ask about the law, or about this matter’s papers. E.g. when has the High Court granted bail under 489-F?',
           flows: 'Legal work, one click each', sugg: ['What are the strongest arguments for my client?', 'Which judgments support bail in a 489-F case?',
             'What must the prosecution prove under Section 489-F PPC?', 'Summarise the key dates in these papers']},
  people: {matter: 'My case', facts: 'What happened', word: 'case', newName: 'Name it, e.g. My cheque case',
           factsHint: 'In your own words, in English or Urdu: what happened, when, who is involved, and what you want.',
           hello: 'Tell us your problem. We’ll work through it with you.', ask: 'Ask anything, in English, اردو or Roman Urdu. E.g. what are my rights if the police seize my car?',
           flows: 'What do you need?', sugg: ['What are my rights in this situation?', 'What should I do next, step by step?',
             'Do I need a lawyer for this?', 'What does this document mean for me?']},
};
// Each workflow: who it is for, what it is, and how it runs.
const WS_FLOWS = [
  {id: 'room', who: 'people', tag: 'Full plan', title: 'Tell me what to do', line: 'Six agents: the law, your rights, precedents, a plan and a letter', uses: 'What happened'},
  {id: 'explain', who: 'both', tag: 'Your papers', title: 'Explain a document', line: 'An FIR, notice, summons or order, in plain words, with deadlines and red flags', uses: 'One paper from the vault'},
  {id: 'cheque_notice', who: 'both', tag: 'Draft', uses: 'Facts'},
  {id: 'reply_to_notice', who: 'both', tag: 'Draft', uses: 'Facts'},
  {id: 'fir_application', who: 'both', tag: 'Draft', uses: 'Facts'},
  {id: 'superdari', who: 'both', tag: 'Draft', uses: 'Facts'},
  {id: 'bail_application', who: 'lawyer', tag: 'Draft', uses: 'Facts'},
  {id: 'strategy', who: 'lawyer', tag: 'Research', uses: 'Facts and the judgments'},
  {id: 'hearing', who: 'both', tag: 'Prepare', uses: 'Facts and the judgments'},
  {id: 'chronology', who: 'both', tag: 'Your papers', uses: 'Every paper in the vault'},
  {id: 'client_letter', who: 'lawyer', tag: 'Client', uses: 'Facts'},
  {id: 'moot', who: 'both', tag: 'Rehearse', title: 'Practice before the judge', line: 'A judge questions you one point at a time, by voice or text, then scores your answers', uses: 'The matter facts and the judgments'},
  {id: 'evidence', who: 'both', tag: 'Verify', title: 'Check the facts against the papers', line: 'Every fact in your draft, marked supported, contradicted or not in the papers, with the words that show it', uses: 'A draft and the vault'},
  {id: 'opponent', who: 'lawyer', tag: 'Stress-test', uses: 'Your draft and the judgments'},
  {id: 'checkbrief', who: 'both', tag: 'Verify', title: 'Check the citations in a draft', line: 'Every section against the official text, every case against the library. No AI', uses: 'A draft, or a paper from the vault'},
  {id: 'review', who: 'lawyer', tag: 'Review', title: 'Review table', line: 'Your papers and chosen judgments, one row each, every cell cited', uses: 'The vault'},
  {id: 'summarize', who: 'lawyer', tag: 'Research', title: 'Brief a judgment', line: 'Facts, issues, holding and relief, tied to paragraphs', uses: 'A library judgment'},
  {id: 'compare', who: 'lawyer', tag: 'Research', title: 'Compare two judgments', line: 'Side by side: where they agree and differ', uses: 'Two library judgments'},
  {id: 'limitation', who: 'people', tag: 'Deadline', title: 'Am I too late?', line: 'Limitation periods worked out to the day. Rules, no AI', uses: 'Dates'},
  {id: 'inheritance', who: 'people', tag: 'Family', title: 'Inheritance shares', line: 'Each heir’s share under Hanafi law. Rules, no AI', uses: 'The heirs'},
];
const WS_PEOPLE_TITLES = {cheque_notice: 'Send a legal notice for my bounced cheque', reply_to_notice: 'Reply to a legal notice I received',
  fir_application: 'Ask the police to register my FIR', superdari: 'Get my seized vehicle back', hearing: 'Prepare for my hearing',
  chronology: 'Put my papers in date order', checkbrief: 'Check the law quoted in a paper I got', moot: 'Practise what to say in court',
  evidence: 'Check my statement against my papers'};
function wsFlows() { return WS_FLOWS.filter(f => f.who === 'both' || f.who === WS.who).map(f => {
  const d = DRAFTS.find(x => x.id === f.id) || {};
  return {...f, title: (WS.who === 'people' && WS_PEOPLE_TITLES[f.id]) || f.title || d.title, line: f.line || d.line, draft: d.fields ? d : null};
}); }
function wsMatters() { return loadMatters(); }
function wsMatter() {
  let ms = wsMatters();
  let m = ms.find(x => x.id === WS.mid);
  if (!m) {
    m = ms[0];
    if (!m) { m = {id: newId(), name: WS.who === 'people' ? 'My case' : 'Untitled matter', items: [], created: Date.now(), updated: Date.now(), facts: ''}; ms.unshift(m); storeMatters(ms); }
    WS.mid = m.id;
  }
  return m;
}
function wsUpdate(fn) { const ms = wsMatters(), m = ms.find(x => x.id === WS.mid); if (!m) return; fn(m); m.updated = Date.now(); storeMatters(ms); }
const VAULT_KEY = id => 'wakeel.vault.' + id;
function wsVault() { try { return JSON.parse(localStorage.getItem(VAULT_KEY(WS.mid))) || []; } catch (e) { return []; } }
function wsStoreVault(docs) { try { localStorage.setItem(VAULT_KEY(WS.mid), JSON.stringify(docs)); return true; } catch (e) { return false; } }
// Everything run in the workspace is kept with the matter, with a record of the AI used: the agents (AI or not), the
// model, the facts checked, and later whether a person reviewed it. That record is the matter's AI disclosure.
function wsSave(item) {
  const d = item.data || {}, f = d.answer && (d.sources || []).length ? checkFacts(d.answer, d.sources, '') : null;
  const agents = item.agents || d._agents || [];
  item.ai = {agents, model: stats && agents.some(x => x.ai) ? (stats.provider || '') + ' ' + (stats.model || '') : '',
             checked: f ? f.checked : 0, missing: f ? f.missing.length : 0, reviewed: false};
  delete item.agents;
  addToMatter(WS.mid, null, item); wsCounts();
}
function wsCounts() {
  const m = wsMatter();
  $('ws-vn').textContent = wsVault().length || '';
  $('ws-hn').textContent = (m.items || []).length || '';
  $('ws-wn').textContent = wsFlows().length + (WS.who === 'lawyer' ? myFlows().length : 0);
}
function wsOpen() {
  try { WS.who = localStorage.getItem('wakeel.wswho') || WS.who; } catch (e) {}
  const q = new URLSearchParams(location.search);
  if (q.get('who') === 'people' || q.get('who') === 'lawyer') WS.who = q.get('who');
  if (q.get('m')) WS.mid = q.get('m');
  wsRender();
}
function wsRender() {
  const T = WS_TEXT[WS.who], m = wsMatter();
  setSeg('mode', WS.who === 'lawyer' ? 'lawyer' : 'citizen');
  document.querySelectorAll('#ws-who button').forEach(b => b.classList.toggle('on', b.dataset.p === WS.who));
  $('ws-matter-l').textContent = T.matter; $('ws-facts-l').textContent = T.facts;
  $('ws-new').textContent = '+ New ' + T.word;
  $('ws-facts').placeholder = T.factsHint; $('ws-facts').value = m.facts || '';
  $('ws-matter').innerHTML = wsMatters().map(x => '<option value="' + x.id + '"' + (x.id === WS.mid ? ' selected' : '') + '>' + escapeHtml(x.name) + '</option>').join('');
  document.querySelectorAll('.ws-word').forEach(x => { x.textContent = T.word; });
  $('ws-hello').textContent = T.hello; $('ws-q').placeholder = T.ask; $('ws-wh').textContent = T.flows;
  $('ws-sugg').innerHTML = WS.who === 'people' ? '<button type="button" class="ws-voice" id="ws-voice">Speak instead · بول کر پوچھیں</button>' : '';
  if ($('ws-voice')) {
    $('ws-voice').addEventListener('click', voiceOpen);
    $('ws-voice').insertAdjacentHTML('afterend', '<button type="button" class="ws-voice" id="ws-rights">Arrested or stopped? Rights now</button>');
    $('ws-rights').addEventListener('click', () => openRights());
  }
  T.sugg.forEach(t => { const b = el('button'); b.type = 'button'; b.textContent = t; b.addEventListener('click', () => { $('ws-q').value = t; $('ws-ask').requestSubmit(); }); $('ws-sugg').appendChild(b); });
  wsShow(WS.view);
  wsCounts();
}
function wsShow(view) {
  WS.view = view;
  document.querySelectorAll('#ws-nav button').forEach(b => b.classList.toggle('on', b.dataset.v === view));
  document.querySelectorAll('.ws-view').forEach(v => { v.hidden = v.dataset.v !== view; });
  if (view === 'vault') wsRenderVault();
  if (view === 'workflows') wsRenderFlows();
  if (view === 'history') wsRenderHistory();
}
$('ws-nav').addEventListener('click', e => { const b = e.target.closest('[data-v]'); if (b) wsShow(b.dataset.v); });
$('ws-who').addEventListener('click', e => {
  const b = e.target.closest('[data-p]');
  if (!b) return;
  WS.who = b.dataset.p; WS.flow = null;
  try { localStorage.setItem('wakeel.wswho', WS.who); } catch (err) {}
  $('ws-form').innerHTML = ''; $('ws-wout').innerHTML = '';
  wsRender();
});
$('ws-matter').addEventListener('change', e => { WS.mid = e.target.value; $('ws-thread').innerHTML = ''; $('ws-vout').innerHTML = ''; $('ws-wout').innerHTML = ''; wsRender(); });
$('ws-new').addEventListener('click', () => {
  const name = (prompt(WS_TEXT[WS.who].newName, '') || '').trim();
  if (!name) return;
  const ms = wsMatters(), m = {id: newId(), name, items: [], created: Date.now(), updated: Date.now(), facts: ''};
  ms.unshift(m); storeMatters(ms); WS.mid = m.id;
  $('ws-thread').innerHTML = ''; $('ws-vout').innerHTML = ''; $('ws-wout').innerHTML = '';
  wsRender(); $('ws-facts').focus();
});
let wsFactsTimer = null;
$('ws-facts').addEventListener('input', () => { clearTimeout(wsFactsTimer); wsFactsTimer = setTimeout(() => wsUpdate(m => { m.facts = $('ws-facts').value; }), 400); });
attachMic($('ws-mic'), $('ws-q'));
$('ws-src').addEventListener('click', e => { const b = e.target.closest('[data-s]'); if (!b) return; WS.src = b.dataset.s;
  document.querySelectorAll('#ws-src button').forEach(x => x.classList.toggle('on', x === b)); });
$('ws-q').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('ws-ask').requestSubmit(); } });
function wsTurn(host, title, src) {
  const t = el('div', 'ws-turn');
  t.innerHTML = '<p class="q" dir="auto"></p><div class="src"></div><div class="out"></div>';
  t.querySelector('.q').textContent = title; t.querySelector('.src').textContent = src;
  host.prepend(t);
  return t.querySelector('.out');
}
// The assistant over the case law: the Ask stream, with this matter as its conversation.
async function wsAskLaw(q, out) {
  const view = pipeView(out.appendChild(el('div')));
  let data = null;
  try {
    const r = await fetch('/api/chat/stream', {method: 'POST', headers: {'Content-Type': 'application/json'},
      // The matter's facts go with the question, so the answer is about this case and not the law in general.
      body: JSON.stringify({message: q + ((wsMatter().facts || '').trim() ? '\n\n(' + (WS.who === 'people' ? 'My situation' : 'The facts of my matter') + ': ' +
                            wsMatter().facts.trim().slice(0, 700) + ')' : ''), thread_id: 'ws-' + WS.mid, mode: segValue('mode'), language: segValue('language')})});
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'stage' || ev.type === 'chunk') view.event(ev);
        else if (ev.type === 'result') data = ev;
      }
    }
  } catch (e) { data = {route: 'ERROR', answer: 'Could not reach the server: ' + e.message, sources: []}; }
  data = data || {route: 'ERROR', answer: 'The answer did not arrive.', sources: []};
  view.finish(data, q);
  data._agents = view.agents();
  return data;
}
$('ws-ask').addEventListener('submit', async e => {
  e.preventDefault();
  const q = $('ws-q').value.trim();
  if (!q) { $('ws-q').focus(); return; }
  $('ws-q').value = '';
  const docs = wsVault();
  const fromVault = WS.src === 'vault';
  if (fromVault && !docs.length) { wsShow('vault'); return; }
  const out = wsTurn($('ws-thread'), q, fromVault ? 'From ' + plural(docs.length, 'paper') + ' in this ' + WS_TEXT[WS.who].word : 'From the case law and statutes');
  const data = fromVault
    ? await watchedPost('/api/explain/ask', {text: docs.map(d => '=== ' + d.name + ' ===\n' + d.text).join('\n\n').slice(0, 20000), question: q,
                                              mode: segValue('mode'), language: segValue('language')}, out.appendChild(el('div')), docs.map(d => d.text).join('\n'))
    : await wsAskLaw(q, out);
  renderResult(out.appendChild(el('div')), data, 'ws' + (++seq), q, {noSave: true, whole: fromVault, reveal: true});
  if (data.route !== 'ERROR') wsSave({kind: 'ask', title: q, query: q, data});
});
// The vault: the matter's papers, read into text once and kept in this browser.
function wsRenderVault(selected) {
  const docs = wsVault(), box = $('ws-docs');
  box.innerHTML = docs.length ? '' : '<p class="ws-empty">No papers yet. Add an FIR, a notice, an order or an agreement.</p>';
  docs.forEach(d => {
    const row = el('label', 'ws-doc');
    row.innerHTML = '<input type="checkbox"><b></b><span></span><p></p>';
    row.querySelector('input').value = d.id;
    row.querySelector('input').checked = !selected || selected.includes(d.id);
    row.querySelector('b').textContent = d.name;
    row.querySelector('span').textContent = d.chars.toLocaleString() + ' characters' + (d.ocr ? ' · read from a photo' : '') + ' · ' + new Date(d.added).toLocaleDateString();
    row.querySelector('p').textContent = d.text.slice(0, 220).replace(/\s+/g, ' ');
    box.appendChild(row);
  });
  const acts = $('ws-acts');
  acts.innerHTML = docs.length ? '<button class="btn" type="button" data-a="table">Review table</button><button class="btn line" type="button" data-a="ask">Ask these papers</button>' +
    '<button class="btn line" type="button" data-a="chrono">Chronology</button><button class="btn line" type="button" data-a="explain">Explain one</button>' +
    '<button class="btn line" type="button" data-a="check">Check citations</button><button class="btn line" type="button" data-a="evidence">Check a draft against these</button>' +
    '<button class="btn line" type="button" data-a="remove">Remove</button>' : '';
  wsCounts();
}
function wsChecked() { const ids = [...document.querySelectorAll('#ws-docs input:checked')].map(i => i.value); return wsVault().filter(d => ids.includes(d.id)); }
async function wsAddFiles(files) {
  for (const f of files) {
    if (f.size > 10 * 1024 * 1024) { $('ws-docs').insertAdjacentHTML('afterbegin', '<p class="error-text">' + escapeHtml(f.name) + ' is larger than 10 MB.</p>'); continue; }
    const row = el('p', 'ws-empty'); row.textContent = 'Reading ' + f.name + '…'; $('ws-docs').prepend(row);
    const isText = /\.(txt|md)$/i.test(f.name) || f.type.startsWith('text/');
    const body = isText ? {filename: f.name, text: await f.text()}
      : {filename: f.name, mime: f.type || '', data: await new Promise(res => { const r = new FileReader(); r.onload = () => res(String(r.result).split(',')[1] || ''); r.readAsDataURL(f); })};
    let d;
    try { d = await (await fetch('/api/extract', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})).json(); }
    catch (e) { d = {route: 'ERROR', answer: e.message}; }
    if (d.route === 'ERROR') { row.className = 'error-text'; row.textContent = f.name + ': ' + d.answer; continue; }
    const docs = wsVault();
    docs.push({id: newId(), name: f.name, text: d.text, chars: d.chars, ocr: d.ocr, added: Date.now()});
    if (!wsStoreVault(docs)) { row.className = 'error-text'; row.textContent = 'This browser’s storage is full; remove a paper first.'; continue; }
    row.remove();
  }
  wsRenderVault();
}
$('ws-files').addEventListener('change', e => { wsAddFiles([...e.target.files]); e.target.value = ''; });
['dragover', 'dragenter'].forEach(t => $('ws-drop').addEventListener(t, e => { e.preventDefault(); $('ws-drop').classList.add('over'); }));
['dragleave', 'drop'].forEach(t => $('ws-drop').addEventListener(t, e => { e.preventDefault(); $('ws-drop').classList.remove('over'); }));
$('ws-drop').addEventListener('drop', e => wsAddFiles([...e.dataTransfer.files]));
const WS_COLUMNS = ['Type of document', 'Parties', 'Key dates', 'Sections cited', 'What it asks for or orders'];
$('ws-acts').addEventListener('click', async e => {
  const b = e.target.closest('[data-a]');
  if (!b) return;
  const docs = wsChecked(), out = $('ws-vout');
  if (!docs.length) { out.innerHTML = '<p class="ws-empty">Tick at least one paper.</p>'; return; }
  const style = {mode: segValue('mode'), language: segValue('language')};
  if (b.dataset.a === 'remove') { const ids = docs.map(d => d.id); wsStoreVault(wsVault().filter(d => !ids.includes(d.id))); wsRenderVault(); return; }
  if (b.dataset.a === 'ask') { WS.src = 'vault'; document.querySelectorAll('#ws-src button').forEach(x => x.classList.toggle('on', x.dataset.s === 'vault')); wsShow('assistant'); $('ws-q').focus(); return; }
  out.innerHTML = '';
  if (b.dataset.a === 'evidence') { wsShow('workflows'); wsStartFlow(wsFlows().find(x => x.id === 'evidence')); return; }
  if (b.dataset.a === 'check') {
    await Promise.all([loadJudgments(), citesReady]);
    const r = renderBriefCheck(out, docs[0].text);
    if (r.md) wsSave({kind: 'check', title: 'Citation check: ' + docs[0].name, query: '', data: {answer: r.md, route: 'CHECK', sources: [], grounded: true}});
    return;
  }
  if (b.dataset.a === 'table') {
    const target = out.appendChild(el('div'));
    const data = await watchedPost('/api/table', {doc_ids: [], columns: WS_COLUMNS, files: docs.slice(0, 8).map(d => ({filename: d.name, text: d.text})), ...style}, target);
    if (data.route === 'ERROR') { target.innerHTML = '<p class="error-text"></p>'; target.firstChild.textContent = data.answer; return; }
    renderReviewTable(target, data, 'wt' + (++seq));
    wsSave({kind: 'table', title: 'Review table: ' + plural(docs.length, 'paper'), query: '', data});
  } else if (b.dataset.a === 'chrono') {
    const target = out.appendChild(el('div'));
    const data = await watchedPost('/api/workflow', {kind: 'chronology', facts: {'Documents': docs.map(d => '=== ' + d.name + ' ===\n' + d.text).join('\n\n').slice(0, 18000)}, ...style}, target);
    renderResult(target, data, 'wc' + (++seq), 'Chronology', {noSave: true, whole: true, reveal: true});
    if (data.route !== 'ERROR') wsSave({kind: 'draft', title: 'Chronology of ' + plural(docs.length, 'paper'), query: '', data});
  } else if (b.dataset.a === 'explain') {
    const d = docs[0], target = out.appendChild(el('div'));
    const data = await watchedPost('/api/explain', {text: d.text, filename: d.name, ...style}, target, d.text);
    renderResult(target, data, 'we' + (++seq), d.name, {noSave: true, whole: true, reveal: true});
    if (data.route !== 'ERROR') wsSave({kind: 'explain', title: 'Explained: ' + d.name, query: '', data});
  }
});
// Workflows: a gallery of legal tasks, each a form that starts from the matter's facts.
const MYFLOWS_KEY = 'wakeel.myflows';
function myFlows() { try { return JSON.parse(localStorage.getItem(MYFLOWS_KEY)) || []; } catch (e) { return []; } }
function storeMyFlows(fs) { try { localStorage.setItem(MYFLOWS_KEY, JSON.stringify(fs)); } catch (e) {} }
function wsRenderFlows() {
  const grid = $('ws-flows');
  grid.innerHTML = '';
  if (WS.who === 'lawyer') {
    const add = el('button', 'ws-flow ws-own-new');
    add.type = 'button';
    add.innerHTML = '<span class="tag">Your own</span><b>+ Build a workflow</b><span>Write the instructions once, e.g. your standard bail grounds, and run them on any matter</span><i>Saved in this browser</i>';
    add.addEventListener('click', () => wsBuildFlow());
    grid.appendChild(add);
    myFlows().forEach(m => {
      const c = el('button', 'ws-flow ws-own' + (WS.flow === m.id ? ' on' : ''));
      c.type = 'button';
      c.innerHTML = '<span class="tag">Yours</span><b></b><span></span><i>Starts from: your instructions, the facts and the judgments</i>';
      c.querySelector('b').textContent = m.name; c.querySelector('span:not(.tag)').textContent = m.instructions.slice(0, 110) + (m.instructions.length > 110 ? '\u2026' : '');
      c.addEventListener('click', () => wsRunOwn(m));
      grid.appendChild(c);
    });
  }
  wsFlows().forEach(f => {
    const c = el('button', 'ws-flow' + (WS.flow === f.id ? ' on' : ''));
    c.type = 'button';
    c.innerHTML = '<span class="tag"></span><b></b><span></span><i></i>';
    c.querySelector('.tag').textContent = f.tag;
    c.querySelector('b').textContent = f.title;
    c.querySelector('span:not(.tag)').textContent = f.line || '';
    c.querySelector('i').textContent = 'Starts from: ' + f.uses;
    c.addEventListener('click', () => wsStartFlow(f));
    grid.appendChild(c);
  });
}
function wsBuildFlow(existing) {
  const form = $('ws-form');
  $('ws-wout').innerHTML = '';
  form.innerHTML = '<div class="ws-formbox"><h3>' + (existing ? 'Edit your workflow' : 'Build your own workflow') + '</h3>' +
    '<p class="ws-empty">Say what to produce and how to lay it out. It will research the judgments, cite them, and use the matter\u2019s facts. It will not invent law, whatever the instructions say.</p>' +
    '<label class="eyebrow" for="wso-name">Name</label><input class="field" id="wso-name" placeholder="e.g. My 489-F bail grounds">' +
    '<label class="eyebrow" for="wso-ins">Instructions</label><textarea class="field" id="wso-ins" rows="7" placeholder="e.g. Write numbered grounds for post-arrest bail in a 489-F case: whether the offence is in the prohibitory clause, whether the cheque was security, delay in the FIR, and the accused\u2019s record. Cite a judgment for each ground. End with a one-paragraph prayer."></textarea>' +
    '<div class="ctas"><button class="btn" type="button" id="wso-save">Save it</button>' + (existing ? '<button class="btn line" type="button" id="wso-del">Delete</button>' : '') + '</div></div>';
  if (existing) { $('wso-name').value = existing.name; $('wso-ins').value = existing.instructions; }
  $('wso-save').addEventListener('click', () => {
    const name = $('wso-name').value.trim(), instructions = $('wso-ins').value.trim();
    if (!name || !instructions) { (name ? $('wso-ins') : $('wso-name')).focus(); return; }
    const fs = myFlows().filter(x => !existing || x.id !== existing.id);
    const m = {id: existing ? existing.id : 'own-' + newId(), name: name.slice(0, 80), instructions: instructions.slice(0, 4000)};
    fs.unshift(m); storeMyFlows(fs);
    WS.flow = m.id; wsRenderFlows(); wsRunOwn(m);
  });
  if (existing) $('wso-del').addEventListener('click', () => { storeMyFlows(myFlows().filter(x => x.id !== existing.id)); WS.flow = null; form.innerHTML = ''; wsRenderFlows(); });
  form.scrollIntoView({behavior: 'smooth', block: 'start'});
}
function wsRunOwn(m) {
  WS.flow = m.id; wsRenderFlows();
  const form = $('ws-form'), out = $('ws-wout'), facts = (wsMatter().facts || '').trim();
  out.innerHTML = '';
  form.innerHTML = '<div class="ws-formbox"><h3></h3><p class="ws-empty"></p><label class="eyebrow" for="wso-run-ins">Instructions</label>' +
    '<textarea class="field" id="wso-run-ins" rows="5"></textarea><label class="eyebrow" for="wso-run-facts">Facts</label>' +
    '<textarea class="field" id="wso-run-facts" rows="5" dir="auto" placeholder="The facts of this matter"></textarea>' +
    '<div class="ctas"><button class="btn" type="button" id="wso-run">Run</button><button class="btn line" type="button" id="wso-edit">Edit or delete</button></div></div>';
  form.querySelector('h3').textContent = m.name;
  form.querySelector('p').textContent = 'Your workflow' + (facts ? ' \u00b7 filled from the matter facts' : '');
  $('wso-run-ins').value = m.instructions; $('wso-run-facts').value = facts;
  $('wso-edit').addEventListener('click', () => wsBuildFlow(m));
  $('wso-run').addEventListener('click', async () => {
    const vals = {'Instructions': $('wso-run-ins').value.trim(), 'Facts': $('wso-run-facts').value.trim()};
    if (!vals.Instructions) { $('wso-run-ins').focus(); return; }
    out.innerHTML = '';
    const target = out.appendChild(el('div'));
    const data = await watchedPost('/api/workflow', {kind: 'custom', facts: vals, mode: segValue('mode'), language: segValue('language')}, target, vals.Facts);
    renderResult(target, data, 'wo' + (++seq), m.name, {noSave: true, whole: false, reveal: true, print: m.name, userText: vals.Facts});
    const md = target.querySelector('.md');
    if (md && data.route !== 'ERROR') { md.contentEditable = 'true'; md.title = 'Click to edit'; md.classList.add('editable'); }
    if (data.route !== 'ERROR') wsSave({kind: 'draft', title: m.name, query: '', data});
    target.scrollIntoView({behavior: 'smooth', block: 'start'});
  });
  form.scrollIntoView({behavior: 'smooth', block: 'start'});
}
async function wsStartFlow(f) {
  WS.flow = f.id;
  wsRenderFlows();
  const form = $('ws-form'), out = $('ws-wout'), m = wsMatter(), facts = (m.facts || '').trim(), docs = wsVault();
  out.innerHTML = '';
  if (f.id === 'room') { go('/case'); $('case-text').value = facts || ''; if (facts) $('case-go').click(); else $('case-text').focus(); return; }
  if (f.id === 'limitation' || f.id === 'inheritance') { go('/calculate'); selectCalc(f.id); return; }
  if (f.id === 'moot') { wsMootSetup(f); return; }
  if (f.id === 'evidence') { wsEvidenceSetup(f); return; }
  if (f.id === 'checkbrief') {
    await Promise.all([loadJudgments(), citesReady]);
    form.innerHTML = '<div class="ws-formbox"><h3></h3><p class="ws-empty"></p>' +
      (docs.length ? '<label class="eyebrow" for="wsb-doc">A paper from the vault</label><select class="field" id="wsb-doc"><option value="">Paste instead</option>' +
        docs.map(d => '<option value="' + d.id + '">' + escapeHtml(d.name) + '</option>').join('') + '</select>' : '') +
      '<label class="eyebrow" for="wsb-text">The draft</label><textarea class="field" id="wsb-text" rows="9" dir="auto" placeholder="Paste a petition, a notice or any draft"></textarea>' +
      '<div class="ctas"><button class="btn" type="button" id="wsb-go">Check</button><button class="btn line" type="button" id="wsb-sample">Try a sample petition</button></div></div>';
    form.querySelector('h3').textContent = f.title; form.querySelector('p').textContent = f.line;
    if ($('wsb-doc')) $('wsb-doc').addEventListener('change', e => { const d = docs.find(x => x.id === e.target.value); $('wsb-text').value = d ? d.text : ''; });
    $('wsb-sample').addEventListener('click', () => { $('wsb-text').value = SAMPLE_PETITION; $('wsb-go').click(); });
    $('wsb-go').addEventListener('click', () => {
      const text = $('wsb-text').value.trim();
      if (!text) { $('wsb-text').focus(); return; }
      out.innerHTML = '';
      const r = renderBriefCheck(out, text);
      const next = el('div', 'ctas'); next.innerHTML = '<button class="btn" type="button">Now argue against it: opposing counsel</button>';
      next.firstChild.addEventListener('click', () => { WS.prefill = text; wsStartFlow(wsFlows().find(x => x.id === 'opponent')); });
      if (WS.who === 'lawyer') out.appendChild(next);
      if (wsVault().length) {
        const ev = el('button', 'btn line', 'Now check its facts against the papers'); ev.type = 'button';
        ev.addEventListener('click', () => { WS.prefill = text; wsStartFlow(wsFlows().find(x => x.id === 'evidence')); });
        (WS.who === 'lawyer' ? next : out.appendChild(el('div', 'ctas'))).appendChild(ev);
      }
      if (r.md) wsSave({kind: 'check', agents: [{name: 'Brief checker', ai: false}], title: 'Citation check' + ($('wsb-doc') && $('wsb-doc').value ? ': ' + $('wsb-doc').selectedOptions[0].textContent : ''),
                        query: '', data: {answer: r.md, route: 'CHECK', sources: [], grounded: true}});
      out.scrollIntoView({behavior: 'smooth', block: 'start'});
    });
    form.scrollIntoView({behavior: 'smooth', block: 'start'});
    return WS.autoSample ? (WS.autoSample = false, $('wsb-sample').click()) : undefined;
  }
  if (f.id === 'review' || f.id === 'explain') {
    if (!docs.length) { form.innerHTML = '<div class="ws-formbox"><h3></h3><p class="ws-empty">Add the papers to the vault first.</p><button class="btn" type="button">Open the vault</button></div>';
      form.querySelector('h3').textContent = f.title; form.querySelector('button').addEventListener('click', () => wsShow('vault')); return; }
    wsShow('vault');
    const b = document.querySelector('#ws-acts [data-a="' + (f.id === 'review' ? 'table' : 'explain') + '"]'); if (b) b.click();
    return;
  }
  if (f.id === 'summarize' || f.id === 'compare') {
    await loadJudgments();
    const opts = '<option value="">Choose a judgment</option>' + judgments.map(j => '<option value="' + escapeHtml(j.doc_id) + '">' + escapeHtml(caseTitle(j) + ' — ' + caseNumber(j)) + '</option>').join('');
    form.innerHTML = '<div class="ws-formbox"><h3></h3><p class="ws-empty"></p><select class="field" id="wsj-a">' + opts + '</select>' +
      (f.id === 'compare' ? '<select class="field" id="wsj-b">' + opts + '</select>' : '') + '<button class="btn" type="button" id="wsj-go">Run</button></div>';
    form.querySelector('h3').textContent = f.title; form.querySelector('p').textContent = f.line;
    $('wsj-go').addEventListener('click', async () => {
      const a = $('wsj-a').value, b2 = f.id === 'compare' ? $('wsj-b').value : '';
      if (!a || (f.id === 'compare' && !b2)) return;
      out.innerHTML = '';
      const target = out.appendChild(el('div'));
      const data = await watchedPost(f.id === 'compare' ? '/api/compare' : '/api/summarize', f.id === 'compare' ? {doc_a: a, doc_b: b2, mode: segValue('mode'), language: segValue('language')}
                                     : {doc_id: a, mode: segValue('mode'), language: segValue('language')}, target);
      renderResult(target, data, 'wj' + (++seq), f.title, {noSave: true, whole: true, reveal: true});
      if (data.route !== 'ERROR') wsSave({kind: f.id, title: f.title + ': ' + $('wsj-a').selectedOptions[0].textContent, query: '', data});
    });
    return;
  }
  const d = f.draft;
  if (!d) return;
  form.innerHTML = '<div class="ws-formbox"><h3></h3><p class="ws-empty"></p><div class="wsf"></div><button class="btn" type="button" id="wsf-go"></button></div>';
  form.querySelector('h3').textContent = f.title; form.querySelector('p').textContent = f.line + (facts ? ' · filled from the ' + WS_TEXT[WS.who].facts.toLowerCase() : '');
  const box = form.querySelector('.wsf');
  let filled = false;
  d.fields.forEach((fd, i) => {
    box.insertAdjacentHTML('beforeend', '<label class="eyebrow" for="wsf' + i + '"></label>' + (fd[1] ? '<textarea class="field" id="wsf' + i + '" rows="4" dir="auto"></textarea>' : '<input class="field" id="wsf' + i + '" dir="auto">'));
    box.querySelector('label[for="wsf' + i + '"]').textContent = fd[0];
    if (fd[1] && !filled) {
      const v = WS.prefill ? WS.prefill : f.id === 'chronology' && docs.length ? docs.map(x => '=== ' + x.name + ' ===\n' + x.text).join('\n\n').slice(0, 18000) : facts;
      WS.prefill = null;
      if (v) { $('wsf' + i).value = v; filled = true; }
    }
  });
  $('wsf-go').textContent = d.endpoint ? 'Run' : 'Draft';
  $('wsf-go').addEventListener('click', async () => {
    const vals = {};
    d.fields.forEach((fd, i) => { const v = $('wsf' + i).value.trim(); if (v) vals[fd[0]] = v; });
    if (!Object.keys(vals).length) { $('wsf0').focus(); return; }
    out.innerHTML = '';
    const target = out.appendChild(el('div'));
    const data = await watchedPost(d.endpoint || '/api/draft', {kind: d.id, facts: vals, mode: segValue('mode'), language: segValue('language')}, target, Object.values(vals).join('\n'));
    renderResult(target, data, 'wf' + (++seq), f.title, {noSave: true, whole: false, reveal: true, print: f.title, userText: Object.values(vals).join('\n')});
    const md = target.querySelector('.md');
    if (md && data.route !== 'ERROR' && !d.endpoint) { md.contentEditable = 'true'; md.title = 'Click to edit the draft'; md.classList.add('editable'); }
    if (data.route !== 'ERROR') wsSave({kind: 'draft', title: f.title, query: '', data});
    target.scrollIntoView({behavior: 'smooth', block: 'start'});
  });
  form.scrollIntoView({behavior: 'smooth', block: 'start'});
}
// ---- Practice before the judge (a moot court): one question a turn, answered by voice or text, then a scored review.
function wsMootSetup(f) {
  const form = $('ws-form'), out = $('ws-wout'), m = wsMatter();
  out.innerHTML = '';
  form.innerHTML = '<div class="ws-formbox"><h3></h3><p class="ws-empty"></p>' +
    '<label class="eyebrow" for="mt-hearing">The hearing</label><input class="field" id="mt-hearing" value="a post-arrest bail application">' +
    '<label class="eyebrow" for="mt-side">Your side</label><input class="field" id="mt-side" value="' + (WS.who === 'lawyer' ? 'accused (petitioner)' : 'myself') + '">' +
    '<label class="eyebrow" for="mt-facts">The facts</label><textarea class="field" id="mt-facts" rows="5" dir="auto"></textarea>' +
    '<label class="mt-speak"><input type="checkbox" id="mt-voice" checked> The judge speaks aloud</label>' +
    '<div class="ctas"><button class="btn" type="button" id="mt-go">Enter the courtroom</button></div></div>';
  form.querySelector('h3').textContent = f.title; form.querySelector('p').textContent = f.line;
  $('mt-facts').value = WS.prefill || m.facts || ''; WS.prefill = null;
  $('mt-go').addEventListener('click', () => {
    const facts = $('mt-facts').value.trim();
    if (!facts) { $('mt-facts').focus(); return; }
    wsMoot({facts, side: $('mt-side').value.trim(), hearing: $('mt-hearing').value.trim(), voice: $('mt-voice').checked, title: f.title});
  });
}
function wsMoot(cfg) {
  const out = cfg.out || $('ws-wout');
  out.innerHTML = '<div class="court"><div class="court-h"><span>Court in session · <b class="court-hearing"></b></span><span class="court-n"></span></div>' +
    '<ol class="court-log"></ol><div class="court-think"></div><form class="court-ans" hidden><textarea class="field" rows="3" dir="auto" placeholder="Your answer, my lord…"></textarea>' +
    '<div class="ctas"><button class="mic" type="button" hidden></button><button class="btn line" type="button" data-a="suggest">Suggest an answer</button>' +
    '<button class="btn" type="submit">Answer</button>' +
    '<button class="btn line" type="button" data-a="end">End and get feedback</button></div></form><div class="court-review"></div></div>';
  const box = out.firstChild, log = box.querySelector('.court-log'), form = box.querySelector('.court-ans'), field = form.querySelector('textarea');
  box.querySelector('.court-hearing').textContent = cfg.hearing;
  attachMic(form.querySelector('.mic'), field);
  const history = [], agents = {}, lang = segValue('language') === 'en' ? 'en' : segValue('language') === 'ur' ? 'ur' : '';
  const say = text => {
    if (!cfg.voice || !lang || !('speechSynthesis' in window)) return;
    const v = pickVoice(speechSynthesis.getVoices(), lang);
    if (!v) return;
    speechSynthesis.cancel();
    speechChunks(speechText(text), 220).forEach(c => { const u = new SpeechSynthesisUtterance(c); u.voice = v; u.lang = v.lang; u.rate = 0.95; speechSynthesis.speak(u); });
  };
  const turn = async finish => {
    form.hidden = true;
    const think = box.querySelector('.court-think');
    think.innerHTML = '';
    const data = await watchedPost('/api/moot', {facts: cfg.facts, side: cfg.side, hearing: cfg.hearing, history, finish, mode: segValue('mode'),
                                                 language: segValue('language')}, think.appendChild(el('div')), cfg.facts);
    (data._agents || []).forEach(a => { agents[a.name] = a; });
    think.innerHTML = '';
    if (data.route === 'ERROR') { think.innerHTML = '<p class="error-text"></p>'; think.firstChild.textContent = data.answer; form.hidden = false; return; }
    if (data.done) {
      box.querySelector('.court-n').textContent = 'Hearing over';
      const rv = box.querySelector('.court-review');
      rv.innerHTML = '<div class="eyebrow">The bench’s review</div>';
      renderResult(rv.appendChild(el('div')), data, 'mt' + (++seq), cfg.hearing, {noSave: true, whole: false, noHelp: true});
      say(data.answer.split('\n').slice(0, 3).join(' '));
      const transcript = history.map(t => (t.role === 'judge' ? '**The court:** ' : '**Counsel:** ') + t.text).join('\n\n');
      if (cfg.save !== false) wsSave({kind: 'moot', title: 'Practice hearing: ' + cfg.hearing, query: '', agents: Object.values(agents),
              data: {...data, answer: '## The hearing\n\n' + transcript + '\n\n' + data.answer}});
      return;
    }
    history.push({role: 'judge', text: data.answer});
    box.querySelector('.court-n').textContent = 'Question ' + data.question + ' of ' + data.of;
    const li = el('li', 'court-judge');
    li.innerHTML = '<span class="who">The court</span><div></div>';
    renderResult(li.lastChild, data, 'mj' + (++seq), cfg.hearing, {noSave: true, whole: true, noHelp: true});
    li.querySelectorAll('.tools, .srclist, .factnote, .followup').forEach(x => x.remove());
    log.appendChild(li);
    say(data.answer);
    form.hidden = false; field.value = ''; field.focus();
    li.scrollIntoView({behavior: 'smooth', block: 'start'});
  };
  form.addEventListener('submit', e => {
    e.preventDefault();
    const a = field.value.trim();
    if (!a) { field.focus(); return; }
    if ('speechSynthesis' in window) speechSynthesis.cancel();
    history.push({role: 'counsel', text: a});
    const li = el('li', 'court-counsel');
    li.innerHTML = '<span class="who">Counsel</span><p dir="auto"></p>';
    li.querySelector('p').textContent = a;
    log.appendChild(li);
    turn(false);
  });
  form.querySelector('[data-a="end"]').addEventListener('click', () => { if (history.some(t => t.role === 'counsel')) turn(true); });
  // Not a lawyer, or stuck? Counsel's reply, written from the facts and the judgments, to send as it is or edit.
  const sug = form.querySelector('[data-a="suggest"]');
  sug.addEventListener('click', async () => {
    sug.disabled = true; const was = field.placeholder; field.value = ''; field.placeholder = 'Writing a suggested answer…';
    let d;
    try { d = await (await fetch('/api/moot', {method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({facts: cfg.facts, side: cfg.side, hearing: cfg.hearing, history, suggest: true, mode: segValue('mode'), language: segValue('language')})})).json(); }
    catch (e) { d = {route: 'ERROR'}; }
    sug.disabled = false; field.placeholder = was;
    if (d.route !== 'ERROR' && d.answer) { field.value = plainWords(d.answer).replace(/\s+/g, ' ').trim(); field.focus(); }
    else field.placeholder = 'No suggestion this time. Type your answer.';
  });
  out.scrollIntoView({behavior: 'smooth', block: 'start'});
  turn(false);
}
// ---- Evidence check: each fact in a draft against the papers in the vault. The quotes are checked here, in the browser.
function wsEvidenceSetup(f) {
  const form = $('ws-form'), out = $('ws-wout'), docs = wsVault();
  out.innerHTML = '';
  if (!docs.length) {
    form.innerHTML = '<div class="ws-formbox"><h3></h3><p class="ws-empty">Add the papers to the vault first: the FIR, notices, orders, receipts.</p><button class="btn" type="button">Open the vault</button></div>';
    form.querySelector('h3').textContent = f.title; form.querySelector('button').addEventListener('click', () => wsShow('vault'));
    return;
  }
  form.innerHTML = '<div class="ws-formbox"><h3></h3><p class="ws-empty"></p><label class="eyebrow" for="ev-draft">The draft</label>' +
    '<textarea class="field" id="ev-draft" rows="8" dir="auto" placeholder="Paste the petition, plaint, reply or statement"></textarea>' +
    '<div class="eyebrow">Against these papers</div><div class="ev-docs">' + docs.map(d => '<label><input type="checkbox" value="' + d.id + '" checked> <span></span></label>').join('') + '</div>' +
    '<div class="ctas"><button class="btn" type="button" id="ev-go">Check the facts</button></div></div>';
  form.querySelector('h3').textContent = f.title; form.querySelector('p').textContent = f.line;
  form.querySelectorAll('.ev-docs span').forEach((x, i) => { x.textContent = docs[i].name; });
  $('ev-draft').value = WS.prefill || ''; WS.prefill = null;
  $('ev-go').addEventListener('click', async () => {
    const draft = $('ev-draft').value.trim(), ids = [...form.querySelectorAll('.ev-docs input:checked')].map(x => x.value);
    const chosen = docs.filter(d => ids.includes(d.id));
    if (!draft) { $('ev-draft').focus(); return; }
    if (!chosen.length) return;
    out.innerHTML = '';
    const papers = chosen.map(d => '=== ' + d.name + ' ===\n' + d.text).join('\n\n').slice(0, 18000);
    const target = out.appendChild(el('div'));
    const data = await watchedPost('/api/workflow', {kind: 'evidence', facts: {'Draft': draft, 'Papers': papers}, mode: segValue('mode'),
                                                     language: segValue('language')}, target, draft);
    renderResult(target, data, 'ev' + (++seq), f.title, {noSave: true, whole: true, reveal: false, print: f.title, noHelp: true});
    markEvidence(target, papers);
    if (data.route !== 'ERROR') wsSave({kind: 'evidence', title: f.title + ': ' + plural(chosen.length, 'paper'), query: '', data});
    target.scrollIntoView({behavior: 'smooth', block: 'start'});
  });
  form.scrollIntoView({behavior: 'smooth', block: 'start'});
}
// Colour each row by its status, and check that each quote really is in the papers.
function markEvidence(root, papers) {
  const flat = t => String(t).toLowerCase().replace(/[^a-z0-9؀-ۿ]+/g, '');
  const hay = flat(papers);
  root.querySelectorAll('table tbody tr').forEach(tr => {
    const cells = tr.querySelectorAll('td');
    if (cells.length < 4) return;
    const status = cells[2].textContent.toLowerCase();
    tr.classList.add(status.includes('contradict') ? 'ev-contra' : status.includes('support') ? 'ev-ok' : 'ev-none');
    const q = cells[3].textContent.match(/["“]([^"”]{6,})["”]/);
    if (!q) return;
    const ok = hay.includes(flat(q[1]));
    cells[3].insertAdjacentHTML('beforeend', ' <span class="ev-q ' + (ok ? 'ok' : 'no') + '">' + (ok ? 'quote found in the paper' : 'quote not found: check it') + '</span>');
  });
}
// ---- The AI-use record: one line for each task in the matter, for a court or a client.
function wsAiRecord(box, m) {
  const items = m.items || [];
  const rec = el('div', 'airec');
  rec.innerHTML = '<div class="eyebrow">AI-use record · ' + escapeHtml(m.name) + '</div><h3>Where AI was used in this ' + WS_TEXT[WS.who].word + '</h3>' +
    '<p>Every task run in this ' + WS_TEXT[WS.who].word + ', the agents that did it (AI or fixed rules), the model, the checks made, and whether a person has reviewed the result. ' +
    'AI supported research and drafting only; decisions stay with people, as Pakistan’s 2026 judicial AI guidelines require.</p>' +
    '<div class="tablewrap"><table><thead><tr><th>When</th><th>Task</th><th>Agents</th><th>Model</th><th>Checks</th><th>Reviewed by a person</th></tr></thead><tbody></tbody></table></div>' +
    '<div class="ctas"><button class="btn" type="button" data-a="pdf">Export the record as PDF</button></div>';
  const tb = rec.querySelector('tbody');
  items.forEach(it => {
    const a = it.ai || {}, tr = el('tr');
    const agents = (a.agents || []).map(x => escapeHtml(x.name) + (x.ai ? ' <i>AI</i>' : ' <i class="rules">no AI</i>')).join(', ') || '—';
    tr.innerHTML = '<td></td><td></td><td>' + agents + '</td><td></td><td></td><td><label><input type="checkbox"> <span></span></label></td>';
    tr.children[0].textContent = new Date(it.saved).toLocaleString();
    tr.children[1].textContent = (KIND_LABEL[it.kind] || 'Task') + ': ' + it.title;
    tr.children[3].textContent = a.model || '—';
    tr.children[4].textContent = a.checked ? (a.checked - a.missing) + ' of ' + a.checked + ' facts found in their sources' : it.kind === 'check' ? 'No AI used' : '—';
    const cb = tr.querySelector('input'), lab = tr.querySelector('span');
    cb.checked = !!a.reviewed; lab.textContent = a.reviewed ? 'Yes, ' + new Date(a.reviewed).toLocaleDateString() : 'Not yet';
    cb.addEventListener('change', () => {
      wsUpdate(mm => { const x = (mm.items || []).find(y => y.id === it.id); if (x) { x.ai = x.ai || {}; x.ai.reviewed = cb.checked ? Date.now() : false; } });
      lab.textContent = cb.checked ? 'Yes, ' + new Date().toLocaleDateString() : 'Not yet';
    });
    tb.appendChild(tr);
  });
  rec.querySelector('[data-a="pdf"]').addEventListener('click', () => printResult(rec, 'AI-use record: ' + m.name));
  box.appendChild(rec);
}
function wsRenderHistory() {
  const m = wsMatter(), box = $('ws-hist');
  box.innerHTML = '';
  const items = [...(m.items || [])].reverse();
  if (!items.length) { box.innerHTML = '<p class="ws-empty">Nothing yet. Everything you run here is kept with this ' + WS_TEXT[WS.who].word + '.</p>'; return; }
  const tools = el('div', 'ws-acts');
  tools.innerHTML = '<button class="btn" type="button">Export all as PDF</button><button class="btn line" type="button">AI-use record</button>';
  box.appendChild(tools);
  tools.lastChild.addEventListener('click', () => { const old = box.querySelector('.airec'); if (old) { old.remove(); return; } wsAiRecord(tools.parentNode.insertBefore(el('div'), tools.nextSibling), m); });
  const list = el('div');
  items.forEach(it => {
    const x = el('div', 'ws-hitem');
    x.innerHTML = '<div class="eyebrow"></div><h3 dir="auto"></h3><div></div>';
    x.querySelector('.eyebrow').textContent = (KIND_LABEL[it.kind] || 'Saved') + ' · ' + new Date(it.saved).toLocaleString();
    x.querySelector('h3').textContent = it.title;
    const body = x.lastChild;
    if (it.kind === 'table') renderReviewTable(body, it.data, 'wh' + it.id, {noSave: true});
    else renderResult(body, it.data, 'wh' + it.id, it.query, {noSave: true, whole: it.kind !== 'ask'});
    body.querySelectorAll('.followup').forEach(y => y.remove());
    list.appendChild(x);
  });
  box.appendChild(list);
  tools.firstChild.addEventListener('click', () => printResult(list, m.name));
}

// ===========================================================================
// VOICE MODE: for people who would rather speak than type, or cannot read. Speech in, the agents at work, speech out.
// ===========================================================================
const VOICE = {lang: 'ur', rec: null, n: 0};
const VOICE_TEXT = {ur: {idle: '\u0628\u0679\u0646 \u062f\u0628\u0627\u0626\u06cc\u06ba \u0627\u0648\u0631 \u0628\u0648\u0644\u06cc\u06ba', listening: '\u0633\u0646 \u0631\u06c1\u06d2 \u06c1\u06cc\u06ba\u2026 \u0628\u0627\u062a \u062e\u062a\u0645 \u06c1\u0648\u0646\u06d2 \u067e\u0631 \u0628\u0679\u0646 \u062f\u0628\u0627\u0626\u06cc\u06ba',
                         working: '\u0627\u06cc\u062c\u0646\u0679 \u06a9\u0627\u0645 \u06a9\u0631 \u0631\u06c1\u06d2 \u06c1\u06cc\u06ba\u2026', reading: '\u062c\u0648\u0627\u0628 \u067e\u0691\u06be \u0631\u06c1\u06d2 \u06c1\u06cc\u06ba', done: '\u062f\u0648\u0628\u0627\u0631\u06c1 \u067e\u0648\u0686\u06be\u0646\u06d2 \u06a9\u06d2 \u0644\u06cc\u06d2 \u0628\u0679\u0646 \u062f\u0628\u0627\u0626\u06cc\u06ba'},
                    en: {idle: 'Press the button and speak', listening: 'Listening\u2026 press again when you finish', working: 'The agents are working\u2026',
                         reading: 'Reading the answer aloud', done: 'Press the button to ask again'}};
function voiceState(k, extra) { $('voice-state').textContent = extra || VOICE_TEXT[VOICE.lang][k]; $('voice').dataset.state = k; }
function voiceOpen() {
  $('voice').hidden = false; document.body.classList.add('voice-on');
  $('voice-mic').innerHTML = MIC_SVG;
  if (!SpeechRec) { $('voice-type').hidden = false; voiceState('idle', 'This browser cannot listen. Use Chrome or Edge, or type below.'); }
  else voiceState('idle');
}
function voiceClose() {
  $('voice').hidden = true; document.body.classList.remove('voice-on');
  if (VOICE.rec) VOICE.rec.abort();
  if ('speechSynthesis' in window) speechSynthesis.cancel();
}
function voiceSpeak(text) {
  if (!('speechSynthesis' in window)) return false;
  const voice = pickVoice(speechSynthesis.getVoices(), VOICE.lang);
  if (!voice) return false;
  speechSynthesis.cancel();
  const chunks = speechChunks(speechText(text), 220);
  chunks.forEach((c, i) => {
    const u = new SpeechSynthesisUtterance(c);
    u.voice = voice; u.lang = voice.lang; u.rate = 0.95;
    if (i === chunks.length - 1) u.onend = () => voiceState('done');
    speechSynthesis.speak(u);
  });
  return true;
}
async function voiceAsk(text) {
  voiceState('working');
  const out = $('voice-out');
  out.innerHTML = '';
  const view = pipeView(out.appendChild(el('div')));
  let data = null;
  try {
    const r = await fetch('/api/chat/stream', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, thread_id: 'voice-' + (++VOICE.n) + '-' + Date.now(), mode: 'citizen', language: VOICE.lang})});
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'stage' || ev.type === 'chunk') view.event(ev); else if (ev.type === 'result') data = ev;
      }
    }
  } catch (e) { data = {route: 'ERROR', answer: 'Could not reach the server: ' + e.message, sources: []}; }
  data = data || {route: 'ERROR', answer: 'The answer did not arrive.', sources: []};
  view.finish(data, text);
  const ans = out.appendChild(el('div', 'voice-ans'));
  renderResult(ans, data, 'v' + (++seq), text, {noSave: true, whole: true});
  if (data.route !== 'ERROR' && voiceSpeak(data.answer)) voiceState('reading');
  else voiceState('done', data.route === 'ERROR' ? '' : (VOICE.lang === 'ur' ? 'This device has no Urdu voice installed, so the answer is shown instead.' : ''));
  if ($('voice-state').textContent === '') voiceState('done');
}
$('voice-open').addEventListener('click', voiceOpen);
$('rights-open').addEventListener('click', () => openRights());
$('rights-open-2').addEventListener('click', () => openRights());
$('voice-x').addEventListener('click', voiceClose);
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('voice').hidden) voiceClose(); });
$('voice-lang').addEventListener('click', e => {
  const b = e.target.closest('[data-l]'); if (!b) return;
  VOICE.lang = b.dataset.l;
  document.querySelectorAll('#voice-lang button').forEach(x => x.classList.toggle('on', x === b));
  voiceState('idle');
});
$('voice-mic').addEventListener('click', () => {
  if (!SpeechRec) { $('voice-text').focus(); return; }
  if (VOICE.rec) { VOICE.rec.stop(); return; }
  if ('speechSynthesis' in window) speechSynthesis.cancel();
  const rec = VOICE.rec = new SpeechRec();
  rec.lang = VOICE.lang === 'ur' ? 'ur-PK' : 'en-IN';
  rec.interimResults = true;
  let heard = '';
  rec.onresult = e => { heard = ''; for (const r of e.results) heard += r[0].transcript; $('voice-heard').textContent = heard; };
  rec.onerror = e => { if (e.error !== 'aborted') voiceState('idle', MIC_ERRORS[e.error] || 'Voice input stopped. Try again, or type.'); if (e.error !== 'aborted') $('voice-type').hidden = false; };
  rec.onend = () => { VOICE.rec = null; if (heard.trim()) voiceAsk(heard.trim()); else if ($('voice').dataset.state === 'listening') voiceState('idle'); };
  $('voice-heard').textContent = ''; $('voice-out').innerHTML = '';
  voiceState('listening');
  try { rec.start(); } catch (err) { VOICE.rec = null; voiceState('idle'); }
});
$('voice-type').addEventListener('submit', e => { e.preventDefault(); const t = $('voice-text').value.trim(); if (!t) return;
  $('voice-heard').textContent = t; $('voice-text').value = ''; voiceAsk(t); });

// ===========================================================================
// DEMO AUTOPILOT: the whole story, played by the app itself, with a caption saying what each step shows.
// It presses the same buttons a person would, so with the demo cache warm nothing waits on the network.
// Space pauses, the right arrow skips a step, Esc stops. Start it from the pitch page, with F9, or ?autopilot.
// ===========================================================================
const AUTO = {on: false, paused: false, skip: false, i: 0};
const autoSleep = async ms => {
  const end = Date.now() + ms;
  while (Date.now() < end || AUTO.paused) {
    if (!AUTO.on) throw new Error('stopped');
    if (AUTO.skip) return;
    await new Promise(r => setTimeout(r, 100));
  }
};
const autoUntil = async (test, ms) => {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (!AUTO.on) throw new Error('stopped');
    if (AUTO.skip) return false;
    try { if (test()) return true; } catch (e) {}
    await new Promise(r => setTimeout(r, 150));
  }
  return false;
};
async function autoType(field, text, cps) {
  field.value = ''; field.focus();
  for (let i = 0; i < text.length; i += 2) {
    if (AUTO.skip) break;
    field.value = text.slice(0, i + 2); field.dispatchEvent(new Event('input'));
    await autoSleep(1000 / (cps || 45));
  }
  field.value = text; field.dispatchEvent(new Event('input'));
}
const autoScroll = (node, off) => { if (node) window.scrollTo({top: node.getBoundingClientRect().top + scrollY - (off || 70), behavior: 'smooth'}); };
const solveCard = (who, start) => SOLVES[who].find(x => x.q.startsWith(start));
const AUTOPILOT = [
  {cap: 'Wakeel AI: legal help built on real Lahore High Court judgments, for the people who need a lawyer and for the lawyers themselves.',
   run: async () => { setSeg('mode', 'citizen'); setSeg('language', 'en'); WS.who = 'lawyer'; go('/'); window.scrollTo(0, 0); await autoSleep(7000); }},
  {cap: 'The map is live: every dot is a real judgment, a case it cites, or a section it applies. Each answer is traced back to them.',
   run: async () => { autoScroll(document.querySelector('.netblock'), 90); await autoSleep(6000); }},
  {cap: 'What it solves: the problems people bring to a lawyer, and the hours lawyers lose. Each one has an agent on it.',
   run: async () => { autoScroll($('solve'), 20); await autoSleep(4500); renderSolve('lawyers'); await autoSleep(4500); renderSolve('people'); }},
  {cap: 'A citizen describes a problem in plain words. An orchestrator reads it and deploys a team of six agents.',
   run: async () => {
     go('/'); window.scrollTo({top: 0, behavior: 'smooth'}); await autoSleep(800);
     await autoType($('live-text'), CASE_TRIES[0][1], 60); await autoSleep(600);
     $('live-form').requestSubmit();
     await autoUntil(() => caseRun && caseRun.team && caseRun.team.length > 1, 40000);
     autoScroll($('room-stage'), 10);
     await autoUntil(() => caseRun.finished, 90000);
   }},
  {cap: 'They worked in parallel: research, statutes word for word, precedents, strategy, a draft notice, and a verifier with no AI.',
   run: async () => { autoScroll($('room-gantt'), 260); await autoSleep(6000); }},
  {cap: 'Tap any legal word for its meaning in Urdu and English, with the law behind it.',
   run: async () => {
     const g = $('case-out').querySelector('.gloss');
     if (g) { autoScroll(g, 200); await autoSleep(1200); openGloss(g); }
     await autoSleep(6000); closeGloss();
   }},
  {cap: 'For a lawyer: the brief checker finds an invented section, a repealed one and a report from 2031. No AI is used.',
   run: async () => { solveCard('lawyers', 'Did the AI').go(); await autoUntil(() => document.querySelector('.brief-check'), 20000);
                      autoScroll(document.querySelector('.brief-check'), 90); await autoSleep(8000); }},
  {cap: 'Practice before the judge: the bench questions counsel one point at a time, citing the judgments, then scores the answers.',
   run: async () => {
     go('/court'); await autoSleep(1500); $('court-voice').checked = false; $('court-sample').click();
     await autoUntil(() => document.querySelector('.court-ans:not([hidden])'), 40000);
     autoScroll(document.querySelector('.court'), 70); await autoSleep(5000);
     const f = document.querySelector('.court-ans');
     if (f) { await autoType(f.querySelector('textarea'), 'My lord, Section 489-F carries at most three years, so it is outside the prohibitory clause, and the cheque was given only as security for a loan.', 70); f.requestSubmit(); }
     await autoUntil(() => document.querySelectorAll('.court-judge').length > 1, 40000);
     autoScroll(document.querySelectorAll('.court-judge')[1], 120); await autoSleep(7000);
   }},
  {cap: 'For someone being arrested right now: their rights in Urdu, each in the Constitution’s own words, with free helplines.',
   run: async () => { openRights('ur'); await autoSleep(5000); const p = $('panel'); p.scrollTo({top: 420, behavior: 'smooth'}); await autoSleep(5000); closePanel(); }},
  {cap: 'And any answer becomes a 30-second video in Urdu, with the law and the free helplines, ready to send on WhatsApp.',
   run: async () => { makeVideo(rightsVideoScript('ur')); await autoUntil(() => !$('vid-acts').hidden, 90000); await autoSleep(5000); closeVideo(); }},
  {cap: 'Built to Pakistan’s 2026 judicial AI guidelines: people decide, every step is shown, every source is checked, nothing is stored.',
   run: async () => { go('/pitch'); await autoSleep(600); autoScroll(document.querySelector('.guide-map'), 140); await autoSleep(8000); }},
  {cap: 'Wakeel AI. Every answer traced to its paragraph.',
   run: async () => { go('/'); window.scrollTo({top: 0, behavior: 'smooth'}); await autoSleep(5000); }},
];
function autoShow() {
  const n = AUTOPILOT.length;
  $('auto-n').textContent = (AUTO.i + 1) + ' / ' + n;
  $('auto-cap').textContent = AUTOPILOT[AUTO.i].cap;
  $('auto-prog').style.width = (100 * AUTO.i / (n - 1)) + '%';
  $('auto-pause').textContent = AUTO.paused ? 'Resume' : 'Pause';
}
async function autopilot() {
  if (AUTO.on) return;
  AUTO.on = true; AUTO.paused = false; AUTO.i = 0;
  $('auto').hidden = false; document.body.classList.add('auto-on');
  try {
    for (; AUTO.i < AUTOPILOT.length; AUTO.i++) {
      AUTO.skip = false;
      autoShow();
      try { await AUTOPILOT[AUTO.i].run(); }
      catch (e) { if (e.message === 'stopped') throw e; }   // a step that fails moves on; the story goes on
      if (!AUTO.skip) await autoSleep(400);
    }
  } catch (e) { /* stopped */ }
  AUTO.on = false; $('auto').hidden = true; document.body.classList.remove('auto-on'); closeGloss();
}
function autoStop() { AUTO.on = false; }
$('auto-pause').addEventListener('click', () => { AUTO.paused = !AUTO.paused; autoShow(); });
$('auto-next').addEventListener('click', () => { AUTO.skip = true; });
$('auto-stop').addEventListener('click', autoStop);
$('pitch-auto').addEventListener('click', autopilot);
document.addEventListener('keydown', e => {
  if (e.key === 'F9') { e.preventDefault(); AUTO.on ? autoStop() : autopilot(); return; }
  if (!AUTO.on) return;
  const typing = e.target.closest && e.target.closest('input, textarea, [contenteditable="true"]');
  if (e.key === 'Escape') autoStop();
  else if (e.key === 'ArrowRight' && !typing) AUTO.skip = true;
  else if (e.key === ' ' && !typing) { e.preventDefault(); AUTO.paused = !AUTO.paused; autoShow(); }
});
if (new URLSearchParams(location.search).has('autopilot')) setTimeout(autopilot, 1200);

// ===========================================================================
// COURTROOM and VERIFY: the moot court and the checks before filing, each on its own page.
// ===========================================================================
const COURT_SAMPLE = 'Client Tariq Mehmood is accused under Section 489-F PPC (FIR 214/2024). A cheque of Rs. 900,000 was dishonoured on 1 March 2024. He says it was security for a business loan. In custody since 5 March 2024; no previous record.';
function courtOpen() { if (!$('court-facts').value.trim() && WS.mid) { const m = wsMatters().find(x => x.id === WS.mid); if (m && m.facts) $('court-facts').value = m.facts; } }
function courtStart() {
  const facts = $('court-facts').value.trim();
  if (!facts) { $('court-facts').focus(); return; }
  wsMoot({facts, side: $('court-side').value.trim(), hearing: $('court-hearing').value.trim(), voice: $('court-voice').checked,
          title: 'Practice hearing', out: $('court-out'), save: false});
}
$('court-go').addEventListener('click', courtStart);
$('court-sample').addEventListener('click', () => {
  $('court-hearing').value = 'a post-arrest bail application'; $('court-side').value = 'accused (petitioner)'; $('court-facts').value = COURT_SAMPLE;
  setSeg('mode', 'lawyer'); courtStart();
});
function verifyTab(t) {
  document.querySelectorAll('#vf-tabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === t));
  document.querySelectorAll('.vf-pane').forEach(x => { x.hidden = x.dataset.pane !== t; });
}
$('vf-tabs').addEventListener('click', e => { const b = e.target.closest('[data-tab]'); if (b) verifyTab(b.dataset.tab); });
async function verifyOpen() {
  await Promise.all([loadJudgments(), citesReady]);
  const top = Object.entries(CITES).sort((a, b) => Object.keys(b[1].refs).length - Object.keys(a[1].refs).length).slice(0, 8);
  $('vf-chips').innerHTML = '<span class="eyebrow">Most cited in the library</span>' + top.map(([k, c]) => '<button type="button" data-k="' + escapeHtml(k) + '"></button>').join('');
  $('vf-chips').querySelectorAll('[data-k]').forEach((b, i) => { b.textContent = top[i][1].display + ' · ' + Object.keys(top[i][1].refs).length; });
  verifyPapers();
}
// Check the law: the brief checker, then opposing counsel on the same draft.
async function verifyLaw(text) {
  await Promise.all([loadJudgments(), citesReady]);
  const out = $('vf-out');
  out.innerHTML = '';
  renderBriefCheck(out, text);
  const next = el('div', 'ctas vf-next');
  next.innerHTML = '<input class="field" id="vf-side" value="Accused (petitioner)" aria-label="Your side"><button class="btn" type="button">Now argue against it: opposing counsel</button>' +
    '<button class="btn line" type="button">Check its facts against the papers</button>';
  next.children[1].addEventListener('click', async () => {
    const target = out.appendChild(el('div'));
    const facts = {'Your draft or argument (paste it)': text, 'Your side (e.g. accused, petitioner, tenant)': $('vf-side').value.trim()};
    const data = await watchedPost('/api/workflow', {kind: 'opponent', facts, mode: segValue('mode'), language: segValue('language')}, target, text);
    renderResult(target, data, 'vo' + (++seq), 'Opposing counsel', {noSave: true, whole: false, reveal: true, print: 'Opposing counsel', noHelp: true});
    target.scrollIntoView({behavior: 'smooth', block: 'start'});
  });
  next.lastChild.addEventListener('click', () => { $('vf-draft').value = text; verifyTab('facts'); $('vf-draft').scrollIntoView({behavior: 'smooth', block: 'center'}); });
  out.appendChild(next);
  out.scrollIntoView({behavior: 'smooth', block: 'start'});
}
$('vf-go').addEventListener('click', () => { const t = $('vf-text').value.trim(); if (t) verifyLaw(t); else $('vf-text').focus(); });
$('vf-sample').addEventListener('click', () => { $('vf-text').value = SAMPLE_PETITION; setSeg('mode', 'lawyer'); verifyLaw(SAMPLE_PETITION); });
// Check the facts: the papers are read into text here and kept only while the page is open.
let VF_PAPERS = [];
function verifyPapers() {
  const box = $('vf-papers'), vault = WS.mid ? wsVault() : [];
  box.innerHTML = VF_PAPERS.length || vault.length ? '' : '<p class="ws-empty">No papers yet: add the FIR, notices, orders or receipts.</p>';
  VF_PAPERS.forEach((d, i) => { const r = el('div', 'vf-paper'); r.innerHTML = '<b></b><span></span><button type="button" class="copy">Remove</button>';
    r.querySelector('b').textContent = d.name; r.querySelector('span').textContent = d.text.length.toLocaleString() + ' characters';
    r.querySelector('button').addEventListener('click', () => { VF_PAPERS.splice(i, 1); verifyPapers(); }); box.appendChild(r); });
  if (vault.length) box.insertAdjacentHTML('beforeend', '<label class="vf-vault"><input type="checkbox" id="vf-usevault"> Also use the ' + plural(vault.length, 'paper') + ' in my Workspace vault</label>');
}
$('vf-files').addEventListener('change', async e => {
  for (const f of [...e.target.files]) {
    const isText = /\.(txt|md)$/i.test(f.name) || f.type.startsWith('text/');
    const body = isText ? {filename: f.name, text: await f.text()}
      : {filename: f.name, mime: f.type || '', data: await new Promise(res => { const r = new FileReader(); r.onload = () => res(String(r.result).split(',')[1] || ''); r.readAsDataURL(f); })};
    let d;
    try { d = await (await fetch('/api/extract', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})).json(); }
    catch (err) { d = {route: 'ERROR', answer: err.message}; }
    if (d.route !== 'ERROR') VF_PAPERS.push({name: f.name, text: d.text});
    else $('vf-papers').insertAdjacentHTML('afterbegin', '<p class="error-text">' + escapeHtml(f.name + ': ' + d.answer) + '</p>');
  }
  e.target.value = '';
  verifyPapers();
});
$('vf-fir').addEventListener('click', () => { if (!VF_PAPERS.some(d => d.name === 'Sample FIR 214/2024')) VF_PAPERS.push({name: 'Sample FIR 214/2024', text: TOUR_FIR}); verifyPapers(); });
$('vf-check').addEventListener('click', async () => {
  const draft = $('vf-draft').value.trim();
  if (!draft) { $('vf-draft').focus(); return; }
  const docs = VF_PAPERS.concat($('vf-usevault') && $('vf-usevault').checked ? wsVault() : []);
  if (!docs.length) { $('vf-papers').scrollIntoView({behavior: 'smooth', block: 'center'}); return; }
  const out = $('vf-fout'); out.innerHTML = '';
  const papers = docs.map(d => '=== ' + d.name + ' ===\n' + d.text).join('\n\n').slice(0, 18000), target = out.appendChild(el('div'));
  const data = await watchedPost('/api/workflow', {kind: 'evidence', facts: {'Draft': draft, 'Papers': papers}, mode: segValue('mode'),
                                                   language: segValue('language')}, target, draft);
  renderResult(target, data, 'vfe' + (++seq), 'Evidence check', {noSave: true, whole: true, print: 'Evidence check', noHelp: true});
  markEvidence(target, papers);
  target.scrollIntoView({behavior: 'smooth', block: 'start'});
});
// Is a case still followed: the citator, from a citation typed or picked.
async function verifyCite(textOrKey) {
  await citesReady;
  let k = CITES[textOrKey] ? textOrKey : null;
  if (!k) { CITE_RE.lastIndex = 0; const m = CITE_RE.exec(textOrKey || ''); k = m ? citeKey(m) : null; }
  const out = $('vf-cout');
  if (!k || !CITES[k]) { out.innerHTML = '<p class="ws-empty">No judgment in the library cites that case, so there is nothing to read. Check the citation’s form (e.g. 2013 SCMR 1205).</p>'; return; }
  $('vf-cite').value = CITES[k].display;
  out.innerHTML = '<h3 class="vf-cite-h"></h3>';
  out.firstChild.textContent = CITES[k].display + ' · cited by ' + plural(Object.keys(CITES[k].refs).length, 'judgment') + ' in the library';
  await runCitator(k, out.appendChild(el('div')), $('vf-cite-go'));
}
$('vf-cite-go').addEventListener('click', () => verifyCite($('vf-cite').value.trim()));
$('vf-cite').addEventListener('keydown', e => { if (e.key === 'Enter') verifyCite($('vf-cite').value.trim()); });
$('vf-chips').addEventListener('click', e => { const b = e.target.closest('[data-k]'); if (b) verifyCite(b.dataset.k); });
// Check a video: still frames are taken here, in the browser; the video itself is never uploaded.
const VV = {frames: [], name: ''};
async function videoFrames(file, most) {
  const v = $('vv-video'), url = URL.createObjectURL(file);
  try {
    v.src = url;
    await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = () => rej(new Error('This video cannot be read in this browser. Try an MP4 straight from the phone.')); });
    const dur = Math.min(isFinite(v.duration) ? v.duration : 0, 180);
    if (!dur || !v.videoWidth) throw new Error('This video has no pictures that can be read.');
    const count = Math.max(2, Math.min(most, Math.ceil(dur / 2)));
    const scale = 640 / Math.max(v.videoWidth, v.videoHeight), c = document.createElement('canvas');
    c.width = Math.round(v.videoWidth * Math.min(1, scale)); c.height = Math.round(v.videoHeight * Math.min(1, scale));
    const out = [];
    for (let i = 0; i < count; i++) {
      const t = Math.min(dur - 0.05, dur * (i + 0.5) / count);
      await new Promise(res => { v.onseeked = res; v.currentTime = t; });
      c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
      out.push({t: Math.round(t * 10) / 10, data: c.toDataURL('image/jpeg', 0.72)});
    }
    return out;
  } finally { URL.revokeObjectURL(url); v.removeAttribute('src'); }
}
const vvTime = t => Math.floor(t / 60) + ':' + pad2(Math.floor(t % 60));
$('vv-file').addEventListener('change', async e => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  VV.frames = []; VV.name = f.name; $('vv-go').disabled = true; $('vv-out').innerHTML = '';
  $('vv-name').textContent = 'Reading ' + f.name + '…';
  $('vv-strip').innerHTML = '';
  try {
    VV.frames = await videoFrames(f, 8);
    $('vv-name').textContent = f.name + ' · ' + plural(VV.frames.length, 'frame') + ' taken, from 0:00 to ' + vvTime(VV.frames[VV.frames.length - 1].t);
    $('vv-strip').innerHTML = VV.frames.map(fr => '<figure><img alt="" src="' + fr.data + '"><figcaption>' + vvTime(fr.t) + '</figcaption></figure>').join('');
    $('vv-go').disabled = false;
  } catch (err) { $('vv-name').textContent = err.message; }
});
$('vv-go').addEventListener('click', async () => {
  if (!VV.frames.length) return;
  const out = $('vv-out'), context = $('vv-context').value.trim();
  out.innerHTML = '';
  const target = out.appendChild(el('div'));
  const data = await watchedPost('/api/video', {frames: VV.frames, context, mode: segValue('mode'), language: segValue('language')}, target, context);
  renderResult(target, data, 'vv' + (++seq), data.restated_query || 'Video evidence', {noSave: true, whole: false, print: 'Video evidence: ' + VV.name});
  // Does the video bear out what the person said? The verdict leads, in plain sight.
  if (data.match && data.match.verdict) {
    const v = data.match.verdict.toLowerCase(), kind = v === 'matches' ? 'ok' : v === 'partly matches' ? 'part' : 'no';
    const banner = el('div', 'vv-verdict vv-' + kind);
    banner.innerHTML = '<span>Does the video match what you said?</span><b></b><p></p>';
    banner.querySelector('b').textContent = data.match.verdict;
    banner.querySelector('p').textContent = data.match.reason;
    target.prepend(banner);
  }
  // Frame by frame: each line of the timeline beside the frame it describes, with the checker's verdict, so anyone
  // can see for themselves that it was in the video.
  if (data.route !== 'ERROR' && (data.rows || []).length) {
    const grid = el('div', 'vv-check');
    grid.innerHTML = '<div class="eyebrow">Frame by frame · see it for yourself</div>';
    data.rows.forEach((r, i) => {
      const fr = VV.frames.find(f => vvTime(f.t) === r.t) || VV.frames[i];
      const row = el('div', 'vv-row vv-' + (r.verdict || 'none').replace(' ', '-'));
      row.innerHTML = (fr ? '<img alt="">' : '<span></span>') + '<div><b></b><p></p></div><em></em>';
      if (fr) row.querySelector('img').src = fr.data;
      row.querySelector('b').textContent = r.t;
      row.querySelector('p').textContent = r.text;
      row.querySelector('em').textContent = {seen: 'Seen in the frame', partly: 'Partly: corrected', 'not seen': 'Not seen: corrected'}[r.verdict] || 'Not checked';
      grid.appendChild(row);
    });
    target.appendChild(grid);
  }
  if (data.route !== 'ERROR' && data.timeline) {
    const next = el('div', 'ctas');
    next.innerHTML = '<button class="btn" type="button">Check an FIR or a statement against this video</button>';
    next.firstChild.addEventListener('click', () => {
      const name = 'Video timeline: ' + VV.name;
      VF_PAPERS = VF_PAPERS.filter(d => d.name !== name).concat({name, text: data.timeline});
      verifyPapers(); verifyTab('facts'); $('vf-draft').focus();
      $('vf-draft').scrollIntoView({behavior: 'smooth', block: 'center'});
    });
    target.appendChild(next);
  }
  target.scrollIntoView({behavior: 'smooth', block: 'start'});
});
// The More menu in the top bar.
$('navmore').querySelector('button').addEventListener('click', e => {
  e.stopPropagation();
  const menu = $('navmore').querySelector('.navmenu');
  menu.hidden = !menu.hidden;
  e.currentTarget.setAttribute('aria-expanded', String(!menu.hidden));
});
document.addEventListener('click', e => { if (!e.target.closest('#navmore')) $('navmore').querySelector('.navmenu').hidden = true; });

// ===========================================================================
// ANSWER VIDEOS: any answer, or the rights card, as a 30-second phone video with captions and a voice, made in the
// browser. The script is the answer's own words: its question, its key points, the law behind it, free helplines.
// ===========================================================================
const VID_W = 720, VID_H = 1280;
const VID_TEXT = {
  en: {q: 'Your question', law: 'The law behind it', help: 'Free legal help', of: 'of',
       lawSay: 'The law behind this: ', helpSay: 'For free legal help, call 1818, the Supreme Court’s legal call centre, or 0800 70806, the Legal Aid Society.',
       end: 'Every answer traced to its paragraph.', note: 'General information, not legal advice.',
       brief: 'Case brief', ground: 'Ground', auth: 'Authorities', authSay: 'Authorities relied on: ',
       endL: 'Prepared with Wakeel AI', noteL: 'Check every authority in its paragraph before you rely on it.'},
  ur: {q: 'آپ کا سوال', law: 'قانون', help: 'مفت قانونی مدد', of: '/',
       lawSay: 'اس کے پیچھے قانون: ', helpSay: 'مفت قانونی مدد کے لیے سپریم کورٹ کی ہیلپ لائن اٹھارہ اٹھارہ پر کال کریں۔',
       end: 'ہر جواب فیصلوں سے ثابت۔', note: 'عمومی معلومات، قانونی مشورہ نہیں۔',
       brief: '\u06a9\u06cc\u0633 \u0628\u0631\u06cc\u0641', ground: '\u0646\u06a9\u062a\u06c1', auth: '\u062d\u0648\u0627\u0644\u06d2', authSay: '\u062c\u0646 \u062d\u0648\u0627\u0644\u0648\u06ba \u067e\u0631 \u0627\u0646\u062d\u0635\u0627\u0631 \u06c1\u06d2: ',
       endL: '\u0648\u06a9\u06cc\u0644 \u0627\u06d2 \u0622\u0626\u06cc \u06a9\u06d2 \u0633\u0627\u062a\u06be \u062a\u06cc\u0627\u0631 \u06a9\u0631\u062f\u06c1', noteL: '\u06c1\u0631 \u062d\u0648\u0627\u0644\u06c1 \u0627\u0633 \u06a9\u06d2 \u067e\u06cc\u0631\u0627\u06af\u0631\u0627\u0641 \u0645\u06cc\u06ba \u062f\u06cc\u06a9\u06be \u06a9\u0631 \u0627\u0633\u062a\u0639\u0645\u0627\u0644 \u06a9\u0631\u06cc\u06ba\u06d4'},
};
// The key points of an answer: its list items first, else its first sentences; disclaimers and table rows left out.
function videoPoints(answer, max) {
  const clean = l => plainWords(l).replace(/^\s*(?:[-*•]|\d+[.)])\s+/, '').replace(/^[^:]{2,40}:\s+(?=\S)/, m => m).replace(/\s+/g, ' ').trim();
  const skip = l => !l || l.length < 18 || /not legal advice|general information|قانونی مشورہ نہیں|legal aid committee/i.test(l);
  const cut = l => { if (l.length <= 150) return l; const t = l.slice(0, 150); const k = Math.max(t.lastIndexOf('. '), t.lastIndexOf('۔ ')); return k > 60 ? t.slice(0, k + 1) : t.replace(/\s+\S*$/, '') + '…'; };
  const lines = String(answer || '').split('\n').filter(l => !/^\s*\|/.test(l) && !/^\s*#/.test(l));
  let pts = lines.filter(l => /^\s*(?:[-*•]|\d+[.)])\s+/.test(l)).map(clean).filter(l => !skip(l));
  if (pts.length < 2) pts = (plainWords(lines.join(' ')).match(/[^.!?۔؟]+[.!?۔؟]/g) || []).map(x => x.trim()).filter(l => !skip(l));
  return pts.slice(0, max || 4).map(cut);
}
// style: 'people' explains it simply and ends with the free helplines; 'lawyer' is a brief to pitch the case to a
// client or a senior: the grounds, the authorities (the judgments it cites and the sections), and a reminder to verify.
function videoScript(data, title, style) {
  style = style || (segValue('mode') === 'lawyer' ? 'lawyer' : 'people');
  const lang = data.language === 'ur' ? 'ur' : 'en';
  const points = videoPoints(data.answer, 3);
  const laws = [...new Set([...linkStatutes(escapeHtml(data.answer || '')).matchAll(/data-stat="([^"]+)"/g)].map(m => m[1]))]
    .filter(k => STATUTES[k]).slice(0, 3).map(k => statuteLabel(k, STATUTES[k]));
  const cited = citedNumbers(data.answer || '');
  const cases = [...new Set((data.sources || []).filter(x => cited.has(x.n)).map(x => {
    const pt = x.parties || {};
    return pt.appellant && pt.respondent ? (pt.appellant + ' v. ' + pt.respondent).replace(/[()]/g, '') : x.title || [x.judgment_type, x.case_no].filter(Boolean).join(' ');
  }).filter(Boolean))].slice(0, 3);
  const script = videoSlides(lang, title || data.restated_query || '', points, laws, style, cases);
  script.data = data; script.sourceTitle = title;
  return script;
}
function videoSlides(lang, title, points, laws, style, cases) {
  const T = VID_TEXT[lang], slides = [], L = style === 'lawyer';
  if (title) slides.push({kind: 'title', head: L ? T.brief : T.q, text: title, say: title});
  points.forEach((t, i) => slides.push({kind: 'point', head: (L ? T.ground + ' ' : '') + pad2(i + 1) + ' ' + T.of + ' ' + pad2(points.length), n: i + 1, text: t, say: t}));
  const auth = L ? (cases || []).concat(laws) : laws;
  if (auth.length) slides.push({kind: 'law', head: L ? T.auth : T.law, text: auth.slice(0, 5).join('\n'), say: (L ? T.authSay : T.lawSay) + auth.slice(0, 5).join(', ')});
  if (!L) slides.push({kind: 'help', head: T.help, text: '', say: T.helpSay});
  slides.push({kind: 'end', head: '', text: L ? T.endL : T.end, sub: L ? T.noteL : T.note, say: ''});
  return {lang, title, slides, style: L ? 'lawyer' : 'people'};
}
function rightsVideoScript(lang) {
  const L = lang === 'ur', items = RIGHTS.filter(r => STATUTES[r.k]).slice(0, 3);
  return videoSlides(L ? 'ur' : 'en', L ? 'گرفتاری کے وقت آپ کے حقوق' : 'Your rights if you are arrested',
                     items.map(r => L ? r.ur : r.en), [...new Set(items.map(r => statuteLabel(r.k, STATUTES[r.k])))].slice(0, 3), 'people', []);
}
function vidWrap(ctx, text, max) {
  const out = [];
  String(text).split('\n').forEach(par => {
    let line = '';
    par.split(/\s+/).forEach(w => { const t = line ? line + ' ' + w : w; if (ctx.measureText(t).width > max && line) { out.push(line); line = w; } else line = t; });
    if (line) out.push(line);
  });
  return out;
}
function vidDraw(ctx, script, i, local, dur, wave) {
  const sl = script.slides[i], rtl = script.lang === 'ur', urFont = '"Segoe UI", "Noto Naskh Arabic", Tahoma, sans-serif';
  const disp = (w, px) => w + ' ' + px + 'px Bahnschrift, "DIN Condensed", "Arial Narrow", sans-serif';
  const body = (w, px) => w + ' ' + px + 'px ' + (rtl ? urFont : '"Segoe UI", system-ui, sans-serif');
  const k = Math.min(1, local / 0.55), ease = 1 - Math.pow(1 - k, 3);
  ctx.fillStyle = '#0a0a0a'; ctx.fillRect(0, 0, VID_W, VID_H);
  ctx.fillStyle = '#d8cdb9'; ctx.fillRect(0, 0, VID_W, 10);
  ctx.textBaseline = 'alphabetic'; ctx.direction = 'ltr'; ctx.textAlign = 'left';
  ctx.font = disp('700', 34); ctx.fillStyle = '#d8cdb9'; ctx.fillText('WAKEEL AI', 56, 96);
  ctx.textAlign = 'right'; ctx.font = body('400', 22); ctx.fillStyle = '#9a9a96'; ctx.fillText((i + 1) + ' / ' + script.slides.length, VID_W - 56, 94);
  ctx.globalAlpha = ease;
  const dy = (1 - ease) * 40, x0 = 56, xr = VID_W - 56, w = VID_W - 112;
  const put = (t, y) => { ctx.textAlign = rtl ? 'right' : 'left'; ctx.direction = rtl ? 'rtl' : 'ltr'; ctx.fillText(t, rtl ? xr : x0, y); };
  if (sl.kind === 'title' || sl.kind === 'point') {
    ctx.font = sl.kind === 'title' ? body('600', 28) : disp('700', 150); ctx.fillStyle = '#d8cdb9';
    if (sl.kind === 'title') put(sl.head, 300 + dy); else { ctx.direction = 'ltr'; ctx.textAlign = 'left'; ctx.fillText(pad2(sl.n), x0, 400 + dy); }
    ctx.font = body(sl.kind === 'title' ? '700' : '600', rtl ? 50 : 52); ctx.fillStyle = '#ffffff';
    const lines = vidWrap(ctx, sl.text, w), lh = rtl ? 86 : 70, y0 = sl.kind === 'title' ? 400 : 520;
    lines.slice(0, 9).forEach((l, j) => put(l, y0 + j * lh + dy));
  } else if (sl.kind === 'law') {
    ctx.font = '600 40px "Segoe UI", system-ui, sans-serif';
    const items = sl.text.split('\n').map(l => vidWrap(ctx, l, w - 72).slice(0, 2));
    const boxH = Math.min(700, 230 + items.reduce((a, ls) => a + ls.length * 50 + 26, 0));
    ctx.fillStyle = '#e4d9c6'; ctx.fillRect(x0, 330 + dy, w, boxH);
    ctx.fillStyle = '#0a0a0a'; ctx.font = disp('700', 150); ctx.direction = 'ltr'; ctx.textAlign = 'left'; ctx.fillText('§', x0 + 36, 490 + dy);
    ctx.font = '600 40px "Segoe UI", system-ui, sans-serif';
    let y = 580;
    items.forEach(ls => { ls.forEach(m => { if (y < 330 + boxH - 20) ctx.fillText(m, x0 + 36, y + dy); y += 50; }); y += 26; });
    ctx.fillStyle = '#d8cdb9'; ctx.font = body('600', 28); put(sl.head, 300 + dy);
  } else if (sl.kind === 'help') {
    ctx.fillStyle = '#d8cdb9'; ctx.font = body('600', 30); put(sl.head, 300 + dy);
    HELPLINES.slice(0, 3).forEach((h, j) => {
      const y = 430 + j * 190 + dy;
      ctx.direction = 'ltr'; ctx.textAlign = 'left'; ctx.font = disp('700', 96); ctx.fillStyle = '#ffffff'; ctx.fillText(h[0], x0, y);
      ctx.font = '500 30px "Segoe UI", system-ui, sans-serif'; ctx.fillStyle = '#d8cdb9'; ctx.fillText(h[2], x0, y + 48);
    });
  } else {
    ctx.direction = 'ltr'; ctx.textAlign = 'center'; ctx.fillStyle = '#ffffff'; ctx.font = disp('700', 120); ctx.fillText('WAKEEL AI', VID_W / 2, 600 + dy);
    ctx.font = body('500', 34); ctx.fillStyle = '#d8cdb9'; ctx.direction = rtl ? 'rtl' : 'ltr'; ctx.fillText(sl.text, VID_W / 2, 690 + dy);
    ctx.font = body('400', 24); ctx.fillStyle = '#9a9a96'; ctx.fillText(sl.sub || '', VID_W / 2, 750 + dy);
  }
  ctx.globalAlpha = 1;
  // the voice, drawn as it speaks
  const n = 42, bw = w / n;
  ctx.fillStyle = '#d8cdb9';
  for (let b = 0; b < n; b++) {
    const v = wave ? wave[Math.floor(b * wave.length / n / 2)] / 255 : 0;
    const h = 6 + v * 90;
    ctx.fillRect(x0 + b * bw + 2, 1110 - h / 2, bw - 5, h);
  }
  ctx.fillStyle = '#2b2b2b'; ctx.fillRect(x0, 1190, w, 6);
  ctx.fillStyle = '#d8cdb9'; ctx.fillRect(x0, 1190, w * Math.min(1, local / dur), 6);
}
const VID = {busy: false, stop: null};
function vidStep(text, state) {
  const li = el('li', state || ''); li.textContent = text; $('vid-steps').appendChild(li); return li;
}
async function makeVideo(script) {
  if (VID.busy || !script || script.slides.length < 3) return;
  VID.busy = true;
  const box = $('vid'), canvas = $('vid-canvas'), video = $('vid-video'), ctx = canvas.getContext('2d');
  box.hidden = false; document.body.classList.add('voice-on');
  canvas.hidden = false; video.hidden = true; video.removeAttribute('src');
  $('vid-steps').innerHTML = ''; $('vid-acts').hidden = true;
  $('vid-title').textContent = (script.style === 'lawyer' ? 'Video brief: ' : '') + (script.title || 'Your answer, as a video');
  vidStep('Script taken from the answer: ' + plural(script.slides.length, 'slide'), 'done');
  vidDraw(ctx, script, 0, 0, 1, null);
  const voiceStep = vidStep('Recording the voice…', 'on');
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  try { await ac.resume(); } catch (e) {}
  const clips = await Promise.all(script.slides.map(async sl => {
    if (!sl.say) return null;
    try {
      const r = await fetch('/api/speak', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                           body: JSON.stringify({text: sl.say, language: script.lang, tone: script.style === 'lawyer' ? 'formal' : 'plain'})});
      if (!r.ok) return null;
      return await ac.decodeAudioData(await r.arrayBuffer());
    } catch (e) { return null; }
  }));
  const voiced = clips.filter(Boolean).length;
  voiceStep.className = 'done';
  voiceStep.textContent = voiced ? 'Voice ready: ' + plural(voiced, 'clip') + ' in ' + (script.lang === 'ur' ? 'Urdu' : 'English') : 'No voice available: captions only';
  const words = t => (String(t).match(/\S+/g) || []).length;
  const durs = script.slides.map((sl, i) => clips[i] ? clips[i].duration + 0.7 : sl.kind === 'end' ? 2.8 : Math.max(2.8, words(sl.text) / 2.6 + 1));
  const starts = durs.reduce((a, d, i) => (a.push(i ? a[i - 1] + durs[i - 1] : 0), a), []), total = starts[starts.length - 1] + durs[durs.length - 1];
  const filmStep = vidStep('Filming…', 'on');
  const dest = ac.createMediaStreamDestination(), an = ac.createAnalyser();
  an.fftSize = 256; an.connect(dest); an.connect(ac.destination);
  const stream = new MediaStream([...canvas.captureStream(30).getVideoTracks(), ...dest.stream.getAudioTracks()]);
  const mime = ['video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm'].find(m => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || '';
  const rec = new MediaRecorder(stream, mime ? {mimeType: mime, videoBitsPerSecond: 3000000} : {});
  const chunks = [];
  rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
  const done = new Promise(res => { rec.onstop = res; });
  rec.start(250);
  // Audio may be held back until someone presses something; then the film runs on the page's clock, silent.
  const live = ac.state === 'running', now = () => live ? ac.currentTime : performance.now() / 1000;
  const t0 = now() + 0.25;
  clips.forEach((b, i) => { if (!b || !live) return; const src = ac.createBufferSource(); src.buffer = b; src.connect(an); src.start(t0 + starts[i] + 0.35); });
  const wave = new Uint8Array(an.frequencyBinCount);
  let stopped = false;
  VID.stop = () => { stopped = true; };
  await new Promise(res => {
    const frame = () => {
      const t = now() - t0;
      if (stopped || t > total) { res(); return; }
      let i = starts.length - 1; while (i > 0 && starts[i] > t) i--;
      an.getByteFrequencyData(wave);
      vidDraw(ctx, script, i, Math.max(0, t - starts[i]), durs[i], wave);
      const sec = Math.max(0, Math.floor(t));
      filmStep.textContent = 'Filming ' + Math.floor(sec / 60) + ':' + pad2(sec % 60) + ' of ' + Math.floor(total / 60) + ':' + pad2(Math.round(total % 60));
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  });
  rec.stop(); await done; ac.close();
  VID.busy = false;
  if (stopped) return;
  const type = (mime || 'video/webm').split(';')[0], ext = type.includes('mp4') ? 'mp4' : 'webm';
  const blob = new Blob(chunks, {type}), url = URL.createObjectURL(blob);
  filmStep.className = 'done'; filmStep.textContent = 'Done: ' + Math.round(total) + ' seconds, ' + (blob.size / 1048576).toFixed(1) + ' MB (' + ext.toUpperCase() + ')';
  canvas.hidden = true; video.hidden = false; video.src = url;
  const file = new File([blob], 'wakeel-answer.' + ext, {type});
  const acts = $('vid-acts');
  acts.hidden = false;
  acts.innerHTML = '<a class="btn" download="wakeel-answer.' + ext + '">Download the video</a>' +
    (navigator.canShare && navigator.canShare({files: [file]}) ? '<button class="btn line" type="button" data-a="share">Share</button>' : '') +
    '<a class="btn line" target="_blank" rel="noopener">Send the summary on WhatsApp</a>' +
    (script.data ? '<button class="btn line" type="button" data-a="other">' + (script.style === 'lawyer' ? 'Make the version for a client' : 'Make the brief for a lawyer') + '</button>' : '') +
    '<button class="btn line" type="button" data-a="again">Make it again</button>';
  acts.firstChild.href = url;
  const summary = [script.title].concat(script.slides.filter(s => s.kind === 'point').map(s => '• ' + s.text))
    .concat(['', 'Free legal help: 1818 (Supreme Court) · 0800-70806 (Legal Aid Society)', 'Wakeel AI']).join('\n');
  acts.querySelector('a[target]').href = 'https://wa.me/?text=' + encodeURIComponent(summary);
  const sh = acts.querySelector('[data-a="share"]');
  if (sh) sh.addEventListener('click', () => navigator.share({files: [file], title: 'Wakeel AI', text: script.title}).catch(() => {}));
  acts.querySelector('[data-a="again"]').addEventListener('click', () => makeVideo(script));
  const other = acts.querySelector('[data-a="other"]');
  if (other) other.addEventListener('click', () => makeVideo(videoScript(script.data, script.sourceTitle, script.style === 'lawyer' ? 'people' : 'lawyer')));
  video.play().catch(() => {});
}
function closeVideo() { if (VID.stop) VID.stop(); $('vid').hidden = true; document.body.classList.remove('voice-on'); const v = $('vid-video'); v.pause(); }
$('vid-x').addEventListener('click', closeVideo);
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('vid').hidden) closeVideo(); });

// The agent room's map: what each agent touches lights up, with a line saying who touched what.
function netSay(who, n, what) {
  if (!n) return;
  const ol = $('room-netcap'), li = el('li');
  li.innerHTML = '<b></b> ';
  li.firstChild.textContent = who;
  li.appendChild(document.createTextNode(what));
  ol.appendChild(li);
  while (ol.children.length > 4) ol.firstChild.remove();
}
$('case-file').addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f) return;
  if (f.size > 10 * 1024 * 1024) { $('case-fname').textContent = 'That file is larger than 10 MB'; caseFile = null; return; }
  const r = new FileReader();
  r.onload = () => { caseFile = {filename: f.name, mime: f.type || '', data: String(r.result).split(',')[1] || ''}; $('case-fname').textContent = f.name; };
  r.readAsDataURL(f);
});
// From an answer, an explained document or a judgment straight into the agent room.
function openCase(situation, documentText, docIds) {
  go('/case');
  $('case-text').value = situation || '';
  caseFile = documentText ? {filename: 'document.txt', text: documentText} : null;
  caseDocs = docIds || [];
  $('case-fname').textContent = documentText ? 'The document you explained is attached' : '';
  if (caseDocs.length) $('case-go').click(); else $('case-text').focus();
}
function caseClock(ms) { return (ms / 1000).toFixed(1) + ' s'; }
function agentName(id) { const a = (caseRun && caseRun.team || []).find(x => x.id === id); return a ? a.name : id === 'orchestrator' ? 'Orchestrator' : id; }
function caseLog(text, who) {
  const li = el('li');
  li.innerHTML = '<span></span><div></div>';
  li.firstChild.textContent = caseClock(performance.now() - caseRun.t0);
  li.lastChild.innerHTML = (who ? '<b>' + escapeHtml(who) + '</b> ' : '') + escapeHtml(text);
  $('case-feed').appendChild(li);
  $('case-feed').scrollTop = $('case-feed').scrollHeight;
}
// ---- the graph ----
// ---- the live map: every judgment, the reported cases it cites and the provisions it applies, drawn as one network.
// It drifts and carries signals along its lines on its own; light() makes nodes flare and send signals outwards.
const NET_DARK = {j: [216, 205, 185], c: [223, 230, 238], p: [244, 240, 232], edge: 'rgba(223,230,238,.07)', hot: 'rgba(216,205,185,.30)',
                  pulse: [223, 230, 238], strong: [244, 240, 232], halo: 'rgba(216,205,185,.16)', flare: [244, 240, 232],
                  label: 'rgba(10,10,10,.85)', labelInk: '#f4f0e8'};
const NET_LIGHT = {j: [31, 58, 86], c: [92, 113, 116], p: [115, 109, 99], edge: 'rgba(31,58,86,.08)', hot: 'rgba(31,58,86,.42)',
                   pulse: [43, 76, 109], strong: [31, 58, 86], halo: 'rgba(31,58,86,.12)', flare: [31, 58, 86],
                   label: 'rgba(244,240,232,.92)', labelInk: '#111111'};
function provisionKey(p) {
  const m = String(p).toLowerCase().match(/\b(section|article|order|rule)\s+([0-9]+[a-z-]*(?:\([0-9a-z]+\))?)/);
  return m ? m[1] + ' ' + m[2] : '';
}
function makeLawNet(canvas, opts) {
  opts = opts || {};
  const W0 = 1000, H0 = Math.round(1000 * (opts.aspect || 0.6));
  const N = {nodes: [], byId: {}, adj: [], edges: [], lit: {}, flares: [], pulses: [], labels: [], built: false};
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  N.ready = Promise.all([loadJudgments(), citesReady]).then(() => {
    const add = (id, kind, label, r) => { N.byId[id] = N.nodes.length; N.nodes.push({id, kind, label, r, ph: Math.random() * 6.28}); N.adj.push([]); };
    const link = (a, b) => { const i = N.byId[a], k = N.byId[b]; if (i === undefined || k === undefined) return; N.edges.push([i, k]); N.adj[i].push(k); N.adj[k].push(i); };
    (judgments || []).forEach(j => add('j:' + j.doc_id, 'j', caseTitle(j), 3 + Math.min(5, Math.sqrt(j.passages || 1) / 1.6)));
    const cites = Object.entries(CITES).sort((a, b) => Object.keys(b[1].refs).length - Object.keys(a[1].refs).length).slice(0, opts.cites || 240);
    cites.forEach(([k, c]) => add('c:' + k, 'c', c.display, 1.4 + Math.min(2.5, Object.keys(c.refs).length * 0.7)));
    const provs = {};
    (judgments || []).forEach(j => [...new Set((j.provisions || []).map(provisionKey).filter(Boolean))].forEach(k => (provs[k] = provs[k] || []).push(j.doc_id)));
    Object.entries(provs).filter(([, ds]) => ds.length > 1).forEach(([k, ds]) => { add('p:' + k, 'p', k.replace(/^./, c => c.toUpperCase()), 2.2); ds.forEach(d => link('j:' + d, 'p:' + k)); });
    cites.forEach(([k, c]) => Object.keys(c.refs).forEach(d => link('j:' + d, 'c:' + k)));
    // A constellation: the judgments placed by what they share, every cited case and provision in a halo around the
    // judgments that use it.
    const J = N.nodes.filter(n => n.kind === 'j'), ji = {};
    J.forEach((n, i) => { ji[N.byId[n.id]] = i; });
    const shared = {};
    N.nodes.forEach((n, i) => { if (n.kind === 'j') return;
      const js = N.adj[i].map(k => ji[k]).filter(k => k !== undefined);
      for (let a = 0; a < js.length; a++) for (let b = a + 1; b < js.length; b++) shared[Math.min(js[a], js[b]) + ',' + Math.max(js[a], js[b])] = 1; });
    layoutGraph(J, Object.keys(shared).map(k => k.split(',').map(Number)), W0, H0, 220);
    let seed = 7;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    N.nodes.forEach((n, i) => {
      if (n.kind === 'j') return;
      const js = N.adj[i].map(k => N.nodes[k]).filter(m => m.kind === 'j');
      const cx = js.reduce((a, m) => a + m.x, 0) / (js.length || 1), cy = js.reduce((a, m) => a + m.y, 0) / (js.length || 1);
      const a = rnd() * 6.283, r = (n.kind === 'p' ? 14 : 22) + rnd() * (js.length > 1 ? 30 : 62);
      n.x = Math.max(12, Math.min(W0 - 12, (js.length ? cx : rnd() * W0) + Math.cos(a) * r));
      n.y = Math.max(12, Math.min(H0 - 12, (js.length ? cy : rnd() * H0) + Math.sin(a) * r * 0.8));
    });
    if (opts.hole) N.nodes.forEach(n => {   // keep the middle clear for what sits on top of the map
      const dx = (n.x - W0 / 2) / (W0 * opts.hole[0]), dy = (n.y - H0 / 2) / (H0 * opts.hole[1]), d = Math.hypot(dx, dy);
      if (d < 1) { const k = (1 + rnd() * 0.25) / Math.max(d, 0.05); n.x = W0 / 2 + dx * k * W0 * opts.hole[0]; n.y = H0 / 2 + dy * k * H0 * opts.hole[1]; }
      n.x = Math.max(10, Math.min(W0 - 10, n.x)); n.y = Math.max(10, Math.min(H0 - 10, n.y));
    });
    N.built = true;
  });
  const ctx = canvas.getContext('2d');
  let C = null;   // the palette, chosen at the first frame (the palettes are defined further down the script)
  let w = 0, h = 0, last = 0, ambient = 0;
  function pos(n, t) {
    const dx = reduce ? 0 : Math.sin(t * 0.00035 + n.ph) * 4, dy = reduce ? 0 : Math.cos(t * 0.0003 + n.ph * 1.3) * 4;
    return [(n.x / W0) * w + dx, (n.y / H0) * h + dy];
  }
  function isLit(i, t) { const l = N.lit[i]; return l && t - l.t < l.ttl; }
  function pulse(a, b, t, strong) { N.pulses.push({a, b, t0: t, dur: strong ? 900 : 1500 + Math.random() * 900, strong}); }
  N.light = (ids, o) => {
    o = o || {};
    const t = performance.now(), found = [];
    ids.forEach((id, n) => {
      const i = N.byId[id];
      if (i === undefined) return;
      found.push(i);
      N.lit[i] = {t: t + n * 120, ttl: o.ttl || 1e12};
      N.flares.push({i, t0: t + n * 120});
      if (o.label !== false && opts.labels && N.nodes[i].kind === 'j') { N.labels = N.labels.filter(l => l.i !== i).concat({i, t0: t, ttl: o.ttl || 1e12}).slice(-(opts.labels || 5)); }
      N.adj[i].slice(0, 14).forEach((k, m) => setTimeout(() => pulse(i, k, performance.now(), true), n * 120 + m * 70));
    });
    return found.length;
  };
  N.spark = () => {   // activity from what is already lit: an agent writing from it
    const lit = Object.keys(N.lit).filter(i => isLit(+i, performance.now()));
    if (!lit.length) return;
    const i = +lit[Math.floor(Math.random() * lit.length)], nb = N.adj[i];
    if (nb.length) pulse(i, nb[Math.floor(Math.random() * nb.length)], performance.now(), true);
  };
  N.reset = () => { N.lit = {}; N.labels = []; N.flares = []; N.pulses = []; };
  function frame(t) {
    requestAnimationFrame(frame);
    if (!N.built || !canvas.isConnected || canvas.offsetParent === null) return;
    C = C || (opts.light ? NET_LIGHT : NET_DARK);
    const dpr = Math.min(2, window.devicePixelRatio || 1), cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) { canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr); }
    w = cw; h = ch;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const P = N.nodes.map(n => pos(n, t));
    // quiet lines; lines touching a lit node brighter
    ctx.lineWidth = 1;
    N.edges.forEach(([a, b]) => {
      const hot = isLit(a, t) || isLit(b, t);
      ctx.strokeStyle = hot ? C.hot : C.edge;
      ctx.beginPath(); ctx.moveTo(P[a][0], P[a][1]); ctx.lineTo(P[b][0], P[b][1]); ctx.stroke();
    });
    // signals along the lines
    if (!reduce && t - ambient > (opts.ambientMs || 70)) { ambient = t; const e = N.edges[Math.floor(Math.random() * N.edges.length)]; if (e) pulse(Math.random() < .5 ? e[0] : e[1], Math.random() < .5 ? e[1] : e[0], t, false); }
    N.pulses = N.pulses.filter(p => t - p.t0 < p.dur);
    N.pulses.forEach(p => {
      const k = Math.max(0, (t - p.t0) / p.dur), a = P[p.a], b = P[p.b];
      for (let s = 0; s < (p.strong ? 4 : 2); s++) {
        const q = Math.max(0, k - s * 0.04), x = a[0] + (b[0] - a[0]) * q, y = a[1] + (b[1] - a[1]) * q;
        ctx.fillStyle = p.strong ? 'rgba(' + C.strong.join(',') + ',' + (0.9 - s * 0.2) + ')' : 'rgba(' + C.pulse.join(',') + ',' + (0.45 - s * 0.2) + ')';
        ctx.beginPath(); ctx.arc(x, y, (p.strong ? 2.6 : 1.5) - s * 0.4, 0, 6.29); ctx.fill();
      }
    });
    // nodes
    N.nodes.forEach((n, i) => {
      const [x, y] = P[i], c = C[n.kind], hot = isLit(i, t);
      if (hot) { ctx.fillStyle = C.halo; ctx.beginPath(); ctx.arc(x, y, n.r + 9, 0, 6.29); ctx.fill(); }
      ctx.fillStyle = 'rgba(' + c.join(',') + ',' + (hot ? 1 : n.kind === 'j' ? .7 : .5) + ')';
      if (n.kind === 'p') { ctx.strokeStyle = ctx.fillStyle; ctx.strokeRect(x - n.r, y - n.r, n.r * 2, n.r * 2); }
      else { ctx.beginPath(); ctx.arc(x, y, hot ? n.r + 1.5 : n.r, 0, 6.29); ctx.fill(); }
    });
    // flares: a ring that opens out where an agent touched the law
    N.flares = N.flares.filter(f => t - f.t0 < 1400);
    N.flares.forEach(f => {
      const k = (t - f.t0) / 1400;
      if (k < 0) return;
      const [x, y] = P[f.i];
      ctx.strokeStyle = 'rgba(' + C.flare.join(',') + ',' + (1 - k) + ')'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, N.nodes[f.i].r + 4 + k * 34, 0, 6.29); ctx.stroke();
    });
    // names of the judgments in play
    N.labels = N.labels.filter(l => t - l.t0 < l.ttl);
    ctx.font = '600 12px ' + getComputedStyle(document.body).fontFamily;
    N.labels.forEach(l => {
      const [x, y] = P[l.i], text = N.nodes[l.i].label.slice(0, 42), tw = ctx.measureText(text).width;
      const lx = Math.min(w - tw - 14, x + 10), ly = Math.max(16, y - 10);
      ctx.fillStyle = C.label; ctx.fillRect(lx - 5, ly - 12, tw + 10, 17);
      ctx.fillStyle = C.labelInk; ctx.fillText(text, lx, ly);
    });
    last = t;
  }
  requestAnimationFrame(frame);
  return N;
}
function roomLayers(team, edges) {
  const layer = {orchestrator: 0};
  for (let k = 0; k < team.length; k++) edges.forEach(([a, b]) => { if (layer[a] !== undefined && (layer[b] === undefined || layer[b] < layer[a] + 1)) layer[b] = layer[a] + 1; });
  team.forEach(a => { if (layer[a.id] === undefined) layer[a.id] = 1; });
  return layer;
}
const AGENT_MARK = {orchestrator: 'OR', research: 'RS', statutes: '§', precedents: 'PR', strategy: 'SG', draft: 'DR', counter: 'CT',
                    explainer: 'EX', reader: 'RD', timeline: 'TL', similar: 'SM', reader_a: 'A', reader_b: 'B', comparator: 'CM', verify: 'VF'};
const ROOM_KINDS_SHOWN = [['situation', 'A situation'], ['question', 'A question'], ['judgment', 'A judgment'], ['document', 'A document'], ['compare', 'Two judgments']];
function roomRoute(kind) {
  const r = $('room-route');
  r.className = 'route' + (kind ? '' : ' deciding');
  r.innerHTML = '<span class="rt-l">Orchestrator routing</span>' + ROOM_KINDS_SHOWN.map(([k, label]) =>
    '<span data-kind="' + k + '" class="' + (kind ? (k === kind ? 'on' : 'off') : '') + '">' + label + '</span>').join('');
}
function roomNodeHtml(a, run) {
  return '<div class="rh"><div class="ring"><svg viewBox="0 0 34 34"><circle class="bg" cx="17" cy="17" r="15"/><circle class="arc" cx="17" cy="17" r="15"/></svg>' +
    '<span>' + (AGENT_MARK[a.id] || a.name.slice(0, 2).toUpperCase()) + '</span></div><b></b><span class="rk ' + (a.ai ? 'ai">AI' : 'rules">No AI') + '</span></div>' +
    '<p class="rnote"></p><p class="rlive" hidden></p><ul class="rmini"></ul>' + (a.id === 'verify' ? '<div class="rmeter"><i style="width:0"></i></div>' : '') +
    '<div class="rt"></div>';
}
function roomDraw() {
  const box = $('case-team'), run = caseRun;
  if (!run || !run.team) return;
  const W = box.clientWidth || 900, vertical = W < 760;
  const layer = roomLayers(run.team, run.edges);
  const L = Math.max(...Object.values(layer)) + 1;
  const groups = Array.from({length: L}, () => []);
  run.team.forEach(a => groups[layer[a.id]].push(a.id));
  const maxK = Math.max(...groups.map(g => g.length));
  const NH = 150, H = vertical ? L * 200 + 10 : Math.max(400, maxK * 182 + 24);
  box.style.height = H + 'px';
  const pos = {};
  groups.forEach((ids, l) => ids.forEach((id, i) => {
    const nw = vertical ? Math.min(190, W / ids.length - 12) : Math.min(222, W / L - 28);
    pos[id] = vertical ? {x: W * (i + 0.5) / ids.length, y: 85 + l * 200, w: nw}
                       : {x: W * (l + 0.5) / L, y: H * (i + 0.5) / ids.length, w: nw};
  }));
  run.pos = pos; run.vertical = vertical;
  box.innerHTML = '<svg></svg>';
  const svg = box.firstChild, NS = 'http://www.w3.org/2000/svg';
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  run.paths = {};
  run.edges.forEach(([a, b]) => {
    const p = pos[a], q = pos[b];
    if (!p || !q) return;
    let d;
    if (vertical) { const y1 = p.y + NH / 2, y2 = q.y - NH / 2; d = 'M' + p.x + ',' + y1 + ' C' + p.x + ',' + (y1 + 44) + ' ' + q.x + ',' + (y2 - 44) + ' ' + q.x + ',' + y2; }
    else { const x1 = p.x + p.w / 2, x2 = q.x - q.w / 2; d = 'M' + x1 + ',' + p.y + ' C' + (x1 + 56) + ',' + p.y + ' ' + (x2 - 56) + ',' + q.y + ' ' + x2 + ',' + q.y; }
    const path = document.createElementNS(NS, 'path');
    path.setAttribute('d', d);
    svg.appendChild(path);
    run.paths[a + '>' + b] = path;
    if (run.labels[a + '>' + b]) roomChip(path, run.labels[a + '>' + b]);
  });
  run.team.forEach(a => {
    const p = pos[a.id], node = el('div', 'rnode');
    node.dataset.agent = a.id;
    node.tabIndex = 0;
    node.style.left = (p.x - p.w / 2) + 'px'; node.style.top = (p.y - NH / 2) + 'px'; node.style.width = p.w + 'px'; node.style.minHeight = NH + 'px';
    node.innerHTML = roomNodeHtml(a, run);
    node.querySelector('b').textContent = a.name;
    node.title = a.role;
    const jump = () => { const sec = $('case-out').querySelector('[data-sec="' + a.id + '"]'); if (sec) sec.scrollIntoView({behavior: 'smooth'}); };
    node.addEventListener('click', jump);
    node.addEventListener('keydown', e => { if (e.key === 'Enter') jump(); });
    box.appendChild(node);
    roomPaintNode(a.id);
  });
  roomEdges();
  roomCards();
}
window.addEventListener('resize', () => { if (caseRun && caseRun.team && pageFromPath(location.pathname) === 'case') roomDraw(); });
function roomPaintNode(id) {
  const run = caseRun, node = $('case-team').querySelector('[data-agent="' + id + '"]');
  if (!node) return;
  node.dataset.status = run.status[id] || 'waiting';
  const a = run.team.find(x => x.id === id);
  node.querySelector('.rnote').textContent = run.notes[id] || (a ? a.role : '');
  const ul = node.querySelector('.rmini');
  const items = run.mini[id] || [];
  if (ul.dataset.n !== String(items.length)) {
    ul.innerHTML = items.slice(0, 2).map(t => '<li></li>').join('');
    [...ul.children].forEach((li, i) => { li.textContent = items[i]; });
    ul.dataset.n = String(items.length);
  }
  const live = node.querySelector('.rlive'), words = run.live[id];
  live.hidden = !(words && run.status[id] === 'working');
  if (!live.hidden) live.textContent = roomTail(words, 120);
  node.querySelector('.rnote').hidden = !live.hidden && !!words;
  ul.hidden = !live.hidden;
  const meter = node.querySelector('.rmeter i');
  if (meter && run.meter) meter.style.width = Math.round(100 * run.meter[0] / Math.max(1, run.meter[1])) + '%';
}
// While data flows into an agent that is working, its incoming line runs; afterwards it stays lit.
function roomEdges() {
  const run = caseRun;
  Object.entries(run.paths || {}).forEach(([key, path]) => {
    const to = key.split('>')[1], used = run.used[key];
    path.setAttribute('class', 'redge' + (used ? ' used' : '') + (used && run.status[to] === 'working' ? ' flow' : ''));
  });
}
function roomChip(path, label) {
  const NS = 'http://www.w3.org/2000/svg', mid = path.getPointAtLength(path.getTotalLength() / 2);
  const t = document.createElementNS(NS, 'text');
  t.setAttribute('class', 'rchip'); t.setAttribute('text-anchor', 'middle');
  t.setAttribute('x', mid.x); t.setAttribute('y', mid.y - 6);
  t.textContent = label;
  path.ownerSVGElement.appendChild(t);
}
function roomPulse(from, to, label) {
  const run = caseRun, key = from + '>' + to, path = run.paths && run.paths[key];
  caseLog(label, agentName(from) + ' → ' + agentName(to) + ':');
  if (!path) return;
  run.used[key] = true;
  roomEdges();
  const svg = path.ownerSVGElement, NS = 'http://www.w3.org/2000/svg', len = path.getTotalLength();
  const trail = [3.5, 2.5, 1.6].map((r, i) => { const c = document.createElementNS(NS, 'circle'); c.setAttribute('r', r); c.setAttribute('class', 'rtrail'); c.style.opacity = 0.55 - i * 0.15; svg.appendChild(c); return c; });
  const dot = document.createElementNS(NS, 'circle'), txt = document.createElementNS(NS, 'text');
  dot.setAttribute('r', 6); dot.setAttribute('class', 'rdot');
  txt.setAttribute('class', 'rlabel'); txt.setAttribute('text-anchor', 'middle'); txt.textContent = label;
  svg.append(dot, txt);
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches, t0 = performance.now(), dur = reduce ? 1 : 1300;
  const at = t => path.getPointAtLength(len * Math.max(0, Math.min(1, t)));
  const step = now => {
    const t = Math.min(1, (now - t0) / dur), e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2, pt = at(e);
    dot.setAttribute('cx', pt.x); dot.setAttribute('cy', pt.y);
    trail.forEach((c, i) => { const q = at(e - 0.05 * (i + 1)); c.setAttribute('cx', q.x); c.setAttribute('cy', q.y); });
    txt.setAttribute('x', pt.x); txt.setAttribute('y', pt.y - 12);
    if (t < 1) requestAnimationFrame(step);
    else {
      dot.remove(); trail.forEach(c => c.remove()); txt.remove();
      if (!run.labels[key]) { run.labels[key] = label; roomChip(path, label); }
      const node = $('case-team').querySelector('[data-agent="' + to + '"]');
      if (node) { node.classList.remove('ping'); void node.offsetWidth; node.classList.add('ping'); }
    }
  };
  requestAnimationFrame(step);
}
function caseSet(id, status, note) {
  const run = caseRun, now = performance.now() - run.t0;
  if (status === 'working' && run.start[id] === undefined) run.start[id] = now;
  if ((status === 'done' || status === 'failed' || status === 'skipped') && run.end[id] === undefined) {
    if (run.start[id] === undefined) run.start[id] = now;
    run.end[id] = now;
  }
  run.status[id] = status;
  if (note) run.notes[id] = note;
  roomPaintNode(id);
  roomCard(id);
  roomEdges();
  if (note && status !== 'waiting') caseLog(note, agentName(id) + ':');
  const wait = $('case-out').querySelector('[data-sec="' + id + '"] .wait');
  if (wait && note && status === 'working') wait.textContent = note + '…';
}
function roomMini(id, items) { caseRun.mini[id] = items.filter(Boolean); roomPaintNode(id); roomCard(id); }
const plainWords = t => String(t || '').replace(/\[S[\d,\s\u2013-]*\]/g, '').replace(/^#+\s*/gm, '').replace(/[*_`>|]/g, '');
const wordCount = t => (String(t || '').match(/\S+/g) || []).length;
function roomTail(text, n) { const t = plainWords(text).replace(/\s+/g, ' ').trim(); return t.length > n ? '\u2026' + t.slice(t.length - n).replace(/^\S*\s/, '') : t; }
// The output wall: one card per agent, showing what it is producing while it produces it.
function roomCards() {
  const run = caseRun, wall = $('room-streams');
  const ids = (run.team || []).map(a => a.id).join(',');
  if (wall.dataset.ids !== ids) {
    wall.innerHTML = (run.team || []).map(a => '<div class="scard" data-card="' + a.id + '"><div class="sh"><i></i><b></b><span></span></div><div class="sbody idle"></div></div>').join('');
    (run.team || []).forEach(a => { wall.querySelector('[data-card="' + a.id + '"] b').textContent = a.name; });
    wall.dataset.ids = ids;
  }
  (run.team || []).forEach(a => roomCard(a.id));
}
function roomCard(id) {
  const run = caseRun, card = $('room-streams').querySelector('[data-card="' + id + '"]');
  if (!card) return;
  const status = run.status[id] || 'waiting', body = card.querySelector('.sbody'), result = run.results[id];
  card.dataset.status = status;
  let text = '', idle = false;
  if (result) text = plainWords(result.answer);
  else if (run.live[id]) text = plainWords(run.live[id]);
  else if ((run.mini[id] || []).length) text = run.mini[id].map(t => '\u00b7 ' + t).join('\n');
  else { text = run.notes[id] || (status === 'waiting' ? 'Waiting for its input' : ''); idle = true; }
  const atEnd = body.scrollHeight - body.scrollTop - body.clientHeight < 30;
  if (body.textContent !== text) body.textContent = text;
  body.className = 'sbody' + (idle ? ' idle' : '') + (run.live[id] && !result && status === 'working' ? ' live' : '');
  if (atEnd && !result) body.scrollTop = body.scrollHeight;
  const n = wordCount(result ? result.answer : run.live[id]);
  card.querySelector('.sh span').textContent = n ? n + ' words' : ({working: 'working', done: 'done', failed: 'failed', skipped: 'skipped'}[status] || '');
}
function roomWords(ev) {
  const run = caseRun;
  run.live[ev.id] = (ev.new ? '' : run.live[ev.id] || '') + ev.text;
  roomPaintNode(ev.id);
  roomCard(ev.id);
  // The same words fill the agent's section of the file below, until its finished answer replaces them.
  const sec = $('case-out').querySelector('[data-sec="' + ev.id + '"] .body');
  if (sec && !run.results[ev.id]) {
    let box = sec.querySelector('.livetext');
    if (!box) { sec.innerHTML = '<div class="livetext"></div>'; box = sec.firstChild; }
    box.textContent = plainWords(run.live[ev.id]);
  }
}
// The timeline under the graph: when each agent started and stopped, so overlapping (parallel) work is visible.
function roomTick() {
  const run = caseRun;
  if (!run) return;
  const now = performance.now() - run.t0, total = Math.max(1000, run.finished ? (run.total || now) : now);
  $('room-clock').textContent = caseClock(run.finished ? total : now);
  const g = $('room-gantt');
  if (run.team && run.team.length > 1) {
    if (g.dataset.n !== String(run.team.length)) {
      g.innerHTML = run.team.map(a => '<div class="gr" data-g="' + a.id + '"><span></span><div class="gt"><i></i></div><span></span></div>').join('');
      run.team.forEach(a => { g.querySelector('[data-g="' + a.id + '"] span').textContent = a.name; });
      g.dataset.n = String(run.team.length);
    }
    run.team.forEach(a => {
      const row = g.querySelector('[data-g="' + a.id + '"]'), bar = row.querySelector('i'), s0 = run.start[a.id];
      if (s0 === undefined) { bar.style.width = '0'; row.lastChild.textContent = ''; return; }
      const e = run.end[a.id] !== undefined ? run.end[a.id] : now;
      bar.style.left = (100 * s0 / total) + '%'; bar.style.width = (100 * Math.max(0, e - s0) / total) + '%';
      bar.className = (a.ai ? '' : 'rules') + (run.end[a.id] === undefined ? ' working' : '');
      row.lastChild.textContent = caseClock(e - s0);
      const t = $('case-team').querySelector('[data-agent="' + a.id + '"] .rt');
      const w = a.id === 'orchestrator' ? 0 : wordCount(run.results[a.id] ? run.results[a.id].answer : run.live[a.id]);
      if (t) t.textContent = (run.end[a.id] === undefined ? 'working ' : 'took ') + caseClock(e - s0) + (w ? ' \u00b7 ' + w + ' words' : '');
    });
  }
  if (!run.finished) requestAnimationFrame(() => setTimeout(roomTick, 90));
}
// ---- the case file ----
function caseSection(id, title) {
  const sec = el('div', 'csec');
  sec.dataset.sec = id;
  sec.innerHTML = '<div class="eyebrow"></div><div class="body"><p class="wait">Waiting</p></div>';
  sec.querySelector('.eyebrow').textContent = title;
  $('case-out').querySelector('.cfile').appendChild(sec);
}
function caseBody(id) { const b = $('case-out').querySelector('[data-sec="' + id + '"] .body'); if (b) b.innerHTML = ''; return b; }
function roomFileShell(kind, team) {
  $('case-out').innerHTML = '<div class="cfile"><div class="eyebrow">Built by the team</div><h2 class="display">' +
    ({situation: 'Case file', question: 'Research file', judgment: 'Judgment file', document: 'Document file', compare: 'Comparison file'}[kind]) + '</h2></div>';
  team.forEach(a => caseSection(a.id, kind === 'document' && a.id === 'reader' ? 'Reader agent · your document explained' : ROOM_SECTIONS[a.id] || a.name));
}
function roomBrief(ev) {
  const b = ev.brief || {}, body = caseBody('orchestrator');
  if (!body) return;
  if (ev.kind === 'judgment' || ev.kind === 'compare') {
    loadJudgments().then(() => {
      body.innerHTML = '<p>' + escapeHtml(ev.reason) + '</p><ul class="slist">' + (ev.doc_ids || []).map(d => {
        const j = (judgments || []).find(x => x.doc_id === d);
        return '<li><button type="button" class="jlink" data-doc="' + escapeHtml(d) + '"><span>Judgment</span><span>' +
          escapeHtml(j ? caseTitle(j) + ' · ' + caseNumber(j) : d) + '</span></button></li>'; }).join('') + '</ul>';
    });
    return;
  }
  const rows = [['Job', ev.reason], ['Issue', b.issue], ['Your side', b.side], ['Facts', (b.facts || []).join(' · ')],
                ['Provisions', (b.provisions || []).join(' · ')], ['Dates', (b.dates || []).join(' · ')], ['You want', b.goal],
                ['To draft', b.draft ? (DRAFTS.find(d => d.id === b.draft) || {}).title || b.draft : '']].filter(r => r[1]);
  body.innerHTML = '<div class="brief"><dl>' + rows.map(r => '<dt>' + r[0] + '</dt><dd dir="auto">' +
    (r[0] === 'Provisions' ? linkStatutes(escapeHtml(r[1])) : escapeHtml(r[1])) + '</dd>').join('') + '</dl></div>';
}
// Statute agent: every section in the provisions handed to it, or named by the writers, from the official text.
function roomStatutes() {
  const texts = [(caseRun.provisions || []).join(' . ')].concat(Object.values(caseRun.results).map(r => r.answer || ''));
  const keys = [...new Set(texts.flatMap(t => [...linkStatutes(escapeHtml(t)).matchAll(/data-stat="([^"]+)"/g)].map(m => m[1])))];
  if (caseRun.team && caseRun.team.some(a => a.id === 'statutes')) caseRun.mini.statutes = keys.map(k => statuteLabel(k, STATUTES[k]));
  const body = caseBody('statutes');
  if (!body) return keys;
  if (!keys.length) { body.innerHTML = '<p class="wait">No section in the statute set has been named yet.</p>'; return keys; }
  const ul = el('ul', 'slist');
  keys.forEach(k => {
    const v = STATUTES[k], li = el('li'), b = el('button');
    b.type = 'button';
    b.innerHTML = '<span></span><span></span>';
    b.firstChild.textContent = statuteLabel(k, v);
    b.lastChild.textContent = v.title;
    b.addEventListener('click', () => openStatute(k));
    li.appendChild(b); ul.appendChild(li);
  });
  body.appendChild(ul);
  return keys;
}
// Precedent agent: the judgments handed to it and the reported cases they rely on; for two, what they share.
async function roomPrecedents(docIds) {
  await Promise.all([loadJudgments(), citesReady]);
  const title = id => { const j = (judgments || []).find(x => x.doc_id === id); return j ? caseTitle(j) + ' · ' + caseNumber(j) : id; };
  const mine = docIds.map(d => precedentsOf(d).mine);
  let keys = [...new Set(mine.flat())];
  const pair = caseRun.kind === 'compare' && docIds.length === 2;   // only a comparison asks what the two share
  const both = pair ? mine[0].filter(k => mine[1].includes(k)) : [];
  keys.sort((a, b) => (both.includes(b) - both.includes(a)) || (Object.keys(CITES[b].refs).length - Object.keys(CITES[a].refs).length));
  const shared = keys.filter(k => Object.keys(CITES[k].refs).length > 1);
  const body = caseBody('precedents');
  body.innerHTML = '<ul class="slist">' + docIds.map(d => '<li><button type="button" class="jlink" data-doc="' + escapeHtml(d) + '"><span>Judgment</span><span>' +
    escapeHtml(title(d)) + '</span></button></li>').join('') + '</ul>' +
    (keys.length ? '<p class="privacy" style="margin-top:12px">' + (pair ? 'The two judgments share ' + plural(both.length, 'reported case') + '. ' : '') +
      'Together they rely on ' + plural(keys.length, 'reported case') + (shared.length ? ', ' + shared.length + ' of them relied on by more than one judgment in the library' : '') +
      ':</p><div class="cchips" style="color:var(--rock)">' + keys.slice(0, 16).map(k => '<button type="button" class="citeref" data-cite="' + escapeHtml(k) + '">' +
      escapeHtml(CITES[k].display) + (Object.keys(CITES[k].refs).length > 1 ? ' <i>' + Object.keys(CITES[k].refs).length + '</i>' : '') + '</button>').join(' · ') + '</div>'
     : '<p class="wait" style="margin-top:10px">These judgments cite no reported case we index.</p>');
  return pair ? 'The two share ' + plural(both.length, 'reported case') + '; ' + plural(keys.length, 'case') + ' in all'
                             : plural(docIds.length, 'judgment') + ' rely on ' + plural(keys.length, 'reported case') + (shared.length ? ', ' + shared.length + ' shared across the library' : '');
}
// Verifier: the fact check over everything the writers produced.
function roomVerify(final) {
  let checked = 0, missing = 0, docs = 0;
  Object.values(caseRun.results).forEach(r => {
    if (!r.grounded) return;
    const f = checkFacts(r.answer, r.sources || [], caseRun.userText);
    checked += f.checked; missing += f.missing.length; docs++;
  });
  const note = docs ? 'Checked ' + plural(checked, 'fact') + ' in ' + plural(docs, 'document') + ': ' + (checked - missing) + ' found in the passage cited' +
                      (missing ? ', ' + missing + ' to check (marked in amber)' : '') : 'Nothing to check yet';
  caseRun.meter = [checked - missing, checked];
  caseRun.mini.verify = docs ? [(checked - missing) + ' of ' + checked + ' facts found in their passages'] : [];
  caseSet('verify', final ? 'done' : 'working', note);
  const body = caseBody('verify');
  if (body) body.innerHTML = '<p>' + escapeHtml(note) + '. Every citation in this file opens its paragraph; the fact check looks for each name, section, number and quotation in it. No AI does the checking.</p>';
}
function roomSimilar(docIds) {
  loadJudgments().then(() => {
    const body = caseBody('similar');
    body.innerHTML = docIds.length ? '<ul class="slist">' + docIds.map(d => { const j = (judgments || []).find(x => x.doc_id === d);
      return '<li><button type="button" class="jlink" data-doc="' + escapeHtml(d) + '"><span>' + escapeHtml(j ? caseNumber(j) : d) + '</span><span>' +
        escapeHtml(j ? caseTitle(j) : '') + '</span></button></li>'; }).join('') + '</ul>' : '<p class="wait">None found.</p>';
  });
}
async function caseEvent(ev) {
  const run = caseRun;
  if (ev.type === 'team') {
    run.kind = ev.kind; run.team = ev.team; run.edges = ev.edges; run.brief = ev.brief || {};
    run.userText += '\n' + [run.brief.side, ...(run.brief.facts || []), run.brief.goal, ...(run.brief.dates || [])].join('\n');
    run.status.orchestrator = 'done';
    run.end.orchestrator = performance.now() - run.t0;
    const plan = $('room-plan');
    plan.hidden = false;
    plan.innerHTML = '<b>ORCHESTRATOR</b><span></span>';
    plan.lastChild.textContent = ev.reason + ' Deploying ' + plural(ev.team.length - 1, 'agent') + ': ' + ev.team.slice(1).map(a => a.name).join(', ') + '.';
    roomRoute(ev.kind);
    run.notes.orchestrator = 'Chose the ' + ev.kind + ' team: ' + (ev.team.length - 1) + ' agents';
    run.mini.orchestrator = [(run.brief.issue || ev.reason || '').slice(0, 90)];
    const b = run.brief;
    run.live.orchestrator = ['Job: ' + ev.reason, b.issue && 'Issue: ' + b.issue, b.side && 'Side: ' + b.side,
      ...(b.facts || []).map(f => 'Fact: ' + f), ...(b.provisions || []).map(x => 'Provision: ' + x), b.goal && 'Wants: ' + b.goal,
      b.draft && 'Draft: ' + b.draft, 'Team: ' + ev.team.slice(1).map(a => a.name).join(', ')].filter(Boolean).join('\n');
    roomDraw();
    if (ev.doc_ids && ev.doc_ids.length) roomNet.ready.then(() => netSay('Orchestrator', roomNet.light(ev.doc_ids.map(d => 'j:' + d)), ' picked the judgments to read'));
    roomFileShell(ev.kind, ev.team);
    roomBrief(ev);
    return;
  }
  if (ev.type === 'done') return caseFinish();
  if (ev.type === 'chunk') { if (Math.random() < .5) roomNet.spark(); return roomWords(ev); }
  if (ev.type !== 'agent') return;
  if (ev.from && ev.handoff) roomPulse(ev.from, ev.id, ev.handoff);
  caseSet(ev.id, ev.status, ev.note);
  if (ev.id === 'statutes' && ev.data) {
    run.provisions = ev.data.provisions || []; roomStatutes();
    roomNet.ready.then(() => netSay('Statute agent', roomNet.light(run.provisions.map(x => 'p:' + provisionKey(x)), {label: false}), ' found these sections across the judgments'));
  }
  if (ev.id === 'research' && ev.data && ev.data.doc_ids) {
    roomNet.ready.then(() => netSay('Research agent', roomNet.light(ev.data.doc_ids.map(d => 'j:' + d)), ' found ' + plural(ev.data.doc_ids.length, 'judgment') + ' on the issue'));
    await loadJudgments();
    roomMini('research', ev.data.doc_ids.map(d => { const j = (judgments || []).find(x => x.doc_id === d); return j ? caseTitle(j) : d; }));
  }
  if (ev.id === 'precedents' && ev.data && ev.data.doc_ids) {
    caseSet('precedents', 'done', await roomPrecedents(ev.data.doc_ids));
    const cited = [...new Set(ev.data.doc_ids.flatMap(d => precedentsOf(d).mine))];
    roomMini('precedents', cited.map(k => CITES[k].display));
    roomNet.ready.then(() => netSay('Precedent agent', roomNet.light(cited.map(k => 'c:' + k), {label: false}), ' traced the reported cases they rely on'));
  }
  if (ev.id === 'similar' && ev.status === 'done' && ev.data) {
    roomSimilar(ev.data.doc_ids || []);
    roomNet.ready.then(() => netSay('Similar-judgment agent', roomNet.light((ev.data.doc_ids || []).map(d => 'j:' + d)), ' found judgments on the same points'));
    await loadJudgments();
    roomMini('similar', (ev.data.doc_ids || []).map(d => { const j = (judgments || []).find(x => x.doc_id === d); return j ? caseTitle(j) : d; }));
  }
  if (ev.status === 'skipped') { const b = caseBody(ev.id); if (b) b.innerHTML = '<p class="wait">' + escapeHtml(ev.note) + '.</p>'; }
  if (ev.status === 'failed') { const b = caseBody(ev.id === 'orchestrator' ? 'orchestrator' : ev.id); if (b) { b.innerHTML = '<p class="error-text"></p>'; b.firstChild.textContent = ev.note; } }
  if (ev.status === 'done' && ev.result) {
    run.results[ev.id] = ev.result;
    roomCard(ev.id);
    // A glimpse of what the agent wrote, inside its node: its first headings or first sentence.
    const lines = String(ev.result.answer || '').split('\n').map(l => l.replace(/\[S[^\]]*\]|[#*_>|`]/g, '').trim()).filter(l => l.length > 3);
    const heads = String(ev.result.answer || '').split('\n').filter(l => /^#{1,3}\s/.test(l)).map(l => l.replace(/^#+\s*|[*_]/g, ''));
    roomMini(ev.id, heads.length >= 2 ? heads : [lines[0] || '']);
    renderResult(caseBody(ev.id), ev.result, 'room-' + ev.id + '-' + run.n, ev.result.restated_query || '',
                 {noSave: true, noHelp: true, whole: ev.id !== 'research', userText: run.userText});
    if (run.status.verify !== 'done' && (run.team || []).some(a => a.id === 'verify')) {
      roomPulse(ev.id, 'verify', {research: 'the law', strategy: 'the arguments', draft: 'the draft', counter: 'the limits', explainer: 'the explanation',
                                  reader: 'the summary', timeline: 'the dates', reader_a: 'a summary', reader_b: 'a summary', comparator: 'the comparison'}[ev.id] || 'its work');
      roomVerify(false);
    }
    const keys = roomStatutes();
    if (run.status.statutes !== 'done') caseSet('statutes', 'working', 'Found ' + plural(keys.length, 'section') + ' so far');
  }
}
function caseFinish() {
  const run = caseRun;
  if (!run || run.finished) return;
  run.total = performance.now() - run.t0;
  run.finished = true;
  setTimeout(roomTick, 0);
  if (run.team) {
    const keys = roomStatutes();
    if (run.team.some(a => a.id === 'statutes')) caseSet('statutes', 'done', keys.length ? 'Pulled ' + plural(keys.length, 'section') + ' word for word from the official text' : 'No section in the statute set was named');
    if (run.team.some(a => a.id === 'verify')) roomVerify(true);
    run.team.forEach(a => { if (run.status[a.id] === 'waiting' || run.status[a.id] === 'working') caseSet(a.id, 'skipped', run.notes[a.id] || 'Not needed'); });
    caseLog('Finished in ' + caseClock(performance.now() - run.t0) + '.', 'Team:');
    const items = Object.entries(run.results).map(([id, r]) => ({kind: id === 'draft' || id === 'strategy' ? 'draft' : 'ask',
      title: (ROOM_SECTIONS[id] || id).split(' · ')[1].replace(/^./, c => c.toUpperCase()) + ': ' + (run.brief.issue || $('case-text').value.slice(0, 80)),
      query: run.brief.issue || '', data: r}));
    if (items.length) {
      const tools = el('div', 'ctas');
      tools.style.margin = '6px 0 0';
      tools.innerHTML = '<button type="button" class="btn">Save it all to a matter</button><button type="button" class="btn line">Export as PDF</button>';
      tools.firstChild.addEventListener('click', () => chooseMatter(items));
      tools.lastChild.addEventListener('click', () => printResult($('case-out'), $('case-out').querySelector('h2').textContent));
      $('case-out').querySelector('.cfile').appendChild(tools);
      if (segValue('mode') === 'citizen') $('case-out').querySelector('.cfile').appendChild(helpStrip());
    }
  }
  $('case-go').disabled = false;
}
$('case-go').addEventListener('click', async () => {
  const situation = $('case-text').value.trim();
  if (!situation && !caseFile && !caseDocs.length) { $('case-text').focus(); return; }
  $('case-go').disabled = true;
  caseRun = {t0: performance.now(), n: ++seq, team: null, edges: [], status: {}, notes: {}, used: {}, results: {}, brief: {},
             provisions: [], finished: false, userText: [situation, caseFile && caseFile.text || ''].join('\n')};
  // Before the team exists, the graph is just the orchestrator.
  caseRun.team = [{id: 'orchestrator', name: 'Orchestrator', role: 'Reads the request, picks the team and briefs it', ai: true}];
  caseRun.start = {orchestrator: 0}; caseRun.end = {}; caseRun.mini = {}; caseRun.labels = {}; caseRun.meter = null; caseRun.live = {};
  $('room-streams').dataset.ids = '';
  caseRun.status.orchestrator = 'working'; caseRun.notes.orchestrator = 'Reading your request';
  roomNet.reset(); $('room-netcap').innerHTML = '';
  $('room-stage').hidden = false; $('room-plan').hidden = true; $('room-gantt').innerHTML = ''; $('room-gantt').dataset.n = '';
  roomRoute(null);
  $('case-feed').innerHTML = ''; $('case-log').hidden = false; $('case-out').innerHTML = '';
  roomDraw();
  roomTick();
  $('room-stage').scrollIntoView({behavior: 'smooth', block: 'start'});
  const body = JSON.stringify({situation, ...(caseFile || {}), doc_ids: caseDocs, mode: segValue('mode'), language: segValue('language')});
  try {
    const r = await fetch('/api/case/stream', {method: 'POST', headers: {'Content-Type': 'application/json'}, body});
    if (!r.ok) {
      const err = await r.json().catch(() => ({answer: 'The request was refused.'}));
      $('case-out').innerHTML = '<p class="error-text"></p>'; $('case-out').firstChild.textContent = err.answer;
      $('case-go').disabled = false; return;
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) await caseEvent(JSON.parse(line));
      }
    }
  } catch (e) {
    caseLog('Connection lost: ' + e.message, 'Team:');
  }
  caseFinish();
  caseDocs = [];
});

// Ask one judgment: the question is answered from that judgment's own paragraphs and nothing else.
let askScope = null;
function setAskScope(scope) {
  askScope = scope;
  const box = $('ask-scope');
  box.hidden = !scope;
  if (!scope) { $('input').placeholder = 'Ask about Pakistani case law, in English, اردو or Roman Urdu'; return; }
  box.innerHTML = '<span>Asking only: <b></b> <span style="color:var(--muted)"></span></span><button type="button">Ask everything again</button>';
  box.querySelector('b').textContent = scope.title;
  box.querySelector('span span').textContent = scope.no || '';
  box.querySelector('button').addEventListener('click', () => setAskScope(null));
  $('input').placeholder = 'Ask a question about this judgment';
  go('/ask');
  $('input').focus();
}
async function askScoped(turn, text) {
  turn.done('Scoped', 'Only ' + askScope.title);
  turn.pending('Reading', 'The whole judgment, ' + MODE_LABEL[segValue('mode')] + ' · ' + LANG_LABEL[segValue('language')]);
  const data = await watchedPost('/api/judgment/ask', {doc_id: askScope.doc_id, question: text, mode: segValue('mode'), language: segValue('language')},
                                 turn.el.querySelector('.body'));
  if (data.route !== 'ERROR') { turn.done('Written', 'Answer from this judgment, citing its paragraphs'); }
  onResult(turn, data);
}

async function ask(text) {
  if (busy || !text.trim()) return;
  if (pageFromPath(location.pathname) !== 'ask') go('/ask');
  busy = true; $('send').disabled = true;
  const turn = makeTurn(text);
  if (askScope) { await askScoped(turn, text); busy = false; $('send').disabled = false; $('input').focus(); return; }
  turn.pending('Understanding', '');
  const body = JSON.stringify({message: text, thread_id: thread, mode: segValue('mode'), language: segValue('language')});
  let got = false;
  try {
    const r = await fetch('/api/chat/stream', {method: 'POST', headers: {'Content-Type': 'application/json'}, body});
    if (!r.ok || !r.body) throw new Error('stream unavailable');
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'stage' || ev.type === 'chunk') {
          if (!turn.pipe) { turn.pipe = pipeView(turn.el.querySelector('.body')); turn.el.querySelector('details.trace').open = false; }
          turn.pipe.event(ev);
        }
        else if (ev.type === 'step') onStep(turn, ev);
        else if (ev.type === 'result') { got = true; if (turn.pipe) turn.pipe.finish(ev, text); onResult(turn, ev); }
      }
    }
    if (!got) throw new Error('stream ended early');
  } catch (err) {
    if (!got) {
      try { onResult(turn, await (await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body})).json()); }
      catch (e2) { onResult(turn, {route: 'ERROR', answer: 'Could not reach the server: ' + e2.message, sources: []}); }
    }
  }
  busy = false; $('send').disabled = false; $('input').focus();
}

$('form').addEventListener('submit', e => {
  e.preventDefault();
  const t = $('input').value;
  $('input').value = ''; $('input').style.height = 'auto';
  ask(t);
});
$('input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('form').requestSubmit(); } });
$('input').addEventListener('input', () => { const i = $('input'); i.style.height = 'auto'; i.style.height = Math.min(i.scrollHeight, 160) + 'px'; });

// ===========================================================================
// SUMMARIZE and COMPARE pages: a chosen judgment, no routing
// ===========================================================================
let sumDoc = null;
function renderSumList(q) {
  const ul = $('sum-list');
  ul.innerHTML = '';
  (judgments || []).filter(j => matches(j, q)).forEach(j => {
    const b = el('button', j.doc_id === sumDoc ? 'on' : '');
    b.type = 'button';
    b.innerHTML = '<b></b><span></span>';
    b.querySelector('b').textContent = caseTitle(j);
    b.querySelector('span').textContent = caseNumber(j) + ' · ' + plural(j.passages, 'passage');
    b.addEventListener('click', () => selectForSummary(j.doc_id));
    const li = el('li'); li.appendChild(b); ul.appendChild(li);
  });
}
$('sum-search').addEventListener('input', e => renderSumList(e.target.value));
function selectForSummary(docId) {
  loadJudgments().then(() => {
    const j = judgments.find(x => x.doc_id === docId);
    if (!j) return;
    sumDoc = docId;
    sumUpload = null;
    $('sum-court').textContent = j.court || 'Judgment';
    $('sum-title').textContent = caseTitle(j);
    $('sum-sub').textContent = caseNumber(j) + ' · ' + plural(j.passages, 'passage') + ((j.judges || []).length ? ' · ' + j.judges.join(', ') : '');
    $('sum-go').disabled = false;
    $('sum-result').innerHTML = ''; $('sum-status').textContent = '';
    renderSumList($('sum-search').value);
  });
}

async function runTask(url, payload, statusEl, resultEl, button, workingText, opts) {
  button.disabled = true;
  resultEl.innerHTML = '';
  statusEl.className = 'status working';
  const tick = startThinking(statusEl, THINKING[url] || [workingText, 'Reading the passages', 'Writing it up']);
  if (resultEl.previousElementSibling && resultEl.previousElementSibling.classList.contains('pipe')) resultEl.previousElementSibling.remove();
  const data = await watchedPost(url, {...payload, mode: segValue('mode'), language: segValue('language')}, resultEl,
                                 Object.values(payload.facts || {}).join('\n'));
  stopThinking(statusEl, tick);
  statusEl.className = 'status';
  statusEl.textContent = (data.route === 'ERROR' ? 'Stopped' : 'Done') + ' · ' +
    MODE_LABEL[segValue('mode')] + ' · ' + LANG_LABEL[segValue('language')];
  renderResult(resultEl, data, 't' + (++seq), data.restated_query || '', Object.assign({reveal: true}, opts || {whole: true}));
  button.disabled = false;
  return data;
}
let sumUpload = null;   // a judgment the user uploads instead of picking one
$('sum-go').addEventListener('click', () => {
  if (sumUpload) {
    runTask('/api/summarize', sumUpload, $('sum-status'), $('sum-result'), $('sum-go'),
            'Reading ' + sumUpload.filename, {whole: true, print: 'Brief: ' + sumUpload.filename});
    return;
  }
  const j = judgments.find(x => x.doc_id === sumDoc);
  runTask('/api/summarize', {doc_id: sumDoc}, $('sum-status'), $('sum-result'), $('sum-go'),
          'Reading ' + plural(j ? j.passages : 0, 'passage'), {whole: true, print: 'Brief: ' + (j ? caseTitle(j) : '')});
});
$('sum-file').addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f) return;
  if (f.size > 10 * 1024 * 1024) { $('sum-court').textContent = 'That file is larger than 10 MB'; return; }
  const r = new FileReader();
  r.onload = () => {
    sumUpload = {filename: f.name, mime: f.type || '', data: String(r.result).split(',')[1] || ''};
    sumDoc = null;
    renderSumList($('sum-search').value);
    $('sum-court').textContent = 'Uploaded judgment';
    $('sum-title').textContent = f.name.replace(/\.[^.]+$/, '');
    $('sum-sub').textContent = Math.max(1, Math.round(f.size / 1024)) + ' KB · read for this brief only, not saved';
    $('sum-go').disabled = false;
    $('sum-result').innerHTML = ''; $('sum-status').textContent = '';
  };
  r.readAsDataURL(f);
});

function renderCompareSelects() {
  ['cmp-a', 'cmp-b'].forEach((id, k) => {
    const sel = $(id), keep = sel.value;
    sel.innerHTML = '<option value="">Choose a judgment</option>' + (judgments || []).map(j =>
      '<option value="' + escapeHtml(j.doc_id) + '">' + escapeHtml(caseTitle(j) + ' — ' + caseNumber(j)) + '</option>').join('');
    if (keep) sel.value = keep;
  });
}
$('cmp-go').addEventListener('click', () => {
  const a = $('cmp-a').value, b = $('cmp-b').value;
  if (!a || !b || a === b) { $('cmp-status').textContent = 'Choose two different judgments'; return; }
  const name = id => { const j = (judgments || []).find(x => x.doc_id === id); return j ? caseTitle(j) : id; };
  runTask('/api/compare', {doc_a: a, doc_b: b}, $('cmp-status'), $('cmp-result'), $('cmp-go'), 'Reading both judgments',
          {whole: true, print: 'Comparison: ' + name(a) + ' vs ' + name(b)});
});

// ===========================================================================
// COMPARE page, second tab: review table (many judgments x chosen columns)
// ===========================================================================
document.querySelectorAll('#cmp-tabs button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('#cmp-tabs button').forEach(x => x.classList.toggle('on', x === b));
  $('cmp-two').hidden = b.dataset.tab !== 'two';
  $('cmp-table').hidden = b.dataset.tab !== 'table';
  if (b.dataset.tab === 'table') loadJudgments().then(() => { renderRtList($('rt-search').value); renderRtCols(); });
}));
const RT_PRESETS = ['Outcome', 'Legal question', 'Provisions applied', 'Relief or sentence', 'Key reasoning', 'Who succeeded'];
// Offered once the user adds their own documents (contracts, notices, FIRs).
const RT_DOC_PRESETS = ['Parties', 'Key dates and deadlines', 'Amounts', 'Obligations'];
const rtSelected = [];
let rtCols = ['Outcome', 'Legal question', 'Provisions applied'];
let rtCustom = [];
const rtFiles = [];   // the user's own documents: {filename, mime, data (base64), size}
function rtUpdate() {
  const rows = rtSelected.length + rtFiles.length;
  $('rt-count').textContent = 'Rows · ' + rows + ' of 8 chosen' + (rtFiles.length ? ' (' + plural(rtFiles.length, 'document') + ')' : '');
  $('rt-go').disabled = !rows || !rtCols.length;
}
function renderRtFiles(message) {
  const ul = $('rt-files');
  ul.innerHTML = '';
  rtFiles.forEach((f, i) => {
    const li = el('li');
    li.innerHTML = '<span></span><button type="button">Remove</button>';
    li.firstChild.textContent = f.filename + ' · ' + Math.max(1, Math.round(f.size / 1024)) + ' KB';
    li.lastChild.addEventListener('click', () => { rtFiles.splice(i, 1); renderRtFiles(); rtUpdate(); });
    ul.appendChild(li);
  });
  if (message) ul.appendChild(el('li', 'ferr', escapeHtml(message)));
  renderRtCols();
}
$('rt-file').addEventListener('change', e => {
  const picked = [...e.target.files];
  e.target.value = '';
  let message = '';
  picked.forEach(f => {
    const used = rtFiles.reduce((s, x) => s + x.size, 0);
    if (rtSelected.length + rtFiles.length >= 8) { message = 'A table has at most 8 rows.'; return; }
    if (used + f.size > 10 * 1024 * 1024) { message = 'Documents can be 10 MB in all; ' + f.name + ' was left out.'; return; }
    const entry = {filename: f.name, mime: f.type || '', data: '', size: f.size};
    rtFiles.push(entry);
    const r = new FileReader();
    r.onload = () => { entry.data = String(r.result).split(',')[1] || ''; };
    r.readAsDataURL(f);
  });
  renderRtFiles(message); rtUpdate();
});
function renderRtList(q) {
  const ul = $('rt-list');
  ul.innerHTML = '';
  const chosenFirst = (a, b) => rtSelected.includes(b.doc_id) - rtSelected.includes(a.doc_id);
  (judgments || []).filter(j => matches(j, q)).sort(chosenFirst).forEach(j => {
    const on = rtSelected.includes(j.doc_id);
    const b = el('button', on ? 'on' : '');
    b.type = 'button';
    b.innerHTML = '<b></b><span></span>';
    b.querySelector('b').textContent = (on ? '✓  ' : '') + caseTitle(j);
    b.querySelector('span').textContent = caseNumber(j) + ' · ' + plural(j.passages, 'passage');
    b.addEventListener('click', () => {
      const i = rtSelected.indexOf(j.doc_id);
      if (i >= 0) rtSelected.splice(i, 1); else if (rtSelected.length + rtFiles.length < 8) rtSelected.push(j.doc_id);
      renderRtList($('rt-search').value); rtUpdate();
    });
    const li = el('li'); li.appendChild(b); ul.appendChild(li);
  });
  rtUpdate();
}
$('rt-search').addEventListener('input', e => renderRtList(e.target.value));
function renderRtCols() {
  const box = $('rt-cols');
  box.innerHTML = '';
  RT_PRESETS.concat(rtFiles.length ? RT_DOC_PRESETS : [], rtCustom).forEach(c => {
    const b = el('button', rtCols.includes(c) ? 'on' : '');
    b.type = 'button';
    b.textContent = c;
    b.addEventListener('click', () => {
      const i = rtCols.indexOf(c);
      if (i >= 0) rtCols.splice(i, 1); else if (rtCols.length < 6) rtCols.push(c);
      renderRtCols(); rtUpdate();
    });
    box.appendChild(b);
  });
  rtUpdate();
}
$('rt-addcol').addEventListener('submit', e => {
  e.preventDefault();
  const v = $('rt-newcol').value.trim();
  if (!v) return;
  if (!rtCustom.includes(v) && !RT_PRESETS.includes(v)) rtCustom.push(v);
  if (!rtCols.includes(v) && rtCols.length < 6) rtCols.push(v);
  $('rt-newcol').value = '';
  renderRtCols();
});

function renderReviewTable(container, data, key, opts) {
  results[key] = {sources: data.sources || [], query: ''};
  const name = r => { if (r.name) return [r.name, 'Your document'];
    const j = (judgments || []).find(x => x.doc_id === r.doc_id); return j ? [caseTitle(j), caseNumber(j)] : [r.doc_id, '']; };
  const cite = n => n ? '<button type="button" class="cite" data-key="' + key + '" data-n="' + n + '">' + n + '</button>' : '';
  let h = '<div class="tablewrap"><table class="rtable"><thead><tr><th>' + (data.rows.some(r => r.name) ? 'Document' : 'Judgment') + '</th>' +
          data.columns.map(c => '<th dir="auto">' + escapeHtml(c) + '</th>').join('') + '</tr></thead><tbody>';
  data.rows.forEach(r => {
    const [t, no] = name(r);
    h += '<tr><td><b>' + escapeHtml(t) + '</b><span>' + escapeHtml(no) + '</span></td>' +
         r.cells.map(c => '<td dir="auto"' + (/^not stated$/i.test(c.text) ? ' class="ns"' : '') + '>' +
           linkStatutes(escapeHtml(c.text)) + cite(c.source) + '</td>').join('') + '</tr>';
  });
  container.innerHTML = h + '</tbody></table></div>';
  const cited = data.rows.reduce((s, r) => s + r.cells.filter(c => c.source).length, 0);
  const total = data.rows.length * data.columns.length;
  const g = el('div', 'ground');
  g.innerHTML = '<span><b>' + data.rows.length + ' × ' + data.columns.length + '</b></span><span>' + cited + ' of ' + total +
                ' cells cited to a paragraph</span>';
  const pdf = el('button', 'copy', 'Save as PDF');
  pdf.type = 'button';
  pdf.addEventListener('click', () => printResult(container, 'Review table'));
  g.appendChild(pdf);
  if (!opts || !opts.noSave) g.appendChild(saveButton(() => ({kind: 'table', title: 'Review table: ' + data.columns.join(', '), query: '', data})));
  container.appendChild(g);

  // Questions over the table, answered from the table alone and citing the same paragraphs.
  const box = el('div', 'followup');
  box.innerHTML = '<div class="eyebrow">Ask about this table</div>' +
    '<form class="fform"><input class="field" placeholder="For example: in which cases was bail granted, and why?" dir="auto">' +
    '<button class="btn" type="submit">Ask</button></form><div class="answers"></div>';
  container.appendChild(box);
  box.querySelector('form').addEventListener('submit', async e => {
    e.preventDefault();
    const input = box.querySelector('input'), q = input.value.trim();
    if (!q) return;
    input.value = '';
    const item = el('div', 'fitem');
    item.innerHTML = '<p class="q" dir="auto"></p><div class="status working">Reading the table</div>';
    item.querySelector('.q').textContent = q;
    box.querySelector('.answers').appendChild(item);
    const ans = await watchedPost('/api/table/ask', {question: q, table: data.answer, mode: segValue('mode'), language: segValue('language')},
                                  item.querySelector('.status'));
    item.querySelector('.status').remove();
    renderResult(item, {...ans, sources: data.sources || []}, key, q, {whole: true, reveal: true});
  });
}
$('rt-go').addEventListener('click', async () => {
  const st = $('rt-status'), out = $('rt-result');
  // Files are read in the browser first; wait for any still reading.
  for (let i = 0; i < 100 && rtFiles.some(f => !f.data); i++) await new Promise(r => setTimeout(r, 50));
  $('rt-go').disabled = true; out.innerHTML = '';
  st.className = 'status working';
  const what = [rtSelected.length ? plural(rtSelected.length, 'judgment') : '', rtFiles.length ? plural(rtFiles.length, 'document') : '']
    .filter(Boolean).join(' and ');
  const tick = startThinking(st, ['Reading ' + what].concat(THINKING['/api/table'].slice(1)));
  if (out.previousElementSibling && out.previousElementSibling.classList.contains('pipe')) out.previousElementSibling.remove();
  const data = await watchedPost('/api/table', {doc_ids: rtSelected, columns: rtCols, mode: segValue('mode'), language: segValue('language'),
                                                files: rtFiles.map(f => ({filename: f.filename, mime: f.mime, data: f.data}))}, out);
  stopThinking(st, tick);
  st.className = 'status';
  st.textContent = data.route === 'ERROR' ? 'Stopped' : 'Done';
  if (data.route === 'ERROR') out.innerHTML = '<p class="error-text">' + escapeHtml(data.answer) + '</p>';
  else renderReviewTable(out, data, 'rt' + (++seq));
  $('rt-go').disabled = false;
});

// ===========================================================================
// EXPLAIN page: the user's own document
// ===========================================================================
let docFile = null;   // {filename, mime, data (base64)}
function takeFile(f) {
  if (!f) return;
  if (f.size > 10 * 1024 * 1024) { $('doc-name').textContent = 'That file is larger than 10 MB'; docFile = null; return; }
  const r = new FileReader();
  r.onload = () => {
    docFile = {filename: f.name, mime: f.type || '', data: String(r.result).split(',')[1] || ''};
    $('doc-name').textContent = f.name + ' · ' + Math.max(1, Math.round(f.size / 1024)) + ' KB';
  };
  r.readAsDataURL(f);
}
$('doc-file').addEventListener('change', e => takeFile(e.target.files[0]));
['dragover', 'dragenter'].forEach(t => $('drop').addEventListener(t, e => { e.preventDefault(); $('drop').classList.add('over'); }));
['dragleave', 'drop'].forEach(t => $('drop').addEventListener(t, () => $('drop').classList.remove('over')));
$('drop').addEventListener('drop', e => { e.preventDefault(); takeFile(e.dataTransfer.files[0]); });
$('doc-go').addEventListener('click', async () => {
  const text = $('doc-text').value.trim();
  if (!docFile && !text) { $('doc-status').textContent = 'Choose a file or paste the text first'; return; }
  const data = await runTask('/api/explain', {...(docFile || {}), text}, $('doc-status'), $('doc-result'), $('doc-go'),
          docFile ? 'Reading ' + docFile.filename : 'Reading the text',
          {whole: false, print: 'Explained: ' + (docFile ? docFile.filename : 'my document')});
  if (data && data.route === 'DOC' && data.document_text) addFollowUp($('doc-result'), data.document_text);
});

// Follow-up questions about the uploaded document. Its text stays in this browser tab.
function addFollowUp(container, documentText) {
  const box = el('div', 'followup');
  box.innerHTML = '<div class="eyebrow">Ask about this document</div>' +
    '<form class="fform"><input class="field" placeholder="For example: what is the deadline in this notice?" dir="auto">' +
    '<button class="btn" type="submit">Ask</button></form><div class="answers"></div>';
  container.appendChild(box);
  box.querySelector('form').addEventListener('submit', async e => {
    e.preventDefault();
    const input = box.querySelector('input'), q = input.value.trim();
    if (!q) return;
    input.value = '';
    const item = el('div', 'fitem');
    item.innerHTML = '<p class="q" dir="auto"></p><div class="status working">Reading the document</div>';
    item.querySelector('.q').textContent = q;
    box.querySelector('.answers').appendChild(item);
    const data = await watchedPost('/api/explain/ask', {text: documentText, question: q, mode: segValue('mode'), language: segValue('language')},
                                   item.querySelector('.status'), documentText);
    item.querySelector('.status').remove();
    renderResult(item, data, 'f' + (++seq), q, {whole: true, reveal: true});
  });
}

// ===========================================================================
// shared: a picker list on the left, detail on the right
// ===========================================================================
function renderPicker(ul, items, current, onPick) {
  ul.innerHTML = '';
  items.forEach((it, i) => {
    const b = el('button', it.id === current ? 'on' : '');
    b.type = 'button';
    b.innerHTML = '<b></b><span></span>';
    b.querySelector('b').textContent = pad2(i + 1) + '  ' + it.title;
    b.querySelector('span').textContent = it.line || '';
    b.addEventListener('click', () => onPick(it.id));
    const li = el('li'); li.appendChild(b); ul.appendChild(li);
  });
}

// Save a result as PDF through the browser's print dialog, with a plain header.
function printResult(node, title) {
  const area = $('print-area');
  area.innerHTML = '<div class="eyebrow">Wakeel AI · ' + escapeHtml(new Date().toLocaleDateString()) + '</div>' +
                   '<h1 class="display">' + escapeHtml(title) + '</h1>';
  area.appendChild(node.cloneNode(true));
  area.querySelectorAll('button.copy, .followup').forEach(x => x.remove());
  document.body.classList.add('printing');
  window.print();
}
window.addEventListener('afterprint', () => document.body.classList.remove('printing'));

// ===========================================================================
// MATTERS: the user's saved work, grouped by matter. Kept in this browser (localStorage) only.
// ===========================================================================
const MATTERS_KEY = 'wakeel.matters';
function loadMatters() { try { return JSON.parse(localStorage.getItem(MATTERS_KEY)) || []; } catch (e) { return []; } }
function storeMatters(ms) { try { localStorage.setItem(MATTERS_KEY, JSON.stringify(ms)); return true; } catch (e) { return false; } }
function newId() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 6); }
function pageKind() { const m = document.body.className.match(/\bp-(\w+)/); return m ? m[1] : 'ask'; }
const KIND_LABEL = {moot: 'Practice hearing', evidence: 'Evidence check', check: 'Citation check', ask: 'Answer', summarize: 'Summary', compare: 'Comparison', table: 'Review table', explain: 'Document explained',
                    draft: 'Draft', guides: 'Guide', library: 'Answer'};
function saveButton(itemFn) {
  const b = el('button', 'copy', 'Save to matter');
  b.type = 'button';
  b.style.marginLeft = '0';
  b.addEventListener('click', () => chooseMatter(itemFn()));
  return b;
}
function chooseMatter(item) {
  const ms = loadMatters();
  openPanel('<div class="eyebrow">Save to a matter</div><h3 dir="auto"></h3>' +
    (ms.length ? '<ul class="plist mlist">' + ms.map(m => '<li><button type="button" data-m="' + m.id + '"></button></li>').join('') + '</ul>' : '') +
    '<form class="fform" id="m-pick-new" style="margin-top:18px"><input class="field" placeholder="New matter, e.g. Aslam v. Tariq (cheque)" maxlength="80">' +
    '<button class="btn" type="submit">Create and save</button></form><div class="note" id="m-saved" style="margin-top:18px" hidden></div>');
  const panel = $('panel');
  panel.querySelector('h3').textContent = Array.isArray(item) ? plural(item.length, 'item') + ' from the agent room' : item.title;
  ms.forEach(m => { const b = panel.querySelector('[data-m="' + m.id + '"]'); b.textContent = m.name;
                    b.insertAdjacentHTML('beforeend', '<span>' + plural(m.items.length, 'item') + '</span>');
                    b.addEventListener('click', () => addToMatter(m.id, null, item)); });
  panel.querySelector('#m-pick-new').addEventListener('submit', e => {
    e.preventDefault();
    const name = e.target.querySelector('input').value.trim();
    if (name) addToMatter(null, name, item);
  });
}
function addToMatter(id, name, item) {
  const ms = loadMatters();
  let m = ms.find(x => x.id === id);
  if (!m) { m = {id: newId(), name, items: [], created: Date.now()}; ms.unshift(m); }
  (Array.isArray(item) ? item : [item]).forEach(it => m.items.push({id: newId(), ...it, saved: Date.now()}));
  m.updated = Date.now();
  const ok = storeMatters(ms);
  const note = $('m-saved');
  if (!note) return;
  note.hidden = false;
  note.innerHTML = ok ? 'Saved to <b></b>. <a href="/matters?m=' + m.id + '" data-link>Open the matter</a>'
                      : 'This browser\'s storage is full. Remove an older matter and try again.';
  if (ok) note.querySelector('b').textContent = m.name;
}
function renderMatters(selected) {
  const ms = loadMatters();
  const current = ms.find(m => m.id === selected) || ms[0];
  const ul = $('m-list');
  ul.innerHTML = ms.length ? '' : '<li class="small" style="padding:12px 0;color:var(--muted)">No matters yet.</li>';
  ms.forEach(m => {
    const b = el('button', current && m.id === current.id ? 'on' : '');
    b.type = 'button';
    b.innerHTML = '<b></b><span></span>';
    b.querySelector('b').textContent = m.name;
    b.querySelector('span').textContent = plural(m.items.length, 'item') + ' · ' + new Date(m.updated || m.created).toLocaleDateString();
    b.addEventListener('click', () => { history.replaceState({}, '', '/matters?m=' + m.id); renderMatters(m.id); });
    const li = el('li'); li.appendChild(b); ul.appendChild(li);
  });
  const view = $('m-view');
  view.innerHTML = '';
  if (!current) return;
  const head = el('div', 'mhead');
  head.innerHTML = '<div class="eyebrow">Matter · ' + plural(current.items.length, 'item') + '</div><h2 class="display" dir="auto"></h2>' +
    '<div class="mtools"><button class="btn" type="button" data-a="pdf">Export as PDF</button>' +
    '<button class="btn line" type="button" data-a="rename">Rename</button><button class="btn line" type="button" data-a="del">Delete matter</button></div>';
  head.querySelector('h2').textContent = current.name;
  view.appendChild(head);
  const items = el('div');
  current.items.forEach(it => {
    const box = el('div', 'mitem');
    box.innerHTML = '<header><div class="eyebrow"></div><button class="copy" type="button">Remove</button></header><h3 dir="auto"></h3>';
    box.querySelector('.eyebrow').textContent = (KIND_LABEL[it.kind] || 'Saved') + ' · ' + new Date(it.saved).toLocaleString();
    box.querySelector('h3').textContent = it.title;
    const body = el('div');
    if (it.kind === 'table') renderReviewTable(body, it.data, 'm' + it.id, {noSave: true});
    else renderResult(body, it.data, 'm' + it.id, it.query, {noSave: true, whole: it.kind !== 'ask'});
    body.querySelectorAll('.followup').forEach(x => x.remove());
    box.appendChild(body);
    box.querySelector('header .copy').addEventListener('click', () => {
      current.items = current.items.filter(x => x.id !== it.id);
      current.updated = Date.now();
      storeMatters(ms); renderMatters(current.id);
    });
    items.appendChild(box);
  });
  if (!current.items.length) items.innerHTML = '<p class="privacy">Nothing saved yet. Press <b>Save to matter</b> under any answer.</p>';
  view.appendChild(items);
  head.querySelector('[data-a="pdf"]').addEventListener('click', () => printResult(items, current.name));
  head.querySelector('[data-a="rename"]').addEventListener('click', () => {
    const name = (prompt('Rename the matter', current.name) || '').trim();
    if (name) { current.name = name; storeMatters(ms); renderMatters(current.id); }
  });
  head.querySelector('[data-a="del"]').addEventListener('click', () => {
    if (!confirm('Delete "' + current.name + '" and everything saved in it?')) return;
    storeMatters(ms.filter(m => m.id !== current.id));
    history.replaceState({}, '', '/matters'); renderMatters(null);
  });
}
$('m-new').addEventListener('submit', e => {
  e.preventDefault();
  const name = $('m-name').value.trim();
  if (!name) return;
  const ms = loadMatters();
  const m = {id: newId(), name, items: [], created: Date.now(), updated: Date.now()};
  ms.unshift(m);
  storeMatters(ms);
  $('m-name').value = '';
  history.replaceState({}, '', '/matters?m=' + m.id);
  renderMatters(m.id);
});

// ===========================================================================
// Feedback, history and activity
// ===========================================================================
// Helpful / not helpful. Only the vote, the route and the fact-check counts are sent.
function feedbackButtons(data, facts) {
  const box = el('span', 'fb');
  box.innerHTML = '<button type="button" data-v="1">Helpful</button><button type="button" data-v="0">Not helpful</button>';
  box.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    fetch('/api/feedback', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({helpful: b.dataset.v === '1', route: data.route,
                            facts_checked: facts.checked || 0, facts_missing: (facts.missing || []).length})}).catch(() => {});
    box.textContent = 'Thanks';
  });
  return box;
}
// Ask history: questions and their answers, kept in this browser so they reopen with no new call.
const HISTORY_KEY = 'wakeel.history';
function loadHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch (e) { return []; } }
function remember(q, data) {
  const h = loadHistory().filter(x => x.q !== q);
  h.unshift({q, data, t: Date.now(), mode: segValue('mode'), language: segValue('language')});
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(0, 30))); } catch (e) {}
}
function historyHtml() {
  const h = loadHistory();
  if (!h.length) return '';
  return '<div class="hist"><div class="eyebrow" style="margin-bottom:10px">Your recent questions · kept in this browser</div>' +
    '<ul class="examples">' + h.slice(0, 8).map((x, i) => '<li><button type="button" data-hist="' + i + '"><span class="q" dir="auto">' +
    escapeHtml(x.q) + '</span><span>' + escapeHtml(new Date(x.t).toLocaleDateString()) + '</span></button></li>').join('') + '</ul></div>';
}
function reopen(i) {
  const x = loadHistory()[i];
  if (!x) return;
  $('thread').innerHTML = '';
  const box = el('div', 'turn');
  box.innerHTML = '<div class="eyebrow">Saved answer · ' + escapeHtml(new Date(x.t).toLocaleString()) + '</div><p class="q" dir="auto"></p><div class="body"></div>';
  box.querySelector('.q').textContent = x.q;
  $('thread').appendChild(box);
  renderResult(box.querySelector('.body'), x.data, 'h' + (++seq), x.data.restated_query || x.q);
}
// ===========================================================================
// Precedent map: judgments (circles) and the reported cases two or more of them rely on (squares).
// Built from the library's own citations, laid out in the browser: no model, no library, no network.
// ===========================================================================
function layoutGraph(nodes, edges, w, h, iters) {
  const k = Math.sqrt(w * h / Math.max(1, nodes.length)) * 0.62;
  nodes.forEach((n, i) => {   // a deterministic start, so the map looks the same every time
    const a = 2 * Math.PI * i / nodes.length;
    n.x = w / 2 + Math.cos(a) * w * 0.32; n.y = h / 2 + Math.sin(a) * h * 0.32;
  });
  for (let it = 0; it < iters; it++) {
    const t = 1 - it / iters;
    nodes.forEach(n => { n.dx = 0; n.dy = 0; });
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      let dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy) || 0.01;
      const f = k * k / d;
      a.dx += dx / d * f; a.dy += dy / d * f; b.dx -= dx / d * f; b.dy -= dy / d * f;
    }
    edges.forEach(e => {
      const a = nodes[e[0]], b = nodes[e[1]];
      const dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy) || 0.01;
      const f = d * d / k;
      a.dx -= dx / d * f; a.dy -= dy / d * f; b.dx += dx / d * f; b.dy += dy / d * f;
    });
    nodes.forEach(n => {
      n.dx -= (n.x - w / 2) * 0.35; n.dy -= (n.y - h / 2) * 0.35;   // gravity keeps clusters on the page
      const disp = Math.hypot(n.dx, n.dy) || 0.01, lim = 40 * t + 1;
      n.x += n.dx / disp * Math.min(disp, lim); n.y += n.dy / disp * Math.min(disp, lim);
      n.x = Math.max(44, Math.min(w - 44, n.x)); n.y = Math.max(30, Math.min(h - 30, n.y));
    });
  }
}
async function renderMap() {
  const box = $('lib-map');
  box.innerHTML = '<div class="status working">Drawing the map</div>';
  await Promise.all([loadJudgments(), citesReady]);
  const shared = Object.keys(CITES).filter(k => Object.keys(CITES[k].refs).length > 1);
  const docIds = [...new Set(shared.flatMap(k => Object.keys(CITES[k].refs)))];
  if (!shared.length) { box.innerHTML = '<p class="privacy">No reported case is relied on by two judgments yet.</p>'; return; }
  const nodes = [], index = {};
  docIds.forEach(d => { index['j' + d] = nodes.length; nodes.push({kind: 'j', id: d}); });
  shared.forEach(k => { index['p' + k] = nodes.length; nodes.push({kind: 'p', id: k}); });
  const edges = [];
  shared.forEach(k => Object.keys(CITES[k].refs).forEach(d => edges.push([index['p' + k], index['j' + d]])));
  // Each connected cluster is laid out in a box of its own, sized by how many nodes it has, and the boxes are
  // packed left to right in rows, so clusters neither crowd each other nor hug the edges of the page.
  const W = 1000, PAD = 14, PER_NODE = 12500;
  const parent = nodes.map((_, i) => i);
  const find = i => parent[i] === i ? i : (parent[i] = find(parent[i]));
  edges.forEach(e => { parent[find(e[0])] = find(e[1]); });
  const groups = {};
  nodes.forEach((_, i) => (groups[find(i)] = groups[find(i)] || []).push(i));
  const clusters = Object.values(groups).sort((a, b) => b.length - a.length);
  let x0 = PAD, y0 = PAD, rowH = 0, H = 0;
  clusters.forEach(ids => {
    const side = Math.max(190, Math.sqrt(ids.length * PER_NODE));
    if (x0 + side > W - PAD && x0 > PAD) { x0 = PAD; y0 += rowH + PAD; rowH = 0; }
    const local = ids.map(i => nodes[i]);
    const at = new Map(ids.map((i, k) => [i, k]));
    layoutGraph(local, edges.filter(e => at.has(e[0])).map(e => [at.get(e[0]), at.get(e[1])]), side, side, 320);
    local.forEach(n => { n.x += x0; n.y += y0; });
    x0 += side + PAD; rowH = Math.max(rowH, side); H = y0 + rowH + PAD;
  });
  const deg = nodes.map(() => 0);
  edges.forEach(e => { deg[e[0]]++; deg[e[1]]++; });
  const nameOf = id => { const j = (judgments || []).find(x => x.doc_id === id); return j ? caseTitle(j) : id; };
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Map of judgments and the reported cases they share');
  const mk = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([a, v]) => e.setAttribute(a, v)); (parent || svg).appendChild(e); return e; };
  const edgeEls = edges.map(e => mk('line', {class: 'g-edge', x1: nodes[e[0]].x, y1: nodes[e[0]].y, x2: nodes[e[1]].x, y2: nodes[e[1]].y}));
  const els = nodes.map((n, i) => {
    const g = mk('g', {tabindex: 0, role: 'button'});
    const r = 6 + Math.sqrt(deg[i]) * 2.4;
    let shape, label = '';
    if (n.kind === 'j') {
      shape = mk('circle', {class: 'g-j', cx: n.x, cy: n.y, r}, g);
      label = nameOf(n.id);
    } else {
      shape = mk('rect', {class: 'g-p', x: n.x - r, y: n.y - r, width: 2 * r, height: 2 * r, transform: 'rotate(45 ' + n.x + ' ' + n.y + ')'}, g);
      label = CITES[n.id].display;
      const t = mk('text', {class: 'g-lbl', x: n.x, y: i % 2 ? n.y + r + 14 : n.y - r - 6, 'text-anchor': 'middle'}, g);
      t.textContent = label;
    }
    mk('title', {}, g).textContent = label;
    n.label = label; n.g = g; n.shape = shape;
    return g;
  });
  box.innerHTML = '';
  const note = el('p', 'privacy');
  note.style.margin = '0 0 12px';
  note.textContent = plural(docIds.length, 'judgment') + ' and ' + plural(shared.length, 'shared reported case') +
    '. A square is a reported case cited by two or more judgments in the library; a circle is a judgment; a line means "cites". ' +
    'Judgments with no shared precedent (' + ((judgments || []).length - docIds.length) + ') are not drawn. Hover to trace a link, click to open it.';
  const cap = el('div', 'cap');
  cap.innerHTML = '<span>Hover a circle or a square.</span>';
  const key = el('div', 'key');
  key.innerHTML = '<span><i style="background:var(--deep);border-radius:50%"></i>Judgment</span>' +
                  '<span><i style="background:var(--sand);border:1.5px solid var(--slate);transform:rotate(45deg) scale(.8)"></i>Reported case relied on by 2 or more</span>';
  box.append(note, svg, cap, key);
  const neighbours = i => new Set(edges.flatMap(e => e[0] === i ? [e[1]] : e[1] === i ? [e[0]] : []));
  const focus = i => {
    const nb = neighbours(i);
    nodes.forEach((n, j) => n.g.classList.toggle('g-dim', j !== i && !nb.has(j)));
    edgeEls.forEach((l, k2) => { const on = edges[k2][0] === i || edges[k2][1] === i; l.classList.toggle('g-dim', !on); l.classList.toggle('g-on', on); });
    nodes[i].shape.classList.add('g-on');
    const n = nodes[i];
    cap.innerHTML = n.kind === 'j'
      ? '<b></b> <span>relies on ' + plural(nb.size, 'shared reported case') + ': ' + [...nb].map(j => escapeHtml(nodes[j].label)).join(' · ') + '</span>'
      : '<b></b> <span>relied on by ' + plural(nb.size, 'judgment') + ': ' + [...nb].map(j => escapeHtml(nodes[j].label)).join(' · ') + '</span>';
    cap.querySelector('b').textContent = n.label;
  };
  const blur = () => {
    nodes.forEach(n => { n.g.classList.remove('g-dim'); n.shape.classList.remove('g-on'); });
    edgeEls.forEach(l => l.classList.remove('g-dim', 'g-on'));
  };
  nodes.forEach((n, i) => {
    n.g.addEventListener('mouseenter', () => focus(i));
    n.g.addEventListener('focus', () => focus(i));
    n.g.addEventListener('mouseleave', blur);
    n.g.addEventListener('blur', blur);
    const open = () => n.kind === 'j' ? openJudgment(n.id) : openCitation(n.id);
    n.g.addEventListener('click', open);
    n.g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  });
  $('lib-sum').textContent = 'Which judgments rely on the same reported cases, drawn from the library\'s own citations. No AI involved.';
}

// Activity: what has been used and how it went. The server never logs the content.
async function renderActivity() {
  const box = $('lib-activity');
  box.innerHTML = '<div class="status working">Counting</div>';
  let d;
  try { d = await (await fetch('/api/activity')).json(); }
  catch (e) { box.innerHTML = '<p class="error-text">Could not load activity.</p>'; return; }
  box.innerHTML = '';
  if (d.off || !d.requests) {
    box.innerHTML = '<p class="privacy">' + (d.off ? 'The usage log is off (--no-activity).' : 'Nothing used yet. Ask a question and come back.') + '</p>';
    return;
  }
  const votes = d.helpful + d.not_helpful;
  const kpis = el('div', 'kpis');
  kpis.innerHTML =
    '<div><b>' + d.requests + '</b><span>requests answered</span></div>' +
    '<div><b>' + Math.round(100 * d.ok / d.requests) + '%</b><span>completed without an error</span></div>' +
    '<div><b>' + (votes ? Math.round(100 * d.helpful / votes) + '%' : '–') + '</b><span>rated helpful' + (votes ? ' (' + plural(votes, 'vote') + ')' : '') + '</span></div>' +
    '<div><b>' + (d.facts_checked ? Math.round(100 * (d.facts_checked - d.facts_missing) / d.facts_checked) + '%' : '–') +
      '</b><span>of checked facts found in their source</span></div>';
  box.append(kpis,
    barList('Requests by feature', 'Since ' + new Date(d.since * 1000).toLocaleDateString() + '.', d.features, {unit: 'request'}),
    barList('Typical response time', 'Median seconds per feature, successful requests.', d.median_seconds, {unit: 'second'}));
  $('lib-sum').textContent = 'How Wakeel AI has been used on this server. Only the feature, timing and votes are logged, never the questions or documents.';
}

// ===========================================================================
// DRAFT page
// ===========================================================================
const DRAFTS = [
  {id: 'reply_to_notice', title: 'Reply to a legal notice', line: 'Answer a notice you received', fields: [
    ['Your name'], ['Sender of the notice'], ['Date of the notice'],
    ['What the notice demands', true], ['Your position', true]]},
  {id: 'bail_application', title: 'Post-arrest bail application', line: 'Section 497 Cr.P.C.', fields: [
    ['Court'], ['Applicant (accused)'], ['FIR number and date'], ['Police station'],
    ['Offences (sections)'], ['Grounds for bail', true]]},
  {id: 'cheque_notice', title: 'Legal notice for a bounced cheque', line: 'Demand payment before going to court', fields: [
    ['Your name'], ['Recipient'], ['Amount'], ['Cheque number, bank and date'],
    ['How the money became due', true], ['Days given to pay']]},
  {id: 'fir_application', title: 'Application to register an FIR', line: 'To the SHO, section 154 Cr.P.C.', fields: [
    ['Your name and CNIC'], ['Police station'], ['Date and place of the incident'],
    ['What happened', true], ['Accused, if known']]},
  {id: 'superdari', title: 'Release of a seized vehicle', line: 'Superdari application to the Magistrate', fields: [
    ['Court'], ['Your name'], ['Vehicle (make, model, registration number)'],
    ['Police station and date of seizure'], ['Ownership documents'], ['Why it should be released', true]]},
  {id: 'strategy', title: 'Case strategy memo', line: 'Arguments for and against, from similar judgments',
   endpoint: '/api/strategy', fields: [
    ['Your side (e.g. accused, petitioner, tenant)'], ['What happened', true], ['What you want the court to do', true],
    ['What the other side says, if known', true]]},
  {id: 'hearing', title: 'Hearing preparation', line: 'Questions the bench may ask, with answers', endpoint: '/api/workflow', fields: [
    ['Your side (e.g. accused, petitioner, mother)'], ['What the hearing is for (e.g. bail, interim custody, stay)'],
    ['The facts', true], ['What the other side says, if known', true]]},
  {id: 'chronology', title: 'Chronology', line: 'A dated timeline from your documents', endpoint: '/api/workflow', fields: [
    ['Paste your documents or tell the story (FIR, notices, orders, letters)', true], ['What the timeline is for, if anything', true]]},
  {id: 'opponent', title: 'Argue against my draft', line: 'Opposing counsel finds the weak points and the judgments against you', endpoint: '/api/workflow', fields: [
    ['Your draft or argument (paste it)', true], ['Your side (e.g. accused, petitioner, tenant)'], ['What the other side says, if known', true]]},
  {id: 'client_letter', title: 'Client update letter', line: 'Explain a hearing to a client in plain words', endpoint: '/api/workflow', fields: [
    ['Client name'], ['Case'], ['What happened at the hearing', true], ['Next date and what the client must do', true]]},
];
let draftKind = null;
function selectDraft(id) {
  const d = DRAFTS.find(x => x.id === id);
  if (!d) return;
  draftKind = id;
  renderPicker($('draft-kinds'), DRAFTS, id, selectDraft);
  $('draft-form').innerHTML = '<div class="fillbar"><button type="button" class="btn line" data-fill="sample">Fill with a sample</button>' +
    '<button type="button" class="btn line" data-fill="doc">Fill from a document</button><div class="fillbox" hidden>' +
    '<textarea class="field" rows="5" dir="auto" placeholder="Paste the FIR, notice, order or your own account here"></textarea>' +
    '<div class="ctas"><label class="btn line vf-add">Or choose a file<input type="file" accept=".pdf,.txt,.md,image/*" hidden></label>' +
    '<button type="button" class="btn" data-fill="go">Fill the form</button><span class="fillnote"></span></div></div></div>' + d.fields.map((f, i) =>
    '<label class="eyebrow" for="df' + i + '">' + escapeHtml(f[0]) + '</label>' +
    (f[1] ? '<textarea class="field" id="df' + i + '" rows="3" dir="auto"></textarea>' +
            '<div class="microw"><button class="mic" type="button" data-for="df' + i + '" hidden></button></div>'
          : '<input class="field" id="df' + i + '" dir="auto">')).join('');
  $('draft-form').querySelectorAll('.mic').forEach(b => attachMic(b, $(b.dataset.for)));
  $('draft-style').hidden = !!d.endpoint;   // the strategy memo is research, not a document to style
  $('draft-go').disabled = false;
  $('draft-go').textContent = d.endpoint ? 'Run' : 'Draft';
  $('draft-result').innerHTML = ''; $('draft-status').textContent = '';
}
// Fill the form: a worked sample (the same facts the demo cache holds, so it drafts at once), or the user's own
// document read into the fields, leaving blank whatever the document does not say.
const DRAFT_SAMPLES = {
  reply_to_notice: ['Tariq Mehmood', 'Muhammad Aslam, through his counsel', '12 March 2024', 'Payment of Rs. 900,000 within 15 days, on a cheque that was dishonoured',
                    'The cheque was handed over as security for a business loan, not as repayment; part of the loan was repaid in cash'],
  bail_application: ['Court of Sessions, Lahore', 'Tariq Mehmood', 'FIR No. 214/2024 dated 3 March 2024', 'Police Station Gulberg, Lahore', 'Section 489-F PPC',
                     'The dispute is a civil one over a loan; the offence is not within the prohibitory clause; the accused has no previous record'],
  cheque_notice: ['Muhammad Aslam', 'Tariq Mehmood', 'Rs. 900,000', 'Cheque No. 00771245, Bank Alfalah, dated 01.02.2024',
                  'Loan of Rs. 900,000 given in January 2024, to be repaid by cheque', '15'],
  fir_application: ['Muhammad Aslam, CNIC 35202-1234567-1', 'Police Station Gulberg, Lahore', '1 March 2024, Bank Alfalah, Gulberg branch',
                    'Tariq Mehmood gave me cheque No. 00771245 for Rs. 900,000 to repay a loan. It was dishonoured for insufficient funds and he refuses to pay.',
                    'Tariq Mehmood, Gulberg III, Lahore'],
  superdari: ['Judicial Magistrate Section 30, Lahore', 'Ali Raza', 'Honda CD-70, 2021, registration LEB-4521', 'Police Station Gulberg, Lahore, 2 October 2026',
              'Registration book and purchase receipt in my name', 'I am the registered owner; the motorbike is not case property in any offence, and I need it for work'],
  strategy: ['Complainant', 'Bail was granted on a compromise: the accused promised to pay Rs. 900,000 in three instalments and paid nothing. The cheque case under section 489-F PPC is pending.',
             'Cancel the bail', 'He says the instalments were never due until the cheque case ends.'],
  hearing: ['Complainant', 'Application to cancel pre-arrest bail',
            'Bail was granted on a compromise: the accused promised to pay Rs. 900,000 in three instalments and paid nothing. The cheque case under section 489-F PPC is pending.',
            'He says the instalments were never due until the cheque case ends.'],
  client_letter: ['Tariq Mehmood', 'State v. Tariq Mehmood, FIR 214/2024, Section 489-F PPC', 'Bail arguments were heard; the court asked for the bank record of the loan and adjourned the matter',
                  '18 November 2026; bring the bank statements and the loan agreement'],
  opponent: [null, 'Accused (petitioner)', ''],
};
$('draft-form').addEventListener('click', async e => {
  const b = e.target.closest('[data-fill]');
  if (!b) return;
  const d = DRAFTS.find(x => x.id === draftKind), box = $('draft-form').querySelector('.fillbox');
  if (!d) return;
  if (b.dataset.fill === 'sample') {
    const vals = DRAFT_SAMPLES[d.id] || [];
    d.fields.forEach((f, i) => { const v = d.id === 'opponent' && i === 0 ? SAMPLE_PETITION : d.id === 'chronology' && i === 0 ? TOUR_FIR : vals[i]; if ($('df' + i) && v != null) $('df' + i).value = v; });
    box.hidden = true;
    return;
  }
  if (b.dataset.fill === 'doc') { box.hidden = !box.hidden; if (!box.hidden) box.querySelector('textarea').focus(); return; }
  const text = box.querySelector('textarea').value.trim(), note = box.querySelector('.fillnote');
  if (!text) { box.querySelector('textarea').focus(); return; }
  b.disabled = true; note.textContent = 'Reading the document…';
  let r;
  try { r = await (await fetch('/api/fill', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                              body: JSON.stringify({labels: d.fields.map(f => f[0]), text})})).json(); }
  catch (err) { r = {route: 'ERROR', answer: err.message}; }
  b.disabled = false;
  if (r.route === 'ERROR') { note.textContent = r.answer || 'Could not read it.'; return; }
  let n = 0;
  d.fields.forEach((f, i) => { const v = (r.values || {})[f[0]]; if (v && $('df' + i)) { $('df' + i).value = v; n++; } });
  note.textContent = 'Filled ' + n + ' of ' + d.fields.length + ' fields from the document. The rest it does not say: fill them in.';
});
$('draft-form').addEventListener('change', async e => {
  const inp = e.target.closest('.fillbox input[type="file"]');
  if (!inp || !inp.files[0]) return;
  const f = inp.files[0], box = inp.closest('.fillbox'), note = box.querySelector('.fillnote');
  note.textContent = 'Reading ' + f.name + '…';
  const isText = /\.(txt|md)$/i.test(f.name) || f.type.startsWith('text/');
  const body = isText ? {filename: f.name, text: await f.text()}
    : {filename: f.name, mime: f.type || '', data: await new Promise(res => { const rd = new FileReader(); rd.onload = () => res(String(rd.result).split(',')[1] || ''); rd.readAsDataURL(f); })};
  let d;
  try { d = await (await fetch('/api/extract', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})).json(); }
  catch (err) { d = {route: 'ERROR', answer: err.message}; }
  inp.value = '';
  if (d.route === 'ERROR') { note.textContent = d.answer; return; }
  box.querySelector('textarea').value = d.text;
  note.textContent = f.name + ' read. Press Fill the form.';
});
// The chronology workflow over one library judgment, opened from its panel.
async function timelineOf(docId, title) {
  go('/draft');
  selectDraft('chronology');
  await runTask('/api/workflow', {kind: 'chronology', doc_id: docId, facts: {}}, $('draft-status'), $('draft-result'),
                $('draft-go'), 'Reading ' + title + ' for its dates', {whole: true, print: 'Timeline: ' + title});
}
$('draft-go').addEventListener('click', async () => {
  const d = DRAFTS.find(x => x.id === draftKind);
  const facts = {};
  d.fields.forEach((f, i) => { const v = $('df' + i).value.trim(); if (v) facts[f[0]] = v; });
  const sample = d.endpoint ? '' : $('draft-sample').value.trim();
  const data = await runTask(d.endpoint || '/api/draft', {kind: draftKind, facts, ...(sample ? {sample} : {})},
                             $('draft-status'), $('draft-result'),
                             $('draft-go'), (d.endpoint ? 'Researching and writing your ' : 'Drafting ') + d.title.toLowerCase(),
                             {whole: false, print: d.title, userText: Object.values(facts).join('\n')});
  const md = $('draft-result').querySelector('.md');
  if (md && data.route !== 'ERROR' && !d.endpoint) { md.contentEditable = 'true'; md.title = 'Click to edit the draft'; md.classList.add('editable'); }
});

// ===========================================================================
// GUIDES page: know your rights. Steps are general information; the judgments are real.
// ===========================================================================
const GUIDES = [
  {id: 'cheque', title: 'My cheque bounced', line: 'Section 489-F PPC',
   steps: [
     'Keep the original cheque and the bank\'s return memo, which gives the reason it was not paid.',
     'Send the person who issued it a written demand (a legal notice) giving a fixed time to pay, and keep proof that it was delivered.',
     'Dishonestly issuing a cheque that bounces is an offence under section 489-F PPC. You can apply at the police station to register an FIR.',
     'Separately, you can file a civil suit to recover the money.',
     'Settlements are common. In Azeem-ud-Din v. Feroze Khan the High Court cancelled an accused\'s bail when he broke a payment compromise.'],
   laws: ['Section 489-F PPC', 'Section 154 Cr.P.C. (registering an FIR)'],
   judgments: ['lhc2023_580', 'lhc2023_466'], draft: 'cheque_notice',
   ask: ['What happens when a cheque given to repay a loan bounces?', 'en']},
  {id: 'vehicle', title: 'Police seized my vehicle', line: 'Section 550 Cr.P.C.',
   steps: [
     'Ask for a copy of the police record of the seizure, showing when and why the vehicle was taken.',
     'Collect proof of ownership: the registration book, purchase documents and your CNIC.',
     'Apply to the Magistrate for release of the vehicle on superdari, which means custody until the case ends.',
     'If the Magistrate refuses, a revision can be filed in the Sessions Court. In Muhammad Akram v. The State the High Court ordered a car handed to its registered owner on a surety bond.'],
   laws: ['Section 550 Cr.P.C. (police power to seize property)'],
   judgments: ['lhc2023_570'], draft: 'superdari',
   ask: ['اگر پولیس نے میری گاڑی قبضے میں لے لی ہو تو میں اسے واپس کیسے حاصل کر سکتا ہوں؟', 'ur']},
  {id: 'custody', title: 'Custody of my child', line: 'Guardians and Wards Act, 1890',
   steps: [
     'Custody and guardianship cases are heard by the Guardian Court; the Family Court acts as one. File where the child ordinarily lives.',
     'File a petition under the Guardians and Wards Act, 1890. While the case is pending you can ask for interim custody under section 12.',
     'The court decides by the welfare of the child, which comes before the wishes of either parent.',
     'Keep the child\'s birth certificate and school and medical records.'],
   laws: ['Section 12, Guardians and Wards Act, 1890 (interim orders)', 'Section 25, Guardians and Wards Act, 1890 (custody)'],
   judgments: ['lhc2023_434', 'lhc2023_354'],
   ask: ['Can a mother get interim custody of a young child while the case is still pending?', 'en']},
  {id: 'bail', title: 'Someone I know was arrested', line: 'Bail, sections 497 and 498 Cr.P.C.',
   steps: [
     'After arrest, a bail application can be made under section 497 Cr.P.C., first to the trial court and then to the High Court.',
     'Before an arrest, pre-arrest bail can be sought under section 498 Cr.P.C.',
     'Courts look at the offence, the evidence and the accused\'s circumstances. Children are dealt with under the Juvenile Justice System Act, 2018.',
     'Bail can later be cancelled, for example if its conditions are broken (section 497(5) Cr.P.C.).'],
   laws: ['Section 497 Cr.P.C. (bail)', 'Section 498 Cr.P.C. (pre-arrest bail)', 'Juvenile Justice System Act, 2018'],
   judgments: ['lhc2023_472', 'lhc2023_466'], draft: 'bail_application',
   ask: ['Can the High Court cancel pre-arrest bail granted on the basis of a compromise?', 'en']},
  {id: 'eviction', title: 'My landlord filed for eviction', line: 'Punjab Rented Premises Act, 2009',
   steps: [
     'Eviction cases under the Punjab Rented Premises Act, 2009 are heard by the Rent Tribunal.',
     'To defend the case you must apply for leave to contest within 10 days of learning of it (sections 21(4) and 22). Without leave you may not be allowed to defend.',
     'If the Tribunal started ex-parte proceedings, apply promptly to set them aside, with your reasons.',
     'Keep your rent receipts and the tenancy agreement.'],
   laws: ['Sections 21 and 22, Punjab Rented Premises Act, 2009'],
   judgments: ['lhc2023_288', 'lhc2023_330'],
   ask: ['What should a tenant do after receiving an eviction petition from the Rent Tribunal?', 'en']},
];
let guideId = null;
function selectGuide(id) {
  const g = GUIDES.find(x => x.id === id);
  if (!g) return;
  guideId = id;
  renderPicker($('guide-list'), GUIDES, id, selectGuide);
  loadJudgments().then(() => {
    const js = g.judgments.map(d => (judgments || []).find(j => j.doc_id === d)).filter(Boolean);
    $('guide').innerHTML =
      '<div class="eyebrow">' + escapeHtml(g.line) + '</div><h2 class="display">' + escapeHtml(g.title) + '</h2>' +
      '<h3>What to do</h3><ol class="gsteps">' + g.steps.map(s => '<li>' + escapeHtml(s) + '</li>').join('') + '</ol>' +
      '<h3>Laws involved</h3><ul class="glaws">' + g.laws.map(s => '<li>' + linkStatutes(escapeHtml(s)) + '</li>').join('') + '</ul>' +
      '<h3>Judgments in the library</h3><ul class="srclist">' + js.map(j =>
        '<li class="cited"><button class="who" type="button" data-doc="' + escapeHtml(j.doc_id) + '">' + escapeHtml(caseTitle(j)) +
        '</button><span class="cno">' + escapeHtml(caseNumber(j)) + '</span></li>').join('') + '</ul>' +
      '<div class="ctas"><button class="btn" type="button" data-act="ask">Ask Wakeel about this</button>' +
      (g.draft ? '<button class="btn line" type="button" data-act="draft">Draft the document</button>' : '') +
      '<button class="btn line" type="button" data-act="explain">Explain my document</button></div>' +
      '<p class="privacy">General information, not legal advice. Laws and procedure change; check with a lawyer.</p>';
    $('guide').querySelectorAll('[data-doc]').forEach(b => b.addEventListener('click', () => openJudgment(b.dataset.doc)));
    $('guide').querySelector('[data-act="ask"]').addEventListener('click', () => { setSeg('language', g.ask[1]); ask(g.ask[0]); });
    const dr = $('guide').querySelector('[data-act="draft"]');
    if (dr) dr.addEventListener('click', () => { go('/draft'); selectDraft(g.draft); });
    $('guide').querySelector('[data-act="explain"]').addEventListener('click', () => go('/explain'));
  });
}

// ===========================================================================
// CALCULATE page: deterministic, no AI
// ===========================================================================
// Limitation Act, 1908, First Schedule. Wording copied from the Act; review before relying on it.
const LIMITATION = [
  ['Suits', 22, 'For compensation for any other injury to the person', [1, 'y'], 'When the injury is committed'],
  ['Suits', 57, 'For money payable for money lent', [3, 'y'], 'When the loan is made'],
  ['Suits', 73, 'On a bill of exchange or promissory note payable on demand and not accompanied by any writing restraining or postponing the right to sue', [3, 'y'], 'The date of the bill or note'],
  ['Suits', 91, 'To cancel or set aside an instrument not otherwise provided for', [3, 'y'], 'When the facts entitling the plaintiff to have the instrument cancelled or set aside become known to him'],
  ['Suits', 113, 'For specific performance of a contract', [3, 'y'], 'The date fixed for the performance, or if no such date is fixed, when the plaintiff has notice that performance is refused'],
  ['Suits', 115, 'For compensation for the breach of any contract, express or implied, not in writing registered and not herein specially provided for', [3, 'y'], 'When the contract is broken'],
  ['Suits', 116, 'For compensation for the breach of a contract in writing registered', [6, 'y'], 'When the period of limitation would begin to run against a suit brought on a similar contract not registered'],
  ['Suits', 120, 'Suit for which no period of limitation is provided elsewhere in this schedule', [6, 'y'], 'When the right to sue accrues'],
  ['Suits', 142, 'For possession of immovable property when the plaintiff, while in possession of the property, has been dispossessed or has discontinued the possession', [12, 'y'], 'The date of the dispossession or discontinuance'],
  ['Appeals', 151, 'From a decree or order of a High Court in the exercise of its original jurisdiction', [20, 'd'], 'The date of the decree or order'],
  ['Appeals', 152, 'Under the Code of Civil Procedure, 1908, to the Court of a District Judge', [30, 'd'], 'The date of the decree or order appealed from'],
  ['Appeals', 154, 'Under the Code of Criminal Procedure, 1898, to any Court other than a High Court', [30, 'd'], 'The date of the sentence or order appealed from'],
  ['Appeals', 155, 'Under the same Code to a High Court, except in the cases provided for by Article 150 and Article 157', [60, 'd'], 'The date of the sentence or order appealed from'],
  ['Appeals', 156, 'Under the Code of Civil Procedure, 1908, to a High Court, except in the cases provided for by Article 151 and Article 153', [90, 'd'], 'The date of the decree or order appealed from'],
  ['Appeals', 157, 'Under the Code of Criminal Procedure, 1898, from an order of acquittal', [6, 'm'], 'The date of the order appealed from'],
  ['Applications', 162, 'For a review of judgment by a High Court in the exercise of its original jurisdiction', [20, 'd'], 'The date of the decree or order'],
  ['Applications', 163, 'By a plaintiff for an order to set aside a dismissal for default of appearance or for failure to pay costs of service of process or to furnish security for costs', [30, 'd'], 'The date of the dismissal'],
  ['Applications', 181, 'Applications for which no period of limitation is provided elsewhere in this schedule or by section 48 of the Code of Civil Procedure, 1908', [3, 'y'], 'When the right to apply accrues'],
];
const UNIT = {y: ['year', 'years'], m: ['month', 'months'], d: ['day', 'days']};

function addPeriod(date, n, unit) {
  const d = new Date(date.getTime());
  if (unit === 'd') d.setDate(d.getDate() + n);
  else {
    const day = d.getDate();
    d.setDate(1);
    d.setMonth(d.getMonth() + (unit === 'y' ? 12 * n : n));
    const last = new Date(d.getFullYear(), d.getMonth() + 1, 0).getDate();
    d.setDate(Math.min(day, last));
  }
  return d;
}
function limitationDeadline(startISO, article) {
  const row = LIMITATION.find(r => r[1] === article);
  const start = new Date(startISO + 'T00:00:00');
  // Section 12(1): the day from which the period is reckoned is excluded, so the period ends
  // on the same calendar day n days / months / years later.
  return {row, deadline: addPeriod(start, row[3][0], row[3][1])};
}

// Hanafi (Sunni) inheritance among the common heirs. Exact fractions throughout.
function gcd(a, b) { a = Math.abs(a); b = Math.abs(b); while (b) [a, b] = [b, a % b]; return a || 1; }
function F(n, d) { if (d === undefined) d = 1; const g = gcd(n, d); return [n / g, d / g]; }
const fadd = (a, b) => F(a[0] * b[1] + b[0] * a[1], a[1] * b[1]);
const fsub = (a, b) => F(a[0] * b[1] - b[0] * a[1], a[1] * b[1]);
const fmul = (a, b) => F(a[0] * b[0], a[1] * b[1]);
const fdiv = (a, b) => F(a[0] * b[1], a[1] * b[0]);
const fval = a => a[0] / a[1];
const fstr = a => a[0] === 0 ? '0' : a[1] === 1 ? String(a[0]) : a[0] + '/' + a[1];

function inheritance(h) {
  // h: {deceased: 'male'|'female', husband, wives, sons, daughters, father, mother, brothers, sisters}
  const shares = [], notes = [];
  const add = (key, label, count, share, why) => shares.push({key, label, count, share, why});
  const children = h.sons + h.daughters > 0;
  const siblings = h.brothers + h.sisters;
  const siblingsInherit = !h.sons && !h.father;
  let spouse = null;
  if (h.deceased === 'female' && h.husband) spouse = ['husband', 'Husband', 1, children ? F(1, 4) : F(1, 2), children ? 'one quarter because the deceased left children' : 'one half because the deceased left no children'];
  if (h.deceased === 'male' && h.wives > 0) spouse = ['wives', h.wives > 1 ? 'Wives' : 'Wife', h.wives, children ? F(1, 8) : F(1, 4), (children ? 'one eighth' : 'one quarter') + (h.wives > 1 ? ', shared equally,' : '') + (children ? ' because the deceased left children' : ' because the deceased left no children')];
  if (spouse) add(...spouse);
  const umariyya = spouse && h.father && h.mother && !children && siblings < 2;
  if (h.mother && !umariyya) {
    const sixth = children || siblings >= 2;
    add('mother', 'Mother', 1, sixth ? F(1, 6) : F(1, 3), sixth ? 'one sixth because the deceased left children or two or more siblings' : 'one third');
  }
  if (h.father && h.sons) add('father', 'Father', 1, F(1, 6), 'one sixth because the deceased left a son');
  if (h.father && !h.sons && h.daughters) add('father', 'Father', 1, F(1, 6), 'one sixth, plus anything left over, because the deceased left daughters but no son');
  if (!h.sons && h.daughters) add('daughters', h.daughters > 1 ? 'Daughters' : 'Daughter', h.daughters, h.daughters > 1 ? F(2, 3) : F(1, 2), h.daughters > 1 ? 'two thirds, shared equally' : 'one half');
  if (siblingsInherit && !h.brothers && h.sisters && !h.daughters) add('sisters', h.sisters > 1 ? 'Full sisters' : 'Full sister', h.sisters, h.sisters > 1 ? F(2, 3) : F(1, 2), h.sisters > 1 ? 'two thirds, shared equally' : 'one half');

  let fixed = shares.reduce((s, x) => fadd(s, x.share), F(0));
  if (umariyya) {   // spouse + father + mother only: mother takes a third of what is left after the spouse
    const m = fmul(F(1, 3), fsub(F(1), spouse[3]));
    add('mother', 'Mother', 1, m, 'one third of what remains after the spouse\'s share (the Umariyyatan rule)');
    fixed = fadd(fixed, m);
    notes.push('Spouse, father and mother only: the mother takes a third of the remainder, not of the whole (Umariyyatan).');
  }
  // Who takes the residue, in order.
  let residuary = null;
  if (h.sons) residuary = {label: 'Sons and daughters', parts: [['sons', h.sons > 1 ? 'Sons' : 'Son', h.sons, 2], ['daughters', h.daughters > 1 ? 'Daughters' : 'Daughter', h.daughters, 1]], why: 'the residue, a son taking twice a daughter\'s share'};
  else if (h.father) residuary = {parts: [['father', 'Father', 1, 1]], why: 'the residue'};
  else if (siblingsInherit && h.brothers) residuary = {parts: [['brothers', h.brothers > 1 ? 'Full brothers' : 'Full brother', h.brothers, 2], ['sisters', h.sisters > 1 ? 'Full sisters' : 'Full sister', h.sisters, 1]], why: 'the residue, a brother taking twice a sister\'s share'};
  else if (siblingsInherit && h.sisters && h.daughters) residuary = {parts: [['sisters', h.sisters > 1 ? 'Full sisters' : 'Full sister', h.sisters, 1]], why: 'the residue, as sisters inherit together with daughters'};

  if (fval(fixed) > 1) {   // 'awl: shares exceed the estate, so all are reduced in proportion
    shares.forEach(x => { x.share = fdiv(x.share, fixed); x.why += ', reduced in proportion (\'awl)'; });
    notes.push('The fixed shares add up to more than the whole estate, so each is reduced in proportion (\'awl).');
  } else if (residuary) {
    const rest = fsub(F(1), fixed);
    const units = residuary.parts.reduce((s, p) => s + p[2] * p[3], 0);
    residuary.parts.forEach(p => {
      if (!p[2]) return;
      const part = fmul(rest, F(p[2] * p[3], units));
      const have = shares.find(x => x.key === p[0]);
      if (have) { have.share = fadd(have.share, part); }
      else add(p[0], p[1], p[2], part, residuary.why);
    });
  } else if (fval(fixed) < 1 && shares.length) {   // radd: the remainder returns to the sharers except the spouse
    const rest = fsub(F(1), fixed);
    const back = shares.filter(x => x.key !== 'husband' && x.key !== 'wives');
    const pool = back.length ? back : shares;
    const base = pool.reduce((s, x) => fadd(s, x.share), F(0));
    pool.forEach(x => { x.share = fadd(x.share, fmul(rest, fdiv(x.share, base))); x.why += ', plus a proportionate return of the remainder (radd)'; });
    notes.push(back.length ? 'Nothing is left for a residuary heir, so the remainder returns to the sharers other than the spouse (radd).'
                           : 'The spouse is the only heir, so the remainder also goes to the spouse.');
  }
  return {shares: shares.filter(x => x.count > 0), notes};
}

const CALCS = [
  {id: 'limitation', title: 'Limitation period', line: 'Limitation Act, 1908'},
  {id: 'inheritance', title: 'Inheritance shares', line: 'Hanafi (Sunni) law'},
];
let calcId = null;
function selectCalc(id) {
  calcId = id;
  renderPicker($('calc-list'), CALCS, id, selectCalc);
  if (id === 'limitation') {
    const opts = ['Suits', 'Appeals', 'Applications'].map(g => '<optgroup label="' + g + '">' +
      LIMITATION.filter(r => r[0] === g).map(r => '<option value="' + r[1] + '">Art. ' + r[1] + ' · ' + escapeHtml(r[2].slice(0, 90)) + (r[2].length > 90 ? '…' : '') + '</option>').join('') + '</optgroup>').join('');
    $('calc').innerHTML =
      '<div class="eyebrow">Limitation Act, 1908 · First Schedule</div><h2 class="display">Limitation period</h2>' +
      '<div class="form"><label class="eyebrow" for="lim-art">What are you filing?</label><select class="field" id="lim-art">' + opts + '</select>' +
      '<label class="eyebrow" for="lim-date" id="lim-from">Start date</label><input class="field" type="date" id="lim-date"></div>' +
      '<div id="lim-out" class="calcout"></div>';
    const update = () => {
      const art = Number($('lim-art').value);
      const row = LIMITATION.find(r => r[1] === art);
      $('lim-from').textContent = 'Period runs from: ' + row[4].toLowerCase();
      if (!$('lim-date').value) { $('lim-out').innerHTML = ''; return; }
      const {deadline} = limitationDeadline($('lim-date').value, art);
      const today = new Date(); today.setHours(0, 0, 0, 0);
      const left = Math.round((deadline - today) / 86400000);
      $('lim-out').innerHTML =
        '<div class="bigline"><span class="display">' + escapeHtml(deadline.toLocaleDateString(undefined, {day: 'numeric', month: 'long', year: 'numeric'})) + '</span>' +
        '<em>' + (left >= 0 ? left + ' days left' : 'expired ' + (-left) + ' days ago') + '</em></div>' +
        '<dl class="kvl"><dt>Article</dt><dd>' + art + '</dd><dt>Description</dt><dd>' + escapeHtml(row[2]) + '</dd>' +
        '<dt>Period</dt><dd>' + row[3][0] + ' ' + UNIT[row[3][1]][row[3][0] === 1 ? 0 : 1] + '</dd>' +
        '<dt>Runs from</dt><dd>' + escapeHtml(row[4]) + '</dd></dl>' +
        '<ul class="glaws"><li>The first day is not counted (section 12(1)).</li>' +
        '<li>For an appeal or review, the time taken to obtain a certified copy of the decree or order is excluded (section 12(2)); add it to this date.</li>' +
        '<li>If the court is closed on the last day, filing on the day it reopens is in time (section 4).</li>' +
        '<li>Provinces have amended some articles. Check the current schedule before relying on a date.</li></ul>';
    };
    $('lim-art').addEventListener('change', update);
    $('lim-date').addEventListener('input', update);
    update();
  } else {
    const n = (id, label, max) => '<label class="eyebrow" for="' + id + '">' + label + '</label><input class="field" type="text" inputmode="numeric" autocomplete="off" data-max="' + max + '" value="0" id="' + id + '">';
    $('calc').innerHTML =
      '<div class="eyebrow">Hanafi (Sunni) law · common heirs</div><h2 class="display">Inheritance shares</h2>' +
      '<div class="form two">' +
      '<label class="eyebrow" for="ih-dec">The deceased was</label><select class="field" id="ih-dec"><option value="male">a man</option><option value="female">a woman</option></select>' +
      '<label class="eyebrow" for="ih-estate">Estate value (optional, PKR)</label><input class="field" type="text" inputmode="numeric" autocomplete="off" data-max="999999999999" id="ih-estate">' +
      '<label class="eyebrow" for="ih-spouse" id="ih-spouse-l">Wives</label><input class="field" type="text" inputmode="numeric" autocomplete="off" data-max="4" value="0" id="ih-spouse">' +
      n('ih-sons', 'Sons', 30) + n('ih-daughters', 'Daughters', 30) +
      '<label class="eyebrow" for="ih-father">Father alive</label><select class="field" id="ih-father"><option value="0">No</option><option value="1">Yes</option></select>' +
      '<label class="eyebrow" for="ih-mother">Mother alive</label><select class="field" id="ih-mother"><option value="0">No</option><option value="1">Yes</option></select>' +
      n('ih-brothers', 'Full brothers', 30) + n('ih-sisters', 'Full sisters', 30) +
      '</div><div id="ih-out" class="calcout"></div>';
    const update = () => {
      const male = $('ih-dec').value === 'male';
      $('ih-spouse-l').textContent = male ? 'Wives (0 to 4)' : 'Husband (0 or 1)';
      $('ih-spouse').dataset.max = male ? 4 : 1;
      cleanCount($('ih-spouse'));
      const v = id => Math.max(0, parseInt($(id).value || '0', 10) || 0);
      const sp = Math.min(v('ih-spouse'), male ? 4 : 1);
      const h = {deceased: male ? 'male' : 'female', husband: male ? 0 : sp, wives: male ? sp : 0,
                 sons: v('ih-sons'), daughters: v('ih-daughters'), father: v('ih-father') > 0, mother: v('ih-mother') > 0,
                 brothers: v('ih-brothers'), sisters: v('ih-sisters')};
      const r = inheritance(h);
      const estate = parseFloat($('ih-estate').value) || 0;
      if (!r.shares.length) { $('ih-out').innerHTML = '<p class="privacy">Enter at least one heir.</p>'; return; }
      $('ih-out').innerHTML =
        '<div class="tablewrap"><table class="lib"><thead><tr><th>Heir</th><th>Share of the estate</th><th>Each</th><th>%</th>' + (estate ? '<th>PKR each</th>' : '') + '<th>Why</th></tr></thead><tbody>' +
        r.shares.map(x => {
          const each = fdiv(x.share, F(x.count));
          return '<tr><td><span class="who">' + escapeHtml(x.label) + (x.count > 1 ? ' × ' + x.count : '') + '</span></td><td class="num">' + fstr(x.share) +
                 '</td><td>' + fstr(each) + '</td><td>' + (fval(each) * 100).toFixed(2) + '</td>' +
                 (estate ? '<td>' + Math.round(estate * fval(each)).toLocaleString() + '</td>' : '') +
                 '<td class="small">' + escapeHtml(x.why) + '</td></tr>';
        }).join('') + '</tbody></table></div>' +
        '<ul class="glaws">' + r.notes.map(s => '<li>' + escapeHtml(s) + '</li>').join('') +
        '<li>Covers spouses, children, parents and full siblings under Hanafi law. It does not cover grandchildren (whom section 4 of the Muslim Family Laws Ordinance, 1961 lets inherit in place of a predeceased parent), grandparents, half-siblings, debts, bequests, or Shia law.</li>' +
        '<li>Check the result with a lawyer before dividing property.</li></ul>';
    };
    $('calc').querySelectorAll('input[data-max]').forEach(x => {
      x.addEventListener('input', () => cleanCount(x));
      x.addEventListener('blur', () => { if (x.value === '' && x.id !== 'ih-estate') x.value = '0'; update(); });
    });
    $('calc').querySelectorAll('input, select').forEach(x => x.addEventListener('input', update));
    update();
  }
}
// A whole number only: digits, no sign or decimal point, no leading zeros ("00001" becomes "1"), never above its limit.
function cleanCount(input) {
  const max = Number(input.dataset.max) || 0;
  let d = input.value.replace(/\D/g, '').replace(/^0+(?=\d)/, '');
  if (d !== '' && Number(d) > max) d = String(max);
  if (input.value !== d) input.value = d;
}

// ===========================================================================
// LIBRARY page
// ===========================================================================
function renderLibrary(q) {
  const tb = $('lib-body');
  tb.innerHTML = '';
  (judgments || []).filter(j => matches(j, q)).forEach(j => {
    const tr = el('tr');
    tr.innerHTML = '<td class="small"></td><td><span class="who"></span></td><td class="small hide-sm"></td>' +
                   '<td class="small hide-sm"></td><td class="num"></td>';
    const td = tr.children;
    td[0].textContent = caseNumber(j);
    if (hasParties(j)) td[1].firstChild.textContent = caseTitle(j);
    else { td[1].firstChild.className = 'small'; td[1].firstChild.textContent = 'Parties not extracted'; }
    td[2].textContent = (j.court || '').replace(/,?\s*(AT\s*)?$/i, '');
    td[3].textContent = (j.provisions || []).slice(0, 3).join(' · ');
    td[4].textContent = j.passages;
    tr.addEventListener('click', () => openJudgment(j.doc_id));
    tb.appendChild(tr);
  });
}
let libTab = 'judgments';
function statuteOrder(k) {
  const v = STATUTES[k], m = String(v.number).match(/^(\d+)(.*)$/);
  return [Object.keys(LAWS).indexOf(k.slice(0, k.indexOf(':') + 1)), m ? parseInt(m[1], 10) : 0, m ? m[2] : ''];
}
function renderStatutes(q) {
  const tb = $('stat-body');
  const words = (q || '').toLowerCase().split(/\s+/).filter(Boolean);
  const keys = Object.keys(STATUTES).filter(k => {
    const v = STATUTES[k], hay = (v.act + ' ' + statuteLabel(k, v) + ' ' + v.title + ' ' + (words.length > 1 ? v.text : '')).toLowerCase();
    return words.every(w => hay.includes(w));
  }).sort((a, b) => { const x = statuteOrder(a), y = statuteOrder(b); return x[0] - y[0] || x[1] - y[1] || x[2].localeCompare(y[2]); });
  tb.innerHTML = '';
  keys.slice(0, 400).forEach(k => {
    const v = STATUTES[k], tr = el('tr');
    tr.innerHTML = '<td class="small"></td><td class="num"></td><td><span class="who"></span></td>';
    tr.children[0].textContent = statuteLaw(k);
    tr.children[1].textContent = v.number;
    tr.children[2].firstChild.textContent = v.title;
    tr.addEventListener('click', () => openStatute(k));
    tb.appendChild(tr);
  });
  const count = (p) => Object.keys(STATUTES).filter(k => k.startsWith(p)).length;
  const acts = Object.keys(LAWS).filter(p => p !== 'const:' && count(p));
  $('lib-sum').textContent = count('const:') + ' articles of the Constitution and ' +
    acts.reduce((n, p) => n + count(p), 0).toLocaleString() + ' sections of ' + acts.length + ' Acts (' +
    acts.map(p => LAWS[p]).join(', ') + '), word for word, with their source.' +
    (keys.length > 400 ? ' Showing the first 400; filter to narrow.' : '');
}
// Insights: counts across the corpus, as single-hue horizontal bars. No AI.
let insightsData = null;
function barList(title, note, rows, opts) {
  opts = opts || {};
  const max = Math.max(1, ...rows.map(r => r.count));
  const panel = el('div', 'ipanel');
  panel.innerHTML = '<h3>' + escapeHtml(title) + '</h3>' + (note ? '<p class="inote">' + escapeHtml(note) + '</p>' : '');
  rows.forEach(r => {
    const row = el(opts.click ? 'button' : 'div', 'brow');
    if (opts.click) row.type = 'button';
    row.title = r.label + ': ' + plural(r.count, opts.unit || 'judgment');
    row.innerHTML = '<span class="bl"></span><span class="bt"><i style="width:' + Math.max(2, Math.round(100 * r.count / max)) + '%"></i></span><span class="bv">' + r.count + '</span>';
    row.querySelector('.bl').innerHTML = opts.link ? linkStatutes(escapeHtml(r.label)) : escapeHtml(r.label);
    if (opts.click) row.addEventListener('click', e => { if (!e.target.closest('.statref')) opts.click(r); });
    panel.appendChild(row);
  });
  return panel;
}
function showCiting(r) {
  loadJudgments().then(() => {
    const js = r.docs.map(d => (judgments || []).find(j => j.doc_id === d)).filter(Boolean);
    openPanel('<div class="eyebrow">Cited in ' + plural(js.length, 'judgment') + '</div><h2 class="display" style="font-size:40px;margin:14px 0 18px">' +
      escapeHtml(r.label) + '</h2><ul class="plist">' + js.map(j => '<li><b>' + escapeHtml(caseNumber(j).split(' ').pop()) +
      '</b><button type="button" class="who" data-doc="' + escapeHtml(j.doc_id) + '" style="text-align:start;color:inherit">' +
      escapeHtml(caseTitle(j)) + '</button></li>').join('') + '</ul>');
    $('panel').querySelectorAll('[data-doc]').forEach(b => b.addEventListener('click', () => openJudgment(b.dataset.doc)));
  });
}
async function renderInsights() {
  const box = $('lib-insights');
  if (!insightsData) {
    box.innerHTML = '<div class="status working">Counting</div>';
    try { insightsData = await (await fetch('/api/insights')).json(); }
    catch (e) { box.innerHTML = '<p class="error-text">Could not load insights.</p>'; return; }
  }
  const d = insightsData;
  box.innerHTML = '';
  box.append(
    barList('Most-cited provisions', 'Number of judgments citing each. Click a row to see them.', d.provisions, {link: true, click: showCiting}),
    barList('Most-cited precedents', 'Reported cases cited by the most judgments in the library.', d.citations, {click: showCiting}),
    barList('Case types', '', d.case_types),
    barList('Year the case was filed', 'From the case number.', d.years),
    barList('Judges', 'Judgments in the library per judge.', d.judges),
    barList('Passages by part of the judgment', 'How the chunker labelled the ' + (stats ? stats.passages : '') + ' passages.', d.sections, {unit: 'passage'}));
  $('lib-sum').textContent = 'Counted across every judgment in the library, with no AI involved. Hover a bar for its exact count.';
}

document.querySelectorAll('#lib-tabs button').forEach(b => b.addEventListener('click', () => {
  libTab = b.dataset.tab;
  document.querySelectorAll('#lib-tabs button').forEach(x => x.classList.toggle('on', x === b));
  $('lib-judgments').hidden = libTab !== 'judgments';
  $('lib-statutes').hidden = libTab !== 'statutes';
  $('lib-insights').hidden = libTab !== 'insights';
  $('lib-activity').hidden = libTab !== 'activity';
  $('lib-map').hidden = libTab !== 'map';
  $('lib-search').hidden = libTab === 'insights' || libTab === 'activity' || libTab === 'map';
  $('lib-search').placeholder = libTab === 'statutes' ? 'Filter by number, title or words in the text' : 'Filter by party, case number, court or provision';
  $('lib-search').value = '';
  if (libTab === 'statutes') renderStatutes('');
  else if (libTab === 'insights') renderInsights();
  else if (libTab === 'activity') renderActivity();
  else if (libTab === 'map') renderMap();
  else { renderLibrary(''); if (stats) paintStats(); }
}));
$('lib-search').addEventListener('input', e => libTab === 'statutes' ? renderStatutes(e.target.value) : renderLibrary(e.target.value));

// ===========================================================================
// start
// ===========================================================================

// ===========================================================================
// landing page motion: typed headline, rotating tagline, count-up numbers, scroll reveal
// ===========================================================================
const calm = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function countUp(node, to, opts) {
  opts = opts || {};
  const dec = opts.dec || 0, suffix = opts.suffix || '';
  const fmt = v => (dec ? v.toFixed(dec) : Math.round(v).toLocaleString()) + suffix;
  if (calm() || node._counted) { node.textContent = fmt(to); return; }
  node._counted = true;
  const t0 = performance.now(), dur = 1100;
  const tick = now => {
    const p = Math.min(1, (now - t0) / dur);
    node.textContent = fmt(to * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(tick); else node.textContent = fmt(to);
  };
  requestAnimationFrame(tick);
}
const TAGLINES = ['Know your rights.', 'Every answer traced to its paragraph.', 'Justice, explained in your own language.',
                  'Research built on real judgments.'];
function heroIntro() {
  const h = $('hero-title'), tag = $('hero-tag');
  if (!h || !tag || h._played) return;
  h._played = true;
  const onHome = () => document.body.classList.contains('p-home');
  if (calm()) { tag.textContent = TAGLINES[0]; return; }
  h.setAttribute('aria-label', 'Wakeel AI');
  h.innerHTML = '';
  const letters = [];
  'Wakeel\nAI'.split('').forEach(c => {
    if (c === '\n') { h.appendChild(document.createElement('br')); return; }
    const sp = el('span', 'hl', c);
    sp.setAttribute('aria-hidden', 'true');
    h.appendChild(sp); letters.push(sp);
  });
  const caret = el('i', 'hero-caret');
  let pi = 0;
  const typeTag = () => {
    if (!onHome()) { setTimeout(typeTag, 1000); return; }
    const phrase = TAGLINES[pi]; let i = 0;
    const t = setInterval(() => {
      tag.textContent = phrase.slice(0, ++i);
      if (i >= phrase.length) { clearInterval(t); setTimeout(eraseTag, 2200); }
    }, 22);
  };
  const eraseTag = () => {
    const phrase = TAGLINES[pi]; let i = phrase.length;
    const t = setInterval(() => {
      tag.textContent = phrase.slice(0, --i);
      if (i <= 0) { clearInterval(t); pi = (pi + 1) % TAGLINES.length; setTimeout(typeTag, 300); }
    }, 9);
  };
  let k = 0;
  const typeTitle = () => {
    if (k >= letters.length) { setTimeout(() => { caret.remove(); typeTag(); }, 450); return; }
    letters[k].classList.add('on');
    letters[k].after(caret);
    k++;
    setTimeout(typeTitle, 120);
  };
  setTimeout(typeTitle, 450);
}
function setupReveal() {
  if (!('IntersectionObserver' in window)) return;
  const targets = [...document.querySelectorAll('.trust-in .eyebrow, .trust h2, .trust-grid > div, .trust-note')];
  targets.forEach(t => {
    t.classList.add('reveal');
    const g = t.parentElement;
    if (g.classList.contains('trust-grid')) t.style.setProperty('--d', [...g.children].indexOf(t) * 0.12 + 's');
  });
  const io = new IntersectionObserver(entries => entries.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add('in');
    io.unobserve(e.target);
    const n = e.target.querySelector('[data-count]');
    if (n) countUp(n, parseFloat(n.dataset.count), {dec: +n.dataset.dec || 0, suffix: n.dataset.suffix || ''});
  }), {threshold: 0.15, rootMargin: '0px 0px -6% 0px'});
  targets.forEach(t => io.observe(t));
}
setupReveal();

showPage(pageFromPath(location.pathname));
// Deep link for a scripted demo: /ask?q=...&mode=lawyer&language=ur asks straight away.
const params = new URLSearchParams(location.search);
['mode', 'language'].forEach(id => {
  const v = params.get(id);
  if (v && document.querySelector('#' + id + ' button[data-value="' + v + '"]')) setSeg(id, v);
});
if (params.get('q')) ask(params.get('q'));

// ===========================================================================
// Guided tour: a card that walks through the app, runs a real example and highlights what it is describing.
// For anyone opening the link without someone presenting. Start it from the home page, or with ?tour=1.
// ===========================================================================
const TOUR_FIR = 'FIRST INFORMATION REPORT\nFIR No. 214/2024 dated 03.03.2024, Police Station Saddar, Bahawalpur.\nOffence: Section 489-F PPC.\n' +
  'The complainant Muhammad Aslam states that the accused Tariq Mehmood borrowed Rs. 900,000 from him and issued Cheque No. 00771245 ' +
  'drawn on Bank Alfalah for repayment, which was dishonoured on presentation due to insufficient funds. The complainant requests that ' +
  'legal action be taken against the accused under the law.';
const TOUR = [
  {path: '/', target: '.home-l h1', title: 'Real judgments, checked answers',
   text: 'Wakeel AI answers Pakistani legal questions from real Lahore High Court judgments, and shows where every claim came from. This tour runs real examples; each step takes a few seconds.'},
  {path: '/case', target: '#case-team', wait: 60000, title: 'An orchestrator and its agents',
   text: 'Any request goes to an orchestrator, which picks a team for the job: here, a situation, so research, statutes, precedents, strategy, drafting and a verifier. Watch each node work and the data move along the lines between them.',
   enter: () => { if (!caseRun || caseRun.finished) { $('case-text').value = CASE_TRIES[0][1]; caseFile = null; $('case-go').click(); } }},
  {path: '/ask', target: '#thread', title: 'Ask in plain words',
   text: 'A real question for a lawyer, in English, answered from the judgments. It also works in Urdu and Roman Urdu, by typing or speaking. Click a small citation number in the answer to open the exact paragraph.',
   enter: () => { setSeg('mode', 'lawyer'); setSeg('language', 'en'); if (!busy) ask('Can the High Court cancel pre-arrest bail granted on the basis of a compromise?'); }},
  {path: '/ask', target: '#thread .body .ground', wait: 120000, title: 'Checked, not just cited',
   text: 'Under the answer: how many passages it cites, and a fact check that looks for every name, section, number and quotation in the paragraph it cites. No AI does the checking. Anything not found is marked in amber.'},
  {path: '/ask', target: '#thread .body .statref', wait: 120000, title: 'The law, word for word',
   text: 'A section like 497(5) Cr.P.C. is a link. It opens the section exactly as published in Pakistan Code, with its source. The Constitution, PPC, CrPC, CPC and the family laws are all in.'},
  {path: '/compare', target: '#cmp-tabs', title: 'Compare many judgments at once',
   text: 'Pick judgments, and your own documents too, choose columns such as outcome or provisions, and get a table where every cell cites its paragraph. Then ask questions of the table.',
   enter: () => { const b = document.querySelector('#cmp-tabs [data-tab="table"]'); if (b) b.click(); }},
  {path: '/explain', target: '#doc-text', title: 'Explain my document',
   text: 'Upload or paste an FIR, notice or court order. It explains it in plain words, ranks red flags and deadlines, and finds similar judgments. A sample FIR is filled in; press Explain to try it. Nothing is stored.',
   enter: () => { $('doc-text').value = TOUR_FIR; }},
  {path: '/draft', target: '#draft-kinds', title: 'Drafts and workflows',
   text: 'Notices, applications and, for lawyers, hearing preparation with the questions a bench may ask, a dated chronology, and a client update letter. Facts you leave out stay as [placeholders]; nothing is invented.',
   enter: () => selectDraft('hearing')},
  {path: '/library', target: '#lib-map', title: 'How the judgments connect',
   text: 'Squares are reported cases that two or more judgments rely on; circles are judgments. Hover to trace a link, click to open. Drawn from the library\'s own citations, with no AI.',
   enter: () => { const b = document.querySelector('#lib-tabs [data-tab="map"]'); if (b) b.click(); }},
  {path: '/matters', target: '#m-new', title: 'Keep a case together',
   text: 'Save any answer, table or draft to a matter, and export the whole matter as one PDF. It lives in your browser only; nothing is sent to or kept on a server.'},
  {path: '/calculate', target: '#calc-list', title: 'Rules, not guesses',
   text: 'Limitation periods and inheritance shares are calculated by fixed rules, with the rule shown underneath. The same inputs always give the same answer, with no AI.'},
  {path: '/', target: '#tour-start', title: 'Now try your own',
   text: 'Ask anything about the {n} judgments in the library, or open Guides if you have a problem and are not sure where to start. This is a research aid, not legal advice.'},
];
let tourI = -1, tourToken = 0, tourPrev = null;
function tourClear() { document.querySelectorAll('.tour-hl').forEach(e => e.classList.remove('tour-hl')); }
function tourEnd() {
  tourI = -1; tourToken++; tourClear(); $('tour').hidden = true;
  if (tourPrev) { setSeg('mode', tourPrev.mode); setSeg('language', tourPrev.language); tourPrev = null; }   // put the toggles back
}
async function tourHighlight(sel, token, waitMs) {
  const t0 = performance.now();
  while (token === tourToken && performance.now() - t0 < waitMs) {
    const e = document.querySelector(sel);
    if (e && e.getBoundingClientRect().width) {
      tourClear();
      e.classList.add('tour-hl');
      e.scrollIntoView({block: 'center', behavior: 'smooth'});
      return;
    }
    await new Promise(r => setTimeout(r, 300));
  }
}
function tourShow(i) {
  if (i < 0 || i >= TOUR.length) return tourEnd();
  if (tourI < 0 && !tourPrev) tourPrev = {mode: segValue('mode'), language: segValue('language')};
  tourI = i; const token = ++tourToken, s = TOUR[i];
  tourClear();
  $('tour').hidden = false;
  $('tour-n').textContent = 'Tour · step ' + (i + 1) + ' of ' + TOUR.length;
  $('tour-t').textContent = s.title;
  $('tour-p').textContent = s.text.replace('{n}', stats ? stats.judgments : 'indexed');
  $('tour-back').disabled = i === 0;
  $('tour-next').textContent = i === TOUR.length - 1 ? 'Finish' : 'Next';
  if (s.path) go(s.path);
  if (s.enter) s.enter();
  if (s.target) tourHighlight(s.target, token, s.wait || 6000);
}
$('tour-start').addEventListener('click', () => tourShow(0));
$('pitch-tour').addEventListener('click', () => tourShow(0));
$('tour-next').addEventListener('click', () => tourShow(tourI + 1));
$('tour-back').addEventListener('click', () => tourShow(tourI - 1));
$('tour-x').addEventListener('click', tourEnd);
document.addEventListener('keydown', e => {
  if (tourI < 0 || /^(INPUT|TEXTAREA|SELECT)$/.test((e.target.tagName || ''))) return;
  if (e.key === 'Escape') tourEnd();
  else if (e.key === 'ArrowRight') tourShow(tourI + 1);
  else if (e.key === 'ArrowLeft') tourShow(tourI - 1);
});
if (params.get('tour')) tourShow(0);

</script>
</body>
</html>
"""


def _read_metadata():
    """(doc_id, chunks) for every indexed judgment, read from the index's metadata files."""
    for meta_file in sorted(W.METADATA_DIR.glob("*.json")):
        try:
            yield meta_file.stem, json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def corpus_stats() -> dict:
    """Counts for the home page and library. Citations and provisions are counted once
    each, ignoring case and line breaks ("Section 497" and "section\\n497" are one)."""
    judgments = passages = 0
    citations, provisions = set(), set()
    for _, chunks in _read_metadata():
        judgments += 1
        passages += len(chunks)
        for chunk in chunks:
            citations.update(_norm(c).lower() for c in chunk.get("citations", []))
            provisions.update(_norm(p).lower() for p in chunk.get("provisions", []))
    return {
        "judgments": judgments,
        "passages": passages,
        "citations": len(citations),
        "provisions": len(provisions),
        "provider": W.PROVIDERS[W.LLM_PROVIDER]["label"],
        "model": W.LLM_MODEL,
    }


def _judgment_summary(doc_id: str, chunks: list) -> dict:
    head = chunks[0] if chunks else {}
    counts = {}
    for chunk in chunks:
        for p in chunk.get("provisions", []):
            key = _norm(p)
            counts[key.lower()] = (counts.get(key.lower(), (0, key))[0] + 1, key)
    top = [label for _, label in sorted(counts.values(), key=lambda x: -x[0])[:5]]
    return {
        "doc_id": doc_id,
        "title": head.get("title", ""),
        "case_no": head.get("case_no", ""),
        "judgment_type": head.get("judgment_type", ""),
        "court": _norm(head.get("court", "")),
        "judges": head.get("judges", []),
        "parties": head.get("parties", {}) if isinstance(head.get("parties"), dict) else {},
        "passages": len(chunks),
        "provisions": top,
    }


_statutes = None


def load_statutes() -> dict:
    """statutes.json (built by build_statutes.py): provision key -> text, title and source."""
    global _statutes
    if _statutes is None:
        try:
            _statutes = json.loads(Path("statutes.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _statutes = {}
    return _statutes


_CASE_TYPES = [  # (match on the type with dots and spaces removed, label)
    (r"^(writpetition|wp)$", "Writ petition"), (r"^(crlmisc|criminalmisc)", "Criminal miscellaneous"),
    (r"^murderreference", "Murder reference"), (r"^(civilrevision|cr)$", "Civil revision"),
    (r"^criminalrevision", "Criminal revision"), (r"^criminalappeal", "Criminal appeal"),
    (r"^rsa$", "Regular second appeal"), (r"^ica$", "Intra-court appeal"), (r"^efa$", "Execution first appeal"),
    (r"^fao$", "First appeal from order"), (r"^ptr$", "Tax reference"), (r"^customsreference", "Customs reference"),
    (r"^electionpetition", "Election petition"), (r"^transferapplication", "Transfer application"),
]


def _case_type(raw: str) -> str:
    key = re.sub(r"[^a-z]", "", (raw or "").lower())
    for pattern, label in _CASE_TYPES:
        if re.search(pattern, key):
            return label
    return (raw or "Other").strip().title() or "Other"


def corpus_insights() -> dict:
    """Counts across the corpus for the Insights tab. Every count is of judgments, not passages."""
    types, years, judges, sections = {}, {}, {}, {}
    provisions, citations = {}, {}
    for doc_id, chunks in _read_metadata():
        head = chunks[0] if chunks else {}
        t = _case_type(head.get("judgment_type", ""))
        types[t] = types.get(t, 0) + 1
        y = re.search(r"((?:19|20)\d{2})\s*$", head.get("case_no", "") or "")
        if y:
            years[y.group(1)] = years.get(y.group(1), 0) + 1
        for j in {_norm(x).title() for x in head.get("judges", []) if _norm(x)}:
            judges[j] = judges.get(j, 0) + 1
        seen_p, seen_c = {}, {}
        for c in chunks:
            s = (c.get("section_type") or "other").capitalize()
            sections[s] = sections.get(s, 0) + 1
            for p in c.get("provisions", []):
                p = _norm(p)
                if re.search(r"(ppc|p\.p\.c|cr\.?p\.?c|c\.p\.c|cpc|constitution|act|ordinance|order)", p, re.I):
                    if p.isupper():   # "ARTICLE 186 OF THE CONSTITUTION"
                        p = re.sub(r"\b(Of|The|And)\b", lambda m: m.group(1).lower(), p.title())
                    seen_p.setdefault(p.lower(), p)
            for cit in c.get("citations", []):
                cit = _norm(cit)
                # A reported citation: a year and a law report. The extractor also yields the
                # tail of a PLD citation on its own ("1989 SC 166" from "PLD 1989 SC 166"); skip it.
                if re.match(r"^(19|20)\d{2} (SC|Lah|Kar|Pesh|Quetta|FSC)\b", cit):
                    continue
                if re.search(r"\b(19|20)\d{2}\b", cit) and re.search(r"\b(SCMR|PLD|PLJ|YLR|MLD|CLC|PCrLJ|PTD|CLD)\b", cit):
                    seen_c.setdefault(cit.lower(), cit)
        for k, label in seen_p.items():
            entry = provisions.setdefault(k, {"label": label, "docs": []})
            entry["docs"].append(doc_id)
        for k, label in seen_c.items():
            entry = citations.setdefault(k, {"label": label, "docs": []})
            entry["docs"].append(doc_id)

    def top(d, n):
        return [{"label": k, "count": v} for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]

    def top_docs(d, n):
        ranked = sorted(d.values(), key=lambda e: (-len(e["docs"]), e["label"]))[:n]
        return [{"label": e["label"], "count": len(e["docs"]), "docs": e["docs"]} for e in ranked]

    return {
        "case_types": top(types, 12),
        "years": [{"label": k, "count": years[k]} for k in sorted(years)],
        "judges": top(judges, 10),
        "sections": top(sections, 8),
        "provisions": top_docs(provisions, 15),
        "citations": top_docs(citations, 12),
    }


def list_judgments() -> dict:
    items = [_judgment_summary(doc_id, chunks) for doc_id, chunks in _read_metadata()]
    items.sort(key=lambda j: (j["parties"].get("appellant") or j["title"] or j["doc_id"]).lower())
    return {"judgments": items}


def citation_rows() -> dict:
    """Every case citation in the library, with the judgment and paragraph that cites it.

    The browser groups these by reported case ("2016 SCMR 859"), so a citation can show which
    judgments in the library rely on it, and a judgment which others share its precedents.
    """
    rows = []
    for doc_id, chunks in _read_metadata():
        for c in chunks:
            para = str(c.get("paragraph_number", "") or "")
            for cite in W._as_list(c.get("citations"), []):
                rows.append({"doc_id": doc_id, "paragraph": para, "citation": re.sub(r"\s+", " ", str(cite)).strip()})
    return {"citations": rows}


def get_judgment(doc_id: str):
    chunks = W.load_judgment(doc_id)
    if not chunks:
        return None
    return {**_judgment_summary(doc_id, chunks),
            "passages": [{"paragraph": str(c.get("paragraph_number", "") or ""),
                          "section_type": c.get("section_type", ""),
                          "text": W.clean_excerpt(c.get("text", ""))} for c in chunks]}


def pdf_text_and_pages(blob: bytes, max_pages: int = 15, image_pages: int = 3):
    """Text of a PDF, plus page images when it has no text layer (a scan)."""
    import pdfplumber
    with pdfplumber.open(io.BytesIO(blob)) as pdf:
        pages = pdf.pages[:max_pages]
        text = "\n".join((p.extract_text() or "") for p in pages)
        images = []
        if len(text.strip()) < 200:
            for page in pages[:image_pages]:
                buf = io.BytesIO()
                page.to_image(resolution=150).original.save(buf, format="PNG")
                images.append(("image/png", buf.getvalue()))
    return text, images


def read_upload(data: dict):
    """(text, images) from an /api/explain request. Everything stays in memory."""
    text = (data.get("text") or "").strip()
    images = []
    if data.get("data"):
        try:
            blob = base64.b64decode(data["data"], validate=False)
        except (binascii.Error, ValueError):
            raise ValueError("The file could not be read.")
        if len(blob) > UPLOAD_LIMIT:
            raise ValueError("The file is larger than 10 MB.")
        mime = (data.get("mime") or "").lower()
        if mime == "application/pdf" or (data.get("filename") or "").lower().endswith(".pdf"):
            try:
                pdf_text, images = pdf_text_and_pages(blob)
            except Exception:
                raise ValueError("The PDF could not be opened.")
            text = (text + "\n" + pdf_text).strip()
        elif mime in IMAGE_TYPES:
            images = [(mime, blob)]
        else:
            raise ValueError("Upload a PDF or a photo (PNG, JPEG or WebP).")
    if not text and not images:
        raise ValueError("Upload a file or paste the document's text.")
    return text, images


class AnswerCache:
    """Saved answers for a scripted demo, so a slow or unavailable model cannot stall it.

    Turned on with `python ui.py --cache`. Answers are keyed on the request and the model
    in use, written as JSON files, and replayed as they were (including the live steps).
    Failed answers are never saved.
    """

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(kind: str, payload: dict) -> str:
        blob = json.dumps([kind, payload, W.LLM_PROVIDER, W.LLM_MODEL], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str):
        try:
            return json.loads((self.folder / f"{key}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key: str, value) -> None:
        tmp = self.folder / f"{key}.tmp"
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.folder / f"{key}.json")


def _usable(result: dict) -> bool:
    return isinstance(result, dict) and result.get("route") not in ("ERROR", None)


class ActivityLog:
    """Which features were used and how they went, for the Activity tab (a small command center).

    One JSON line per request: the endpoint, its route, status and seconds; and the helpful /
    not helpful votes. Questions, answers and documents are never written here.
    """

    FEATURES = {"/api/chat": "Ask", "/api/chat/stream": "Ask", "/api/summarize": "Summarize",
                "/api/compare": "Compare", "/api/table": "Review table", "/api/table/ask": "Review table",
                "/api/explain": "Explain", "/api/explain/ask": "Explain", "/api/draft": "Draft",
                "/api/strategy": "Workflows", "/api/workflow": "Workflows", "/api/judgment/ask": "Ask a judgment", "/api/case/stream": "Agent room",
                "/api/citator": "Citator", "/api/moot": "Moot court", "/api/video": "Video evidence"}

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def add(self, **row):
        row = {"t": round(time.time(), 1), **row}
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def summary(self) -> dict:
        rows = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            pass
        calls = [r for r in rows if r.get("kind") == "request"]
        votes = [r for r in rows if r.get("kind") == "feedback"]
        by_feature, times = {}, {}
        for r in calls:
            f = self.FEATURES.get(r.get("path"), "Other")
            by_feature[f] = by_feature.get(f, 0) + 1
            if r.get("ok"):
                times.setdefault(f, []).append(r.get("seconds", 0))
        median = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
        ok = sum(1 for r in calls if r.get("ok"))
        return {
            "requests": len(calls), "ok": ok, "errors": len(calls) - ok,
            "cached": sum(1 for r in calls if r.get("cached")),
            "since": min((r["t"] for r in rows), default=None),
            "features": sorted(({"label": k, "count": v} for k, v in by_feature.items()), key=lambda x: -x["count"]),
            "median_seconds": sorted(({"label": k, "count": round(median(v), 1)} for k, v in times.items()),
                                     key=lambda x: -x["count"]),
            "helpful": sum(1 for v in votes if v.get("helpful")),
            "not_helpful": sum(1 for v in votes if not v.get("helpful")),
            "facts_checked": sum(int(v.get("facts_checked") or 0) for v in votes),
            "facts_missing": sum(int(v.get("facts_missing") or 0) for v in votes),
        }


# Seconds a replayed answer takes when the cache simulates the model, close to what the
# real calls took (explain, draft and the rest measured at 7 to 29 s on gpt-4o-mini).
SIMULATED_STEP_SECONDS = {"classify": (0.8, 1.6), "retrieve": (1.6, 3.0), "retrieve_pair": (2.0, 3.6),
                          "answer": (3.5, 6.5), "summarize": (4.5, 8.0), "compare": (5.0, 9.0), "clarify": (1.0, 2.0)}
SIMULATED_TASK_SECONDS = {"summarize": (6.0, 10.0), "compare": (8.0, 13.0), "table": (6.0, 10.0),
                          "explain": (7.0, 11.0), "draft": (6.0, 10.0), "workflow": (8.0, 13.0),
                          "strategy": (8.0, 13.0)}


def make_handler(agent, index_fn, stats_fn=corpus_stats, cache=None, replay_delay=0.35, activity=None, speak_fn=None,
                 simulate=False):
    """Build the request handler. agent/index_fn/stats_fn are injected so tests can stub them.

    cache: an AnswerCache, or None. Only the first turn of a conversation is cached; a
    follow-up depends on the turns before it.
    simulate: when a saved answer is replayed, wait about as long as the model took, step by
    step, so the replay looks like a live run. Off by default (and in tests).
    """

    lock = threading.Lock()
    seen_threads = set()
    speak = speak_fn or W.synthesize_speech
    spoken = {}   # voice clips already made: in the demo cache's folder when there is one, else in memory

    def speech(text: str, language: str, tone: str = "plain") -> bytes:
        key = hashlib.sha256(f"{W.SPEECH_MODEL}|{W.SPEECH_VOICE}|{language}|{tone}|{text}".encode("utf-8")).hexdigest()[:32]
        path = cache.folder / "speech" / f"{key}.mp3" if cache else None
        if key in spoken:
            return spoken[key]
        if path and path.exists():
            return path.read_bytes()
        audio = speak(text, language) if tone == "plain" else speak(text, language, tone)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
        elif len(spoken) < 300:
            spoken[key] = audio
        return audio

    def first_turn(thread_id: str) -> bool:
        with lock:
            fresh = thread_id not in seen_threads
            seen_threads.add(thread_id)
        return fresh

    def cached_task(kind, payload, run):
        """Run a non-chat task through the cache. What the agents did is saved with the answer ("watch"), so a page
        watching a replay sees the same agents and words, at about the pace they came."""
        key = cache.key(kind, payload) if cache else None
        hit = cache.get(key) if key else None
        if hit is not None:
            steps = hit.get("watch") or []
            if W.watching() is not None and steps:
                for event in steps:
                    time.sleep((0.05 if simulate else 0.012) if event.get("type") == "chunk" else (0.5 if simulate else 0.12))
                    W.watch_emit(event)
            elif simulate:
                time.sleep(random.uniform(*SIMULATED_TASK_SECONDS.get(kind, (6.0, 10.0))))
            return {k: v for k, v in hit.items() if k != "watch"}
        watch, token = (W.watching(), None) if W.watching() is not None else W.watch_begin()
        mark = len(watch.log)
        try:
            result = run()
        finally:
            if token is not None:
                W.watch_end(token)
        if key and _usable(result):
            cache.put(key, {**result, "watch": watch.log[mark:]})
        return result

    def watched(events):
        """Run events() on a worker thread with the watch on, yielding its events and the watch's in the order they
        happen."""
        inbox, end = queue.Queue(), object()

        def work():
            _, token = W.watch_begin(inbox.put)
            try:
                for event in events():
                    inbox.put(event)
            except Exception as e:
                inbox.put({"type": "result", "answer": f"Error: {e}", "route": "ERROR", "restated_query": "",
                           "sources": [], "grounded": False})
            finally:
                W.watch_end(token)
                inbox.put(end)

        threading.Thread(target=work, daemon=True).start()
        while True:
            event = inbox.get()
            if event is end:
                return
            yield event

    def turn_events(message, thread_id, mode, language):
        """Progress events for one turn, ending with a result. Works with agents that cannot stream."""
        if hasattr(agent, "run_stream"):
            yield from agent.run_stream(message, thread_id=thread_id, mode=mode, language=language)
        else:
            yield {"type": "result",
                   **agent.run_detailed(message, thread_id=thread_id, mode=mode, language=language)}

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
            self._log(code, payload)

        def _log(self, code, payload):
            """One line in the activity log per feature request: never the content."""
            if not activity or self.command != "POST" or self.path not in ActivityLog.FEATURES:
                return
            if code in (400, 413):
                return   # the request was turned away before any work: a user slip, not a failure
            route = str((payload or {}).get("route", ""))
            activity.add(kind="request", path=self.path, route=route, ok=code == 200 and route != "ERROR",
                         seconds=round(time.time() - getattr(self, "_t0", time.time()), 2),
                         cached=bool(getattr(self, "_cached", False)))

        def _watched(self, data):
            """A request from a page that asked to watch the work: the agents' events as they happen, then the
            answer as {"type": "result", "status": <HTTP status it would have had>, ...}."""
            out = {}
            self._json = lambda code, payload: out.update(code=code, payload=payload)   # captured, sent as the last event

            def events():
                self._handle(data)
                yield {"type": "result", "status": out.get("code", 500),
                       **(out.get("payload") or {"answer": "No answer was produced.", "route": "ERROR", "sources": []})}
            try:
                self._stream(watched(events))
            finally:
                del self._json
            self._log(out.get("code", 500), out.get("payload"))

        def _stream(self, events):
            """Send events as JSON lines over a chunked response, flushing each one."""
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def write(event):
                line = (json.dumps(event) + "\n").encode("utf-8")
                self.wfile.write(b"%X\r\n%s\r\n" % (len(line), line))
                self.wfile.flush()

            try:
                try:
                    for event in events:
                        write(event)
                except (BrokenPipeError, ConnectionError):
                    raise
                except Exception as e:
                    # Headers are already sent, so report the failure as the final event.
                    write({"type": "result", "answer": f"Error: {e}", "route": "ERROR",
                           "restated_query": "", "sources": [], "grounded": False})
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionError):
                self.close_connection = True  # the browser went away mid-answer

        def do_GET(self):
            path = urlsplit(self.path).path   # ignore ?q=... deep-link parameters
            if path in APP_PATHS:
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/healthz":
                self._json(200, {"ok": True})
            elif path == "/api/stats":
                try:
                    self._json(200, stats_fn())
                except Exception as e:
                    self._json(500, {"error": f"Could not read corpus stats: {e}"})
            elif path == "/api/judgments":
                self._json(200, list_judgments())
            elif path == "/api/statutes":
                self._json(200, {"statutes": load_statutes()})
            elif path == "/api/insights":
                self._json(200, corpus_insights())
            elif path == "/api/citations":
                self._json(200, citation_rows())
            elif path == "/api/activity":
                self._json(200, activity.summary() if activity else {"requests": 0, "off": True})
            elif path.startswith("/api/judgments/"):
                found = get_judgment(unquote(path[len("/api/judgments/"):]))
                self._json(200 if found else 404, found or {"error": "No such judgment in the index."})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            self._t0, self._cached = time.time(), False
            length = int(self.headers.get("Content-Length") or 0)
            if length > BODY_LIMIT:
                self.close_connection = True   # the unread body cannot be reused
                self._json(413, {"answer": "The file is larger than 10 MB.", "route": "ERROR", "sources": []})
                return
            raw = self.rfile.read(length) if length else b"{}"

            if self.path == "/api/index":
                with lock:
                    try:
                        self._json(200, {"message": index_fn()})
                    except Exception as e:
                        self._json(500, {"message": f"Indexing failed: {e}"})
                return

            if self.path not in POST_PATHS:
                self._json(404, {"error": "not found"})
                return

            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except ValueError:
                self._json(400, {"answer": "Malformed request.", "route": "ERROR"})
                return

            if data.get("watch") is True and self.path in WATCHED_PATHS:
                self._watched(data)
                return
            self._handle(data)

        def _handle(self, data):
            """One feature request, answered with self._json or a stream."""
            mode = data.get("mode") or W.DEFAULT_MODE
            language = data.get("language") or W.DEFAULT_LANGUAGE

            def guarded(fn):
                with lock:
                    try:
                        return fn()
                    except Exception as e:
                        return {"answer": f"Error: {e}", "route": "ERROR", "sources": []}

            if self.path == "/api/explain":
                try:
                    text, images = read_upload(data)
                except ValueError as e:
                    self._json(400, {"answer": str(e), "route": "ERROR", "sources": []})
                    return
                # Whitespace is ignored, so a pasted document matches however its lines end.
                digest = hashlib.sha256((re.sub(r"\s+", " ", text).strip() +
                                         "".join(hashlib.sha256(b).hexdigest() for _, b in images))
                                        .encode("utf-8")).hexdigest()
                self._json(200, cached_task(
                    "explain", {"doc": digest, "mode": mode, "language": language},
                    lambda: guarded(lambda: agent.explain_document(text, images, mode=mode, language=language))))
                return

            if self.path == "/api/explain/ask":
                text, question = str(data.get("text") or ""), str(data.get("question") or "").strip()
                if not text.strip() or not question:
                    self._json(400, {"answer": "Upload a document and ask a question about it.",
                                     "route": "ERROR", "sources": []})
                    return
                self._json(200, guarded(lambda: agent.ask_document(text, question, mode=mode, language=language)))
                return

            if self.path == "/api/table":
                doc_ids = data.get("doc_ids") if isinstance(data.get("doc_ids"), list) else []
                columns = data.get("columns") if isinstance(data.get("columns"), list) else []
                files = [f for f in (data.get("files") or []) if isinstance(f, dict)][:8]
                if not (doc_ids or files) or not columns:
                    self._json(400, {"answer": "Choose at least one judgment or document, and one column.",
                                     "route": "ERROR", "sources": []})
                    return
                # The user's own documents: read in memory, never stored or indexed.
                uploads = []
                try:
                    for f in files:
                        text, images = read_upload(f)
                        uploads.append((str(f.get("filename") or "Document")[:120], text, images))
                except ValueError as e:
                    self._json(400, {"answer": f"{f.get('filename') or 'A file'}: {e}", "route": "ERROR", "sources": []})
                    return
                digests = [hashlib.sha256((re.sub(r"\s+", " ", t).strip() + "".join(
                    hashlib.sha256(b).hexdigest() for _, b in imgs)).encode("utf-8")).hexdigest()
                    for _, t, imgs in uploads]

                def build():
                    documents = []
                    for name, text, images in uploads:
                        if images and len(text) < 200:   # a photo or a scan: read its pages first
                            text = (text + "\n\n" + agent.transcribe_images(images)).strip()
                        documents.append({"name": name, "text": text})
                    return agent.review_table(doc_ids, columns, mode=mode, language=language, documents=documents)

                self._json(200, cached_task(
                    "table", {"doc_ids": doc_ids, "docs": digests, "columns": columns, "mode": mode, "language": language},
                    lambda: guarded(build)))
                return

            if self.path == "/api/table/ask":
                question, table = str(data.get("question") or ""), str(data.get("table") or "")
                if not question.strip() or not table.strip():
                    self._json(400, {"answer": "Build a table and ask a question about it.", "route": "ERROR", "sources": []})
                    return
                self._json(200, guarded(lambda: agent.ask_table(question, table, mode=mode, language=language)))
                return

            if self.path == "/api/strategy":
                facts = data.get("facts") if isinstance(data.get("facts"), dict) else {}
                if not any(str(v).strip() for v in facts.values()):
                    self._json(400, {"answer": "Describe your situation first.", "route": "ERROR", "sources": []})
                    return
                self._json(200, cached_task(
                    "strategy", {"facts": facts, "mode": mode, "language": language},
                    lambda: guarded(lambda: agent.strategy_memo(facts, mode=mode, language=language))))
                return

            if self.path == "/api/case/stream":
                situation = str(data.get("situation") or "").strip()
                upload = {k: data.get(k) for k in ("text", "data", "mime", "filename") if data.get(k)}
                doc_ids = [str(d) for d in (data.get("doc_ids") or []) if isinstance(d, str)][:2]
                if not situation and not upload and not doc_ids:
                    self._json(400, {"answer": "Describe your situation first.", "route": "ERROR", "sources": []})
                    return
                text, images = "", []
                if upload:
                    try:
                        text, images = read_upload(upload)
                    except ValueError as e:
                        self._json(400, {"answer": str(e), "route": "ERROR", "sources": []})
                        return
                digest = hashlib.sha256((re.sub(r"\s+", " ", text).strip() + "".join(
                    hashlib.sha256(b).hexdigest() for _, b in images)).encode("utf-8")).hexdigest()
                key = cache.key("case", {"situation": situation, "doc": digest, "doc_ids": doc_ids, "mode": mode,
                                         "language": language}) if cache else None
                hit = cache.get(key) if key else None
                if hit is not None:
                    def replay():
                        for event in hit:
                            if event.get("type") == "agent":
                                time.sleep(random.uniform(0.8, 2.2) if simulate and event.get("status") == "done" else replay_delay)
                            elif event.get("type") == "chunk":
                                time.sleep(0.06 if simulate else 0.02)
                            yield event
                    self._stream(replay())
                    self._cached = True
                    self._log(200, {"route": "CASE"})
                    return
                recorded = []

                def events():
                    doc_text = text
                    if images and len(text) < 200:   # a photo or a scan: the intake agent needs its words
                        yield {"type": "agent", "id": "orchestrator", "status": "working", "note": "Reading your document's pages"}
                        try:
                            doc_text = (text + "\n\n" + agent.transcribe_images(images)).strip()
                        except Exception as e:
                            yield {"type": "agent", "id": "orchestrator", "status": "failed", "note": f"Could not read the document: {e}"}
                            yield {"type": "done"}
                            return
                    yield from agent.agent_room(situation, doc_text, doc_ids, mode=mode, language=language)

                def recording():
                    failed = False
                    for event in events():
                        recorded.append(event)
                        failed = failed or event.get("status") == "failed"
                        if key and event.get("type") == "done" and not failed:
                            cache.put(key, recorded)
                        yield event
                with lock:
                    self._stream(recording())
                self._log(200, {"route": "ERROR" if any(e.get("status") == "failed" for e in recorded) else "CASE"})
                return

            if self.path == "/api/fill":
                labels = data.get("labels") if isinstance(data.get("labels"), list) else []
                text = str(data.get("text") or "")
                if not labels or not text.strip():
                    self._json(400, {"answer": "Give a document and a form to fill.", "route": "ERROR", "values": {}})
                    return
                self._json(200, cached_task("fill", {"labels": labels, "text": hashlib.sha256(text.encode("utf-8")).hexdigest()},
                                            lambda: guarded(lambda: agent.fill_fields(labels, text))))
                return

            if self.path == "/api/video":
                frames = [f for f in (data.get("frames") or []) if isinstance(f, dict)][:W.WakeelAgent.VIDEO_MAX_FRAMES]
                frames = [f for f in frames if isinstance(f.get("data"), str) and len(f["data"]) < 600000]
                if not frames:
                    self._json(400, {"answer": "Choose a video first.", "route": "ERROR", "sources": []})
                    return
                context = str(data.get("context") or "").strip()[:2000]
                digest = hashlib.sha256("".join(f["data"] for f in frames).encode("utf-8")).hexdigest()
                self._json(200, cached_task("video", {"frames": digest, "context": context, "mode": mode, "language": language},
                                            lambda: guarded(lambda: agent.video_evidence(frames, context, mode=mode, language=language))))
                return

            if self.path == "/api/speak":
                text = str(data.get("text") or "").strip()[:W.SPEECH_MAX_CHARS]
                if not text:
                    self._json(400, {"answer": "Nothing to read aloud.", "route": "ERROR"})
                    return
                try:
                    audio = speech(text, language, "formal" if data.get("tone") == "formal" else "plain")
                except Exception as e:
                    self._json(503, {"answer": f"The voice could not be made: {W._api_error_message(e)}", "route": "ERROR"})
                    return
                self._send(200, audio, "audio/mpeg")
                return

            if self.path == "/api/moot":
                facts = str(data.get("facts") or "").strip()
                if not facts:
                    self._json(400, {"answer": "Give the facts of the matter first.", "route": "ERROR", "sources": []})
                    return
                history = data.get("history") if isinstance(data.get("history"), list) else []
                args = (facts, str(data.get("side") or ""), str(data.get("hearing") or ""), history, bool(data.get("finish")))
                suggest = bool(data.get("suggest"))
                # Keyed on the whole hearing so far: a rehearsed hearing (the demo's) replays, and a new answer is a new turn.
                self._json(200, cached_task("moot", {"facts": facts, "side": args[1], "hearing": args[2], "history": history,
                                                     "finish": args[4], "mode": mode, "language": language,
                                                     **({"suggest": True} if suggest else {})},
                                            lambda: guarded(lambda: agent.moot_turn(*args, mode=mode, language=language, suggest=suggest))))
                return

            if self.path == "/api/citator":
                citation = str(data.get("citation") or "").strip()
                refs = data.get("refs") if isinstance(data.get("refs"), dict) else {}
                if not citation or not refs:
                    self._json(400, {"answer": "Choose a reported case first.", "route": "ERROR", "sources": []})
                    return
                self._json(200, cached_task("citator", {"citation": citation, "refs": refs},
                                            lambda: guarded(lambda: agent.citator(citation, refs))))
                return

            if self.path == "/api/extract":
                # A file for the workspace's vault: its text, read in memory and sent back. Nothing is stored here;
                # the vault lives in the user's browser.
                try:
                    text, images = read_upload(data)
                except ValueError as e:
                    self._json(400, {"answer": str(e), "route": "ERROR"})
                    return
                ocr = bool(images) and len(text.strip()) < 200
                if ocr:
                    with lock:
                        try:
                            text = (text + "\n\n" + agent.transcribe_images(images)).strip()
                        except Exception as e:
                            self._json(200, {"answer": f"Could not read the pages: {e}", "route": "ERROR"})
                            return
                text = text.strip()
                if not text:
                    self._json(400, {"answer": "No readable text was found in this file.", "route": "ERROR"})
                    return
                self._json(200, {"filename": str(data.get("filename") or "Document")[:120], "text": text[:60000],
                                 "chars": len(text), "ocr": ocr, "route": "EXTRACT"})
                return

            if self.path == "/api/judgment/ask":
                doc_id, question = str(data.get("doc_id") or ""), str(data.get("question") or "").strip()
                if not doc_id or not question:
                    self._json(400, {"answer": "Choose a judgment and type a question.", "route": "ERROR", "sources": []})
                    return
                self._json(200, cached_task(
                    "judgment_ask", {"doc_id": doc_id, "question": question, "mode": mode, "language": language},
                    lambda: guarded(lambda: agent.ask_judgment(doc_id, question, mode=mode, language=language))))
                return

            if self.path == "/api/feedback":
                if activity:
                    activity.add(kind="feedback", helpful=bool(data.get("helpful")), route=str(data.get("route") or "")[:20],
                                 facts_checked=int(data.get("facts_checked") or 0), facts_missing=int(data.get("facts_missing") or 0))
                self._json(200, {"ok": True})
                return

            if self.path == "/api/workflow":
                kind = str(data.get("kind") or "")
                facts = data.get("facts") if isinstance(data.get("facts"), dict) else {}
                doc_id = str(data.get("doc_id") or "")
                if kind not in W.WORKFLOWS:
                    self._json(400, {"answer": "Choose a workflow.", "route": "ERROR", "sources": []})
                    return
                if not any(str(v).strip() for v in facts.values()) and not doc_id:
                    self._json(400, {"answer": "Describe your situation first.", "route": "ERROR", "sources": []})
                    return
                self._json(200, cached_task(
                    "workflow", {"kind": kind, "facts": facts, "doc_id": doc_id, "mode": mode, "language": language},
                    lambda: guarded(lambda: agent.run_workflow(kind, facts, mode=mode, language=language, doc_id=doc_id))))
                return

            if self.path == "/api/draft":
                kind = str(data.get("kind") or "")
                facts = data.get("facts") if isinstance(data.get("facts"), dict) else {}
                if kind not in W.DRAFT_KINDS:
                    self._json(400, {"answer": "Choose a document type.", "route": "ERROR", "sources": []})
                    return
                # Optionally one of the user's own documents, whose layout and tone the draft follows.
                sample = str(data.get("sample") or "").strip()[:W.STYLE_SAMPLE_CHARS]
                kwargs = {"sample": sample} if sample else {}
                self._json(200, cached_task(
                    "draft", {"kind": kind, "facts": facts, "mode": mode, "language": language,
                              **({"sample": hashlib.sha256(sample.encode("utf-8")).hexdigest()} if sample else {})},
                    lambda: guarded(lambda: agent.draft_document(kind, facts, mode=mode, language=language, **kwargs))))
                return

            if self.path == "/api/summarize" and not data.get("doc_id"):
                # A judgment the user uploads or pastes, rather than one from the library.
                try:
                    text, images = read_upload(data)
                except ValueError as e:
                    self._json(400, {"answer": str(e), "route": "ERROR", "sources": []})
                    return
                self._json(200, guarded(lambda: agent.summarize_document(text, images, mode=mode, language=language)))
                return

            if self.path in ("/api/summarize", "/api/compare"):
                ids = [data.get("doc_id")] if self.path == "/api/summarize" else [data.get("doc_a"), data.get("doc_b")]
                if not all(isinstance(i, str) and i for i in ids):
                    self._json(400, {"answer": "Choose a judgment first.", "route": "ERROR", "sources": []})
                    return
                if self.path == "/api/summarize":
                    run = lambda: agent.summarize_judgment(ids[0], mode=mode, language=language)
                else:
                    run = lambda: agent.compare_judgments(ids[0], ids[1], mode=mode, language=language)
                self._json(200, cached_task(self.path.rsplit("/", 1)[1],
                                            {"ids": ids, "mode": mode, "language": language},
                                            lambda: guarded(run)))
                return

            message = (data.get("message") or "").strip()
            if not message:
                self._json(400, {"answer": "Empty message.", "route": "ERROR"})
                return

            thread_id = data.get("thread_id") or str(uuid.uuid4())
            key = None
            if cache and first_turn(thread_id):
                key = cache.key("chat", {"message": message, "mode": mode, "language": language})
            hit = cache.get(key) if key else None

            # The agent and its checkpointer are not thread-safe; serialise turns.
            if self.path == "/api/chat/stream":
                if hit is not None:
                    def replay():
                        words = any(e.get("type") == "chunk" for e in hit)   # recorded with the words: they set the pace
                        for event in hit:
                            if event.get("type") == "step":
                                if simulate and not words and event.get("node") in SIMULATED_STEP_SECONDS:
                                    time.sleep(random.uniform(*SIMULATED_STEP_SECONDS[event["node"]]))
                                else:
                                    time.sleep(replay_delay)
                            elif event.get("type") == "chunk":
                                time.sleep(0.05 if simulate else 0.012)
                            elif event.get("type") == "stage":
                                time.sleep(0.4 if simulate and event.get("status") == "working" else 0.05)
                            yield event
                    self._stream(replay())
                    self._cached = True
                    self._log(200, hit[-1] if hit else {})
                    return
                recorded = []

                def recording():
                    for event in watched(lambda: turn_events(message, thread_id, mode, language)):
                        recorded.append(event)
                        # Save before sending the final event, so a repeat that arrives
                        # the moment this answer finishes already finds it.
                        if key and event.get("type") == "result" and _usable(event):
                            cache.put(key, recorded)
                        yield event
                with lock:
                    self._stream(recording())
                self._log(200, recorded[-1] if recorded else {"route": "ERROR"})
                return

            if hit is not None:
                self._json(200, next((e for e in hit if e.get("type") == "result"), hit[-1]))
                return
            with lock:
                try:
                    result = agent.run_detailed(message, thread_id=thread_id,
                                                mode=mode, language=language)
                except Exception as e:
                    result = {"answer": f"Error: {e}", "route": "ERROR"}
            if key and _usable(result):
                cache.put(key, [{"type": "result", **result}])
            self._json(200, result)

    return Handler


def build_agent():
    W.require_api_key()
    return W.WakeelAgent(W.make_chat_model(temperature=0), second_opinion=W.make_second_opinion(0))


def main():
    ap = argparse.ArgumentParser(description="Web UI for Wakeel AI")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-index", action="store_true", help="skip indexing pdfs/ at startup")
    ap.add_argument("--no-activity", action="store_true",
                    help="do not keep the usage log (features used and timings; never content)")
    ap.add_argument("--cache", nargs="?", const="demo_cache", default=None, metavar="DIR",
                    help="save answers and replay them for repeat questions (default folder: demo_cache)")
    ap.add_argument("--fast-cache", action="store_true",
                    help="replay saved answers at once; by default they take about as long as the model did")
    args = ap.parse_args()

    if not args.no_index:
        print("Checking for documents to index...")
        W.index_pdfs()

    print(f"Initializing {W.PROVIDERS[W.LLM_PROVIDER]['label']} model {W.LLM_MODEL}...")
    agent = build_agent()

    # Load the embedding model and vector store now, so the first question does not
    # spend ~15 seconds on it in front of an audience.
    print("Loading the search model...")
    try:
        W.get_embed_model()
        W.get_collection()
    except Exception as e:
        print(f"[warn] search model not preloaded, the first question will be slower: {e}")

    cache = AnswerCache(args.cache) if args.cache else None
    if cache:
        print(f"Answer cache on: {cache.folder} ({len(list(cache.folder.glob('*.json')))} saved answers)")
    activity = None if args.no_activity else ActivityLog("logs/activity.jsonl")
    handler = make_handler(agent, lambda: W.index_documents_tool("index"), cache=cache, activity=activity,
                           simulate=bool(cache) and not args.fast_cache)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"\nWakeel AI is running at http://{args.host}:{args.port}  (Ctrl-C to stop)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
