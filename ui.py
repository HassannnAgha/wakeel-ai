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
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

import wakeel_ai as W

APP_PATHS = ("/", "/index.html", "/ask", "/summarize", "/compare", "/explain", "/draft",
             "/guides", "/calculate", "/library", "/matters", "/developers")
POST_PATHS = ("/api/chat", "/api/chat/stream", "/api/summarize", "/api/compare", "/api/explain",
              "/api/explain/ask", "/api/draft", "/api/table", "/api/table/ask", "/api/strategy", "/api/workflow", "/api/feedback")

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
           display: flex; align-items: center; gap: 28px; padding: 0 28px; height: 56px; }
  .mark { font-size: 22px; text-decoration: none; }
  nav { display: flex; gap: 22px; height: 100%; }
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
  .step { display: grid; grid-template-columns: 34px 110px 1fr 54px; gap: 10px; padding: 8px 0;
          border-bottom: 1px solid var(--line); align-items: baseline; }
  .step .n { font-family: var(--display); color: var(--deep); }
  .step .k { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; }
  .step .v { color: var(--muted); }
  .step .ms { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; }
  .step.pending .k::after { content: "_"; animation: blink 1s steps(1) infinite; }
  .step.planned .k, .step.planned .v, .step.planned .n { color: var(--muted); opacity: .55; }
  .step.planned .ms { font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); opacity: .7; }
  .step.fail .k { color: #b42318; }
  @keyframes blink { 50% { opacity: 0; } }
  @media (max-width: 640px) { .step { grid-template-columns: 28px 1fr 48px; } .step .v { grid-column: 2 / 4; } }

  .composer { position: fixed; left: 0; right: 0; bottom: 0; background: var(--beige); border-top: 1px solid var(--line);
              padding: 14px 28px; display: none; }
  body.p-ask .composer { display: block; }
  .composer form { max-width: 860px; margin: 0 auto; display: flex; gap: 0; }
  .composer textarea { flex: 1; resize: none; border: 1px solid var(--ink); border-right: 0; padding: 11px 14px;
                       min-height: 46px; max-height: 160px; outline: none; background: var(--white); border-radius: 0; }
  .composer .btn { height: auto; }
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
  .scrim { position: fixed; inset: 0; z-index: 35; background: rgba(0,0,0,.25); display: none; }
  body.panel-open .scrim { display: block; }
</style>
</head>
<body>
<header>
  <a href="/" data-link class="mark display">Wakeel AI</a>
  <nav>
    <a href="/ask" data-link data-nav="ask">Ask</a>
    <a href="/summarize" data-link data-nav="summarize">Summarize</a>
    <a href="/compare" data-link data-nav="compare">Compare</a>
    <a href="/explain" data-link data-nav="explain">Explain</a>
    <a href="/draft" data-link data-nav="draft">Draft</a>
    <a href="/guides" data-link data-nav="guides">Guides</a>
    <a href="/calculate" data-link data-nav="calculate">Calculate</a>
    <a href="/library" data-link data-nav="library">Library</a>
    <a href="/matters" data-link data-nav="matters">Matters</a>
  </nav>
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
        <h1 class="display">Wakeel<br>AI</h1>
        <p>Legal research on real court judgments. Every answer is written from the judgments
           themselves and linked to the paragraph it came from, for citizens and lawyers, in
           English, Urdu or Roman Urdu.</p>
        <div class="ctas">
          <a href="/ask" data-link class="btn">Ask a question</a>
          <a href="/guides" data-link class="btn line">Know your rights</a>
          <a href="/library" data-link class="btn line">Browse the library</a>
        </div>
        <div class="court">
          <div><span>Court</span><b>Lahore High Court</b></div>
          <div><span>Year</span><b>2023</b></div>
          <div><span>Answers in</span><b>EN · اردو · Roman</b></div>
        </div>
      </div>
      <div class="home-r">
        <div class="eyebrow">Knowledge base</div>
        <div class="bignums">
          <div><b class="display" data-stat="judgments">–</b><span>Judgments</span></div>
          <div><b class="display" data-stat="passages">–</b><span>Indexed passages</span></div>
          <div><b class="display" data-stat="citations">–</b><span>Case citations</span></div>
          <div><b class="display" data-stat="provisions">–</b><span>Statutory provisions</span></div>
        </div>
        <div class="tasks">
          <a href="/ask" data-link><span class="n">01</span><span class="t display">Ask</span><span class="d">A question, answered with citations</span></a>
          <a href="/summarize" data-link><span class="n">02</span><span class="t display">Summarize</span><span class="d">One judgment, section by section</span></a>
          <a href="/compare" data-link><span class="n">03</span><span class="t display">Compare</span><span class="d">Two judgments, side by side</span></a>
          <a href="/explain" data-link><span class="n">04</span><span class="t display">Explain</span><span class="d">Your FIR, notice or court order, in plain words</span></a>
          <a href="/draft" data-link><span class="n">05</span><span class="t display">Draft</span><span class="d">Notices and applications; hearing prep, timelines, client letters</span></a>
          <a href="/calculate" data-link><span class="n">06</span><span class="t display">Calculate</span><span class="d">Limitation periods and inheritance shares</span></a>
        </div>
        <a href="/developers" data-link class="devlink">For developers: the API →</a>
        <div class="specs">
          <div><span>Search</span><b>Hybrid</b></div>
          <div><span>Languages</span><b>3</b></div>
          <div><span>Model</span><b><i class="dot"></i><span id="model-short" style="display:inline;font-size:inherit;letter-spacing:0;text-transform:none;color:inherit">–</span></b></div>
        </div>
      </div>
    </div>
    <div class="trust">
      <div class="trust-in">
        <div class="eyebrow">Measured, not claimed</div>
        <h2 class="display">Why you can trust it</h2>
        <div class="trust-grid">
          <div><b>99.5%</b><span>of test questions find the right judgment in the top ten results</span></div>
          <div><b>0.97</b><span>mean reciprocal rank: the right judgment almost always comes first</span></div>
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
      <div class="insights" id="lib-activity" hidden></div>
    </div>
  </section>
</main>

<div class="composer">
  <form id="form">
    <textarea id="input" rows="1" placeholder="Ask about Pakistani case law, in English, اردو or Roman Urdu"></textarea>
    <button class="mic" id="ask-mic" type="button" hidden></button>
    <button class="btn" id="send" type="submit">Ask</button>
  </form>
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
function secs(ms) { return (ms / 1000).toFixed(1) + 's'; }
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
const PAGES = ['home', 'ask', 'summarize', 'compare', 'explain', 'draft', 'guides', 'calculate', 'library', 'matters', 'developers'];
function pageFromPath(p) { const n = p.replace(/^\/+|\/+$/g, ''); return PAGES.includes(n) ? n : 'home'; }
function showPage(name) {
  document.querySelectorAll('main > section').forEach(s => s.classList.toggle('on', s.dataset.page === name));
  document.querySelectorAll('nav a').forEach(a => a.classList.toggle('on', a.dataset.nav === name));
  const onLink = document.querySelector('nav a.on');
  if (onLink && onLink.parentElement.scrollWidth > onLink.parentElement.clientWidth) onLink.scrollIntoView({block: 'nearest', inline: 'center'});
  document.body.className = document.body.className.replace(/\bp-\w+/g, '').trim() + ' p-' + name;
  closePanel();
  window.scrollTo(0, 0);
  if (name === 'ask') { renderThreadIntro(); $('input').focus(); }
  if (name === 'summarize' || name === 'compare' || name === 'library') loadJudgments();
  if (name === 'draft' && !draftKind) renderPicker($('draft-kinds'), DRAFTS, null, selectDraft);
  if (name === 'guides' && !guideId) selectGuide(GUIDES[0].id);
  if (name === 'calculate' && !calcId) selectCalc(CALCS[0].id);
  if (name === 'matters') renderMatters(new URLSearchParams(location.search).get('m'));
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
    if (v !== undefined) b.textContent = Number(v).toLocaleString();
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
fetch('/api/citations').then(r => r.json()).then(d => { CITES = indexCitations(d.citations || []); }).catch(() => {});
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
      escapeHtml(name(d)) + '</button></span></li>').join('') + '</ul>');
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
  const text = sentence.replace(/\[S[^\]]*\]/g, ' ').replace(/[*_`#>|]/g, ' ');
  (text.match(/[“"]([^”"]{12,240})[”"]/g) || []).forEach(q =>
    out.push({kind: 'quotation', label: q.trim(), test: hay => hay.includes(factText(q.slice(1, -1)).trim())}));
  (text.match(/\b\d[\d,]*(?:\.\d+)?(?:\s*-\s*[A-Za-z]{1,2}\b)?/g) || []).forEach(raw => {
    const m = raw.replace(/,(?=\d)/g, '').match(/^(\d+(?:\.\d+)?)(?:\s*-\s*([A-Za-z]{1,2}))?$/);
    if (!m || /^(st|nd|rd|th)$/i.test(m[2] || '')) return;
    const num = m[1], suffix = m[2] || '';
    if (num.length === 1 && !suffix) return;   // "2 options", list counts: too noisy to check
    out.push({kind: suffix ? 'section' : 'number', label: raw.trim(), test: hay => hasNumber(hay, num.replace('.', '\\.'), suffix)});
  });
  const words = (text.match(/[A-Za-z][A-Za-z'-]*/g) || []).map(w => w.replace(/['’]s$/, ''));
  const common = x => NOT_NAMES.has(x) || lowerAll.includes(' ' + x + ' ');
  const isName = (w, i) => {
    if (!/^[A-Z][a-z]/.test(w) || w.length < 4) return false;
    const lw = w.toLowerCase();
    if (common(lw) || common(lw.replace(/s$/, ''))) return false;   // a common word, capitalised
    // The first word is capitalised anyway ("Consequently", "Agar"); count it as a name only
    // when the next word is capitalised too ("Rashida Bibi").
    return i > 0 || !!(words[1] && /^[A-Z][a-z]/.test(words[1]));
  };
  // Neighbouring name words make one name ("Kamran Shah"); every word of it must be in the passage.
  let run = [];
  const flush = () => {
    if (!run.length) return;
    const parts = run.map(w => w.toLowerCase());
    out.push({kind: 'name', label: run.join(' '), test: hay => parts.every(p => hay.includes(p))});
    run = [];
  };
  words.forEach((w, i) => { if (isName(w, i)) run.push(w); else flush(); });
  flush();
  const seen = new Set();
  return out.filter(f => !seen.has(f.kind + f.label.toLowerCase()) && seen.add(f.kind + f.label.toLowerCase()));
}
function checkFacts(answer, sources) {
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
      if (f.test(hay)) return;
      const elsewhere = Object.keys(byN).map(Number).filter(n => f.test(byN[n]));
      result.missing.push({kind: f.kind, label: f.label, cited: ns, elsewhere});
    });
  });
  return result;
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

function renderResult(container, data, key, query, opts) {
  opts = opts || {};
  const sources = data.sources || [];
  const isError = data.route === 'ERROR';
  results[key] = {sources, query};
  const ans = el('div', 'md' + (data.route === 'SQ' ? ' numbered' : '') + (isError ? ' error-text' : ''));
  ans.innerHTML = renderMarkdown(data.answer || 'No answer produced.', key, sources);
  markSeverity(ans);
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
  const facts = data.grounded ? checkFacts(data.answer, sources) : {checked: 0, missing: []};
  if (facts.checked) {
    const f = el('span');
    f.innerHTML = facts.missing.length
      ? '<span class="warn">' + plural(facts.missing.length, 'detail') + ' to check</span>'
      : '<span><b>Facts checked</b> ' + facts.checked + ' of ' + facts.checked + '</span>';
    f.title = 'Every name, section, number and quotation in a cited sentence is looked for in the passage it cites. No AI.';
    g.appendChild(f);
  }
  g.appendChild(copy);
  if (!opts.noSave) g.appendChild(saveButton(() => ({kind: pageKind(), title: opts.print || query || 'Answer', query, data})));
  if (!opts.noSave) g.appendChild(feedbackButtons(data, facts));
  if (opts.print) {
    const pdf = el('button', 'copy', 'Save as PDF');
    pdf.type = 'button';
    pdf.style.marginLeft = '0';
    pdf.addEventListener('click', () => printResult(container, opts.print));
    g.appendChild(pdf);
  }
  container.appendChild(g);
  if (facts.missing.length) renderFactNote(container, ans, facts.missing, sources, query);
  if (!opts.whole) renderSourceList(container, sources, key, cited);
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
    '<button class="btn line" data-act="time">Timeline</button></div>' +
    precHtml +
    '<div class="eyebrow">Passages</div>' +
    '<ul class="plist">' + (j.passages || []).map(p =>
      '<li><b>' + (p.paragraph && p.paragraph !== '0' ? '¶' + escapeHtml(p.paragraph) : 'Head') + '</b><span dir="auto">' +
      escapeHtml((p.text || '').slice(0, 220)) + ((p.text || '').length > 220 ? '…' : '') + '</span></li>').join('') + '</ul>');
  $('panel').querySelector('[data-act="sum"]').addEventListener('click', () => { go('/summarize'); selectForSummary(docId); });
  $('panel').querySelector('[data-act="cmp"]').addEventListener('click', () => { go('/compare'); loadJudgments().then(() => { $('cmp-a').value = docId; }); });
  $('panel').querySelector('[data-act="time"]').addEventListener('click', () => timelineOf(docId, caseTitle(j)));
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
    MODE_LABEL[segValue('mode')] + '</span><span>' + LANG_LABEL[segValue('language')] + '</span><span class="time"></span></div>' +
    '<details class="trace" open><summary>Working</summary><ol class="steps"></ol></details><div class="body"></div>';
  turn.querySelector('.q').textContent = text;
  t.appendChild(turn);
  turn.scrollIntoView({block: 'start', behavior: 'smooth'});
  const steps = turn.querySelector('.steps');
  const t0 = performance.now();
  let last = t0, pending = null, n = 0;
  const timer = setInterval(() => { turn.querySelector('.time').textContent = secs(performance.now() - t0); }, 100);
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
      const li = row(fail ? 'fail' : '', k, v, secs(now - last));
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
    turn.pending('Writing', MODE_LABEL[segValue('mode')] + ' · ' + LANG_LABEL[segValue('language')]);
  } else if (ev.node === 'answer' || ev.node === 'summarize' || ev.node === 'compare') {
    if (ev.route === 'ERROR') turn.done('Stopped', 'The language model did not respond', true);
    else turn.done('Written', {answer: 'Answer with citations', summarize: 'Summary by section', compare: 'Side-by-side comparison'}[ev.node]);
  }
}

function onResult(turn, data) {
  clearInterval(turn.timer);
  turn.clearPending();
  if (data.route === 'ERROR' && !turn.el.querySelector('.step.fail')) turn.done('Stopped', 'See the message below', true);
  const total = performance.now() - turn.t0;
  turn.el.querySelector('.time').textContent = secs(total);
  const tr = turn.el.querySelector('.trace');
  if (turn.count()) { tr.querySelector('summary').textContent = plural(turn.count(), 'step') + ' · ' + secs(total); tr.open = false; }
  else tr.remove();
  renderResult(turn.el.querySelector('.body'), data, turn.key, data.restated_query || turn.restated || turn.text);
  if (data.route !== 'ERROR') remember(turn.text, data);
}

async function ask(text) {
  if (busy || !text.trim()) return;
  if (pageFromPath(location.pathname) !== 'ask') go('/ask');
  busy = true; $('send').disabled = true;
  const turn = makeTurn(text);
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
        if (ev.type === 'step') onStep(turn, ev);
        else if (ev.type === 'result') { got = true; onResult(turn, ev); }
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
  const t0 = performance.now();
  const tick = setInterval(() => { statusEl.textContent = workingText + ' · ' + secs(performance.now() - t0); }, 100);
  let data;
  try {
    data = await (await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...payload, mode: segValue('mode'), language: segValue('language')})})).json();
  } catch (e) { data = {route: 'ERROR', answer: 'Could not reach the server: ' + e.message, sources: []}; }
  clearInterval(tick);
  statusEl.className = 'status';
  statusEl.textContent = (data.route === 'ERROR' ? 'Stopped' : 'Done') + ' · ' + secs(performance.now() - t0) + ' · ' +
    MODE_LABEL[segValue('mode')] + ' · ' + LANG_LABEL[segValue('language')];
  renderResult(resultEl, data, 't' + (++seq), data.restated_query || '', opts || {whole: true});
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
    let ans;
    try {
      ans = await (await fetch('/api/table/ask', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: q, table: data.answer, mode: segValue('mode'), language: segValue('language')})})).json();
    } catch (err) { ans = {route: 'ERROR', answer: 'Could not reach the server: ' + err.message, sources: []}; }
    item.querySelector('.status').remove();
    renderResult(item, {...ans, sources: data.sources || []}, key, q, {whole: true});
  });
}
$('rt-go').addEventListener('click', async () => {
  const st = $('rt-status'), out = $('rt-result');
  // Files are read in the browser first; wait for any still reading.
  for (let i = 0; i < 100 && rtFiles.some(f => !f.data); i++) await new Promise(r => setTimeout(r, 50));
  $('rt-go').disabled = true; out.innerHTML = '';
  st.className = 'status working';
  const t0 = performance.now();
  const what = [rtSelected.length ? plural(rtSelected.length, 'judgment') : '', rtFiles.length ? plural(rtFiles.length, 'document') : '']
    .filter(Boolean).join(' and ');
  const tick = setInterval(() => { st.textContent = 'Reading ' + what + ' · ' + secs(performance.now() - t0); }, 100);
  let data;
  try {
    data = await (await fetch('/api/table', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({doc_ids: rtSelected, columns: rtCols, mode: segValue('mode'), language: segValue('language'),
                            files: rtFiles.map(f => ({filename: f.filename, mime: f.mime, data: f.data}))})})).json();
  } catch (e) { data = {route: 'ERROR', answer: 'Could not reach the server: ' + e.message}; }
  clearInterval(tick);
  st.className = 'status';
  st.textContent = (data.route === 'ERROR' ? 'Stopped' : 'Done') + ' · ' + secs(performance.now() - t0);
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
    let data;
    try {
      data = await (await fetch('/api/explain/ask', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: documentText, question: q, mode: segValue('mode'), language: segValue('language')})})).json();
    } catch (err) { data = {route: 'ERROR', answer: 'Could not reach the server: ' + err.message, sources: []}; }
    item.querySelector('.status').remove();
    renderResult(item, data, 'f' + (++seq), q, {whole: true});
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
const KIND_LABEL = {ask: 'Answer', summarize: 'Summary', compare: 'Comparison', table: 'Review table', explain: 'Document explained',
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
  panel.querySelector('h3').textContent = item.title;
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
  m.items.push({id: newId(), ...item, saved: Date.now()});
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
  {id: 'client_letter', title: 'Client update letter', line: 'Explain a hearing to a client in plain words', endpoint: '/api/workflow', fields: [
    ['Client name'], ['Case'], ['What happened at the hearing', true], ['Next date and what the client must do', true]]},
];
let draftKind = null;
function selectDraft(id) {
  const d = DRAFTS.find(x => x.id === id);
  if (!d) return;
  draftKind = id;
  renderPicker($('draft-kinds'), DRAFTS, id, selectDraft);
  $('draft-form').innerHTML = d.fields.map((f, i) =>
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
                             {whole: false, print: d.title});
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
    const n = (id, label, max) => '<label class="eyebrow" for="' + id + '">' + label + '</label><input class="field" type="number" min="0" max="' + max + '" value="0" id="' + id + '">';
    $('calc').innerHTML =
      '<div class="eyebrow">Hanafi (Sunni) law · common heirs</div><h2 class="display">Inheritance shares</h2>' +
      '<div class="form two">' +
      '<label class="eyebrow" for="ih-dec">The deceased was</label><select class="field" id="ih-dec"><option value="male">a man</option><option value="female">a woman</option></select>' +
      '<label class="eyebrow" for="ih-estate">Estate value (optional, PKR)</label><input class="field" type="number" min="0" id="ih-estate">' +
      '<label class="eyebrow" for="ih-spouse" id="ih-spouse-l">Wives</label><input class="field" type="number" min="0" max="4" value="0" id="ih-spouse">' +
      n('ih-sons', 'Sons', 30) + n('ih-daughters', 'Daughters', 30) +
      '<label class="eyebrow" for="ih-father">Father alive</label><select class="field" id="ih-father"><option value="0">No</option><option value="1">Yes</option></select>' +
      '<label class="eyebrow" for="ih-mother">Mother alive</label><select class="field" id="ih-mother"><option value="0">No</option><option value="1">Yes</option></select>' +
      n('ih-brothers', 'Full brothers', 30) + n('ih-sisters', 'Full sisters', 30) +
      '</div><div id="ih-out" class="calcout"></div>';
    const update = () => {
      const male = $('ih-dec').value === 'male';
      $('ih-spouse-l').textContent = male ? 'Wives (0 to 4)' : 'Husband (0 or 1)';
      $('ih-spouse').max = male ? 4 : 1;
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
    $('calc').querySelectorAll('input, select').forEach(x => x.addEventListener('input', update));
    update();
  }
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
    barList('Passages by part of the judgment', 'How the chunker labelled the 770 passages.', d.sections, {unit: 'passage'}));
  $('lib-sum').textContent = 'Counted across every judgment in the library, with no AI involved. Hover a bar for its exact count.';
}

document.querySelectorAll('#lib-tabs button').forEach(b => b.addEventListener('click', () => {
  libTab = b.dataset.tab;
  document.querySelectorAll('#lib-tabs button').forEach(x => x.classList.toggle('on', x === b));
  $('lib-judgments').hidden = libTab !== 'judgments';
  $('lib-statutes').hidden = libTab !== 'statutes';
  $('lib-insights').hidden = libTab !== 'insights';
  $('lib-activity').hidden = libTab !== 'activity';
  $('lib-search').hidden = libTab === 'insights' || libTab === 'activity';
  $('lib-search').placeholder = libTab === 'statutes' ? 'Filter by number, title or words in the text' : 'Filter by party, case number, court or provision';
  $('lib-search').value = '';
  if (libTab === 'statutes') renderStatutes('');
  else if (libTab === 'insights') renderInsights();
  else if (libTab === 'activity') renderActivity();
  else { renderLibrary(''); if (stats) paintStats(); }
}));
$('lib-search').addEventListener('input', e => libTab === 'statutes' ? renderStatutes(e.target.value) : renderLibrary(e.target.value));

// ===========================================================================
// start
// ===========================================================================
showPage(pageFromPath(location.pathname));
// Deep link for a scripted demo: /ask?q=...&mode=lawyer&language=ur asks straight away.
const params = new URLSearchParams(location.search);
['mode', 'language'].forEach(id => {
  const v = params.get(id);
  if (v && document.querySelector('#' + id + ' button[data-value="' + v + '"]')) setSeg(id, v);
});
if (params.get('q')) ask(params.get('q'));
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
                "/api/strategy": "Workflows", "/api/workflow": "Workflows"}

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


def make_handler(agent, index_fn, stats_fn=corpus_stats, cache=None, replay_delay=0.35, activity=None):
    """Build the request handler. agent/index_fn/stats_fn are injected so tests can stub them.

    cache: an AnswerCache, or None. Only the first turn of a conversation is cached; a
    follow-up depends on the turns before it.
    """

    lock = threading.Lock()
    seen_threads = set()

    def first_turn(thread_id: str) -> bool:
        with lock:
            fresh = thread_id not in seen_threads
            seen_threads.add(thread_id)
        return fresh

    def cached_task(kind, payload, run):
        """Run a non-chat task through the cache."""
        key = cache.key(kind, payload) if cache else None
        hit = cache.get(key) if key else None
        if hit is not None:
            return hit
        result = run()
        if key and _usable(result):
            cache.put(key, result)
        return result

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
                        for event in hit:
                            if event.get("type") == "step":
                                time.sleep(replay_delay)
                            yield event
                    self._stream(replay())
                    self._cached = True
                    self._log(200, hit[-1] if hit else {})
                    return
                recorded = []

                def recording():
                    for event in turn_events(message, thread_id, mode, language):
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
    return W.WakeelAgent(W.make_chat_model(temperature=0))


def main():
    ap = argparse.ArgumentParser(description="Web UI for Wakeel AI")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-index", action="store_true", help="skip indexing pdfs/ at startup")
    ap.add_argument("--no-activity", action="store_true",
                    help="do not keep the usage log (features used and timings; never content)")
    ap.add_argument("--cache", nargs="?", const="demo_cache", default=None, metavar="DIR",
                    help="save answers and replay them for repeat questions (default folder: demo_cache)")
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
    handler = make_handler(agent, lambda: W.index_documents_tool("index"), cache=cache, activity=activity)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"\nWakeel AI is running at http://{args.host}:{args.port}  (Ctrl-C to stop)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
