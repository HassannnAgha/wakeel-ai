"""Build statutes.json: the text of the provisions the statute linker can show.

    .venv/Scripts/python.exe build_statutes.py          # the hand-checked provisions only
    .venv/Scripts/python.exe build_statutes.py --all    # the whole PPC and Constitution, and the PDF Acts
    .venv/Scripts/python.exe build_statutes.py --acts               # refresh the PDF Acts only
    .venv/Scripts/python.exe build_statutes.py --act cpc [file.pdf] # one of them

The CrPC, CPC and family laws come from official PDFs rather than web pages: see PDF_ACTS
and crpc_lines().

Downloads each source page, cuts out the listed sections and articles word for word, removes
the amendment footnote markers (keeping the amended wording), and records the source URL
and date. Nothing is generated or paraphrased. Check the output against pakistancode.gov.pk
before relying on it; the source pages may lag behind recent amendments.

To add a provision, add (act, number, title-start, next-number) to PROVISIONS and rerun.
"""
import datetime
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

SOURCES = {
    "ppc": ("Pakistan Penal Code, 1860", "http://www.pakistani.org/pakistan/legislation/1860/actXLVof1860.html"),
    "const1": ("Constitution of Pakistan, 1973", "http://www.pakistani.org/pakistan/constitution/part1.html"),
    "const2": ("Constitution of Pakistan, 1973", "http://www.pakistani.org/pakistan/constitution/part2.ch1.html"),
    "const7": ("Constitution of Pakistan, 1973", "http://www.pakistani.org/pakistan/constitution/part7.ch3.html"),
}

# (source, kind, number, start of title, the number of the next provision, cut before)
PROVISIONS = [
    ("ppc", "section", "302", "Punishment of qatl-i-amd", "303", None),
    ("ppc", "section", "324", "Attempt to commit qatl-i-amd", "325", None),
    ("ppc", "section", "392", "Punishment for robbery", "393", None),
    ("ppc", "section", "405", "Criminal breach of trust", "406", "Illustrations"),
    ("ppc", "section", "489-F", "Dishonestly issuing a cheque", "489-G", None),
    ("const1", "article", "4", "Right of individuals", "5", None),
    ("const2", "article", "9", "Security of person", "9A", None),
    ("const2", "article", "10A", "Right to fair trial", "11", None),
    ("const7", "article", "199", "Jurisdiction of High Court", "200", None),
]


def page_text(url: str) -> str:
    # The source site refuses urllib's default headers (HTTP 406).
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                                               "Accept": "text/html,application/xhtml+xml"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("latin-1")
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>|</li>|</tr>|</td>", "\n", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return re.sub(r"[ \t\xa0]+", " ", text)


def clean(block: str) -> str:
    """Drop amendment footnote markers like '133 [' and '] 133', and re-flow wrapped lines.

    Clause markers that stand on their own line, such as '(1)' or '(a)', start a new line.
    """
    block = re.sub(r"(?<!\w)\d{1,3}\s\[", "", block)
    block = re.sub(r"\]\s\d{1,3}(?!\w)", "", block)
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    out, marker = [], None
    for line in lines:
        if re.fullmatch(r"\((?:\d+[A-Z]?|[a-z]{1,2}|[ivx]+)\)", line):
            marker = line
            continue
        if marker:
            out.append(marker + " " + line)
            marker = None
        elif out and not re.match(r"(Provided|Explanation)", line):
            out[-1] += " " + line
        else:
            out.append(line)
    return re.sub(r" {2,}", " ", "\n".join(out)).strip()


#: The heading of any provision: its number alone on a line ("326." or "5"), then its title.
_HEADING = re.compile(r"\n\s*\d{1,3}[A-Z]?(?:-[A-Z]{1,2})?\.?\s*\n\s*(?:\n\s*)*[A-Z][a-z]")


def extract(text: str, number: str, title: str, nxt: str, cut):
    start = re.search(r"\n\s*" + re.escape(number) + r"\.?\s*\n?\s*" + re.escape(title), text)
    if not start:
        return None
    seg = text[start.end() - len(title):]
    # The title runs to its closing full stop or colon, sometimes across a line break.
    t = re.match(r"(.{0,240}?[.:])\s*\n", seg, re.S)
    full_title = re.sub(r"\s+", " ", t.group(1)).strip(" .:-") if t else title
    rest = seg[t.end():] if t else seg[len(title):]
    # Stop at the next provision: the expected number, or failing that (when it has been
    # omitted from the statute, like PPC 325) the next heading of any number.
    end = re.search(r"\n\s*" + re.escape(nxt) + r"\.?\s*\n", rest)
    if not end or end.start() > 12000:
        end = _HEADING.search(rest)
    body = rest[: end.start()] if end else rest[:4000]
    if cut and cut in body:
        body = body[: body.index(cut)]
    return full_title, clean(body)


CONSTITUTION_BASE = "http://www.pakistani.org/pakistan/constitution/"
CONSTITUTION_PAGES = (["part1.html", "part2.ch1.html", "part2.ch2.html"] +
                      [f"part3.ch{i}.html" for i in (1, 2, 3)] + [f"part4.ch{i}.html" for i in (1, 2, 3)] +
                      [f"part5.ch{i}.html" for i in (1, 2, 3)] + [f"part6.ch{i}.html" for i in (1, 2, 3)] +
                      ["part7.ch1.html", "part7.ch1A.html", "part7.ch2.html", "part7.ch3.html", "part7.ch3A.html",
                       "part7.ch4.html", "part8.ch1.html", "part8.ch2.html", "part9.html", "part10.html",
                       "part11.html"] + [f"part12.ch{i}.html" for i in range(1, 8)])

#: Shown with every article. The source copy is maintained by hand and may lag behind amendments.
CONSTITUTION_NOTE = ("From a published copy of the Constitution, which may lag behind the latest "
                     "amendments. Check pakistancode.gov.pk before relying on it.")
PPC_NOTE = "From a published copy of the Code. Recent amendments may not be reflected. Check pakistancode.gov.pk."

#: Any provision heading: its number alone on a line ("489-F." or "199"), then a capitalised title.
_ANY_HEADING = re.compile(r"\n\s*(\d{1,3}[A-Z]{0,2}(?:-[A-Z]{1,2})?)\.?\s*\n\s*(?:\n\s*)*(?=[A-Z][a-z'])")


def split_all(text: str):
    """Every provision on a page, as (number, title, body), in page order."""
    heads = list(_ANY_HEADING.finditer(text))
    for i, h in enumerate(heads):
        seg = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        lines = seg.split("\n")
        # The title is the first line, joined with the next while it clearly runs on
        # ("... in accordance with" / "law, etc."). Some titles end in a full stop, some don't.
        k = next((j for j, l in enumerate(lines) if l.strip()), None)
        if k is None:
            continue
        title = lines[k].strip()
        while (not re.search(r"[.:]$", title) and re.search(r"\b(with|and|of|to|the|in|for|or|by|on),?$|,$", title)
               and k + 1 < len(lines)):
            k += 1
            nxt = lines[k].strip()
            if nxt:
                title += " " + nxt
        # A title wrapped onto the next line continues in lowercase ("... from person" /
        # "concerned in proceeding ..."); join it back.
        while not re.search(r"[.:]$", title.strip()):
            j = next((x for x in range(k + 1, len(lines)) if lines[x].strip()), None)
            if j is None or not re.match(r"[a-z]", lines[j].strip()):
                break
            title += " " + lines[j].strip()
            k = j
        title = re.sub(r"\s+", " ", title).strip(" .:-")
        body = clean("\n".join(lines[k + 1:]))
        if body and len(title) < 200:
            yield h.group(1), title, body


#: Footnotes about amendments also start with a bare number on a line; their "titles" give them away.
_FOOTNOTE_TITLE = re.compile(r"^(The following|Subs|Substituted|Ins|Inserted|Added|Omitted|Renumbered|Re-numbered|"
                             r"Amended|Deleted|Rep|Repealed|Clause|Article|Para|Paragraph|Words?|Proviso|Sic|See|"
                             r"Vide|Now|Comma|Full stop|Semi-colon|Brackets?|Figures?|Section|Sections|Sub-section)\b",
                             re.I)


def real_provision(number: str, title: str, highest: int) -> bool:
    n = int(re.match(r"\d+", number).group())
    footnote = _FOOTNOTE_TITLE.match(title) or re.match(r"^For the\b", title) or "Gazette" in title
    return n <= highest and not footnote and len(title) <= 160


CRPC_ACT = "Code of Criminal Procedure, 1898"
CRPC_URL = "https://pakistancode.gov.pk/pdffiles/administrator7db1e56f0f1d39a6e67573ec6b0944e2.pdf"
CRPC_NOTE = ("From the official Pakistan Code copy of the Code. Amendment markers and footnotes are removed and "
             "the amended wording kept; provincial amendments are not shown. Check pakistancode.gov.pk.")


def crpc_lines(pdf_path: str):
    """The Code's enacted text, line by line, as lists of (word, bold).

    The PDF sets the enacted text in one size per page (12pt or 11pt Times). Footnotes and
    amendment markers are much smaller, so keeping only body-size characters drops them without
    touching the wording. Cross-headings ("First offenders") are bold italic and are dropped too.
    """
    import collections
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            sizes = collections.Counter(round(c["size"]) for c in page.chars if c["size"] >= 10)
            if not sizes:
                continue
            floor = sizes.most_common(1)[0][0] * 0.88
            # The PDF encodes hyphens as space characters, but keeps a hyphen's width: a third
            # of an em, against a quarter for a real space. Put them back ("non-bailable").
            for c in page.chars:
                if c["text"] == " " and c["size"] and abs((c["x1"] - c["x0"]) / c["size"] - 1 / 3) < 0.01:
                    c["text"] = "-"
            body = page.filter(lambda o: o.get("object_type") != "char" or o["size"] >= floor)
            words = body.extract_words(extra_attrs=["fontname"], keep_blank_chars=False, use_text_flow=True)
            words = [w for w in words if not re.fullmatch(r"Page|\d+|of", w["text"]) or w["top"] < page.height - 70]
            line, top = [], None
            for w in words:
                if top is not None and abs(w["top"] - top) > 4:
                    yield line
                    line = []
                top = w["top"]
                font = w["fontname"]
                if "BoldItalic" in font:
                    continue
                line.append((w["text"], "Bold" in font))
            if line:
                yield line


def _crpc_number(raw: str) -> str:
    """'561A' -> '561-A', the way judgments cite it."""
    m = re.fullmatch(r"(\d+)([A-Z]{0,2})", raw)
    return m.group(1) + ("-" + m.group(2) if m.group(2) else "")


def split_crpc(lines):
    """Every section of the Code as (number, title, body), from the first section to the schedules."""
    current, collecting, last = None, False, (0, "")
    for line in lines:
        # Inserted or substituted sections open with the amendment bracket: "[122. Power to ...".
        while line and line[0][0].strip("[") == "":
            line = line[1:]
        if not line:
            continue
        first = line[0][0].lstrip("[")
        if re.fullmatch(r"\d[\dl]{0,2}[A-Z]{0,2}\.", first):
            first = first.replace("l", "1")   # the source has "4l7." for 417
        line = [(first, line[0][1])] + line[1:]
        text = " ".join(w for w, _ in line)
        first, bold = line[0]
        head = re.fullmatch(r"(\d{1,3}[A-Z]{0,2})\.", first) if bold else None
        if head:
            n = re.match(r"(\d+)([A-Z]*)", head.group(1))
            key = (int(n.group(1)), n.group(2))
            # Section numbers only go up; anything else is a stray bold number, not a heading.
            if key > last and key[0] - last[0] <= 120:
                if current:
                    yield current
                last = key
                current = [head.group(1), [], []]
                collecting = True
                line = line[1:]
        elif current and all(b for _, b in line):
            if re.match(r"SCHEDULE\b", text):
                break
            if not collecting:
                continue   # a chapter heading ("CHAPTER XXXIX", "OF BAIL")
        if not current:
            continue
        rest = []
        for word, b in line:
            if collecting and b:
                current[1].append(word)
                if re.search(r"[.:]_*$", word):
                    collecting = False
            else:
                collecting = False
                rest.append(word)
        if rest:
            current[2].append(" ".join(rest))
    if current:
        yield current


def clean_crpc(body_lines) -> str:
    """Re-flow wrapped lines; drop the square brackets of amendments (keeping the words) and omission marks."""
    out = []
    for line in body_lines:
        line = line.replace("\u037e", ";")   # a Greek question mark, used as a semicolon
        line = re.sub(r"\[?\*(?:\s*\*)*\]?", "", line)
        line = re.sub(r"^\[(\d+[A-Z]?)\]\s", r"(\1) ", line)   # a substituted sub-section: "[1] Subject to"
        line = re.sub(r"[\[\]]", "", line)
        # A footnote marker set at body size, glued to the next word ("in 3official Gazette").
        line = re.sub(r"(?<![\w(])\d{1,2}(?=[a-z]{3,})", "", line).strip()
        if not line:
            continue
        starts = re.match(r"\((?:\d+(?:-?[A-Z])?|[a-z]{1,2}|[ivx]+)\)\s|(Provided|Explanation|Illustration)\b", line)
        # "... under sub-section" / "(1) shall ..." is a reference wrapped across lines, not a new clause.
        reference = out and re.search(r"\b(?:sub-)?(?:section|clause)s?,?$|\b(?:and|or|of|under|in)$", out[-1])
        if out and (not starts or reference):
            out[-1] += " " + line
        else:
            out.append(line)
    return re.sub(r"\s+([,;:.])", r"\1", re.sub(r" {2,}", " ", "\n".join(out))).strip()


OFFICIAL_NOTE = ("From the official Pakistan Code copy of the Act. Amendment markers and footnotes are removed "
                 "and the amended wording kept. Check pakistancode.gov.pk before relying on it.")

#: Acts read from official PDFs: key -> (name, source URL, note). The key is the statute
#: prefix ("cpc:115"). All use the Pakistan Code layout that crpc_lines() reads.
PDF_ACTS = {
    "crpc": (CRPC_ACT, CRPC_URL, CRPC_NOTE),
    "cpc": ("Code of Civil Procedure, 1908",
            "https://pakistancode.gov.pk/pdffiles/administrator6598dabbad120033d4d42d717dcf9755.pdf", OFFICIAL_NOTE),
    "gwa": ("Guardians and Wards Act, 1890",
            "https://pakistancode.gov.pk/pdffiles/administratord265c726d2d564a9377ed7a8e04708ae.pdf", OFFICIAL_NOTE),
    "mflo": ("Muslim Family Laws Ordinance, 1961",
             "https://pakistancode.gov.pk/pdffiles/administratoreecaf3b490e2d43d2e3b50c0c068b5d7.pdf", OFFICIAL_NOTE),
    "dmma": ("Dissolution of Muslim Marriages Act, 1939",
             "https://pakistancode.gov.pk/pdffiles/administratorfb32d6015ae887e6d6b85018961842ea.pdf", OFFICIAL_NOTE),
    # A provincial law, so not on Pakistan Code: the Ministry of Law's copy, as it applies in Punjab.
    "fca": ("Punjab Family Courts Act, 1964",
            "https://molaw.gov.pk/SiteImage/Misc/files/THE%20WEST%20PAKISTAN%20FAMILY%20COURTS%20ACT%2C%201964.pdf",
            "From the Ministry of Law's copy of the West Pakistan Family Courts Act, 1964 as it applies in Punjab. "
            "Amendment markers and footnotes are removed. Check the Punjab Code before relying on it."),
}

CROSS_HEADING_AT_END = re.compile(r"\s+(?:[A-Z]|[IVX]+)\.\s*[\u2014\u2013-]?\s*(?![^.]*\b[A-Z]\b)"
                                  r"[A-Z][A-Za-z ,'\u2019-]{3,120}\.?$")

#: Headings the PDFs set as two marginal notes ("Short title." / "commencement.").
TITLE_FIXES = {"crpc:1": "Short title and commencement"}


def build_pdf_act(key: str, pdf_path: str = None) -> dict:
    """Every section of one Act from its official PDF (downloaded unless a path is given)."""
    import tempfile
    act, url, note = PDF_ACTS[key]
    if not pdf_path:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(urllib.request.urlopen(req, timeout=300).read())
        tmp.close()
        pdf_path = tmp.name
    today = datetime.date.today().isoformat()
    out = {}
    for raw, title_words, body_lines in split_crpc(crpc_lines(pdf_path)):
        title = re.sub(r"\[?\*(?:\s*\*)*\]?|[\[\]]", "", " ".join(title_words)).rstrip("_")
        title = re.sub(r"\s+", " ", title).strip(" .:-\u2013\u2014")
        body = clean_crpc(body_lines)
        # A title that runs past a full stop ("Short title. commencement.", "rape, etc. by medical
        # practitioner.") leaves its tail at the start of the body, in lowercase.
        tail = re.match(r"([a-z][^.()]{0,80}?)\.(?:_*)\s*", body)
        if tail:
            title = TITLE_FIXES.get(f"{key}:{raw}", title + ". " + tail.group(1))
            body = body[tail.end():]
        body = re.sub(r"^[_\u2013\u2014-]+\s*", "", body)
        # A run of underscores closes a chapter; what follows is chapter and part headings.
        body = re.split(r"\s*_{3,}", body)[0].strip()
        # A lettered cross-heading for the next group of sections ("B.—Search-warrants") ends up
        # at the end of the section before it. Illustrations ("B. B may sue A") are kept: real
        # headings never contain a lone capital letter.
        body = CROSS_HEADING_AT_END.sub("", body).strip()
        # A repealed section set in plain type runs into the one before it ("2. Repeal. Rep. by ...").
        body = re.split(r"\s+\d{1,3}[A-Z]?\.\s+[A-Z][^.]{2,60}\.\s+(?:Rep\.|Omitted)", body)[0].strip()
        if not body or re.search(r"\b(Repealed|Omitted)\b", title) or len(title) > 200:
            continue
        if "Illustration" in body:
            body = body[: body.index("Illustration")].rstrip()
        out[f"{key}:{raw}"] = {"act": act, "kind": "section", "number": _crpc_number(raw), "title": title,
                               "text": body, "source": url, "note": note, "retrieved": today}
    return out


def build_crpc(pdf_path: str = None) -> dict:
    return build_pdf_act("crpc", pdf_path)


def build_all():
    """The whole PPC and the whole Constitution (articles only, not schedules)."""
    today = datetime.date.today().isoformat()
    out = {}
    act, url = SOURCES["ppc"]
    for number, title, body in split_all(page_text(url)):
        if not real_provision(number, title, 511):
            continue
        if "Illustration" in body:
            body = body[: body.index("Illustration")].rstrip()
        out.setdefault(f"ppc:{number.upper()}", {"act": act, "kind": "section", "number": number, "title": title,
                                                 "text": body, "source": url, "note": PPC_NOTE, "retrieved": today})
    for page in CONSTITUTION_PAGES:
        url = CONSTITUTION_BASE + page
        try:
            text = page_text(url)
        except Exception as e:
            print(f"skipped {page}: {e}", file=sys.stderr)
            continue
        for number, title, body in split_all(text):
            if not real_provision(number, title, 280):
                continue
            out.setdefault(f"const:{number.upper()}", {"act": "Constitution of Pakistan, 1973", "kind": "article",
                                                       "number": number, "title": title, "text": body, "source": url,
                                                       "note": CONSTITUTION_NOTE, "retrieved": today})
    return out


def summary(out: dict) -> str:
    counts = [f"{sum(k.startswith('ppc:') for k in out)} PPC sections",
              f"{sum(k.startswith('const:') for k in out)} Constitution articles"]
    counts += [f"{sum(k.startswith(p + ':') for k in out)} {p.upper()} sections" for p in PDF_ACTS]
    return ", ".join(counts)


def merge_into_statutes(new: dict, prefixes) -> dict:
    """statutes.json with the given Acts' sections replaced by new, everything else kept."""
    try:
        out = json.loads(Path("statutes.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        out = {}
    out = {k: v for k, v in out.items() if k.split(":")[0] not in prefixes}
    out.update(new)
    Path("statutes.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main():
    if "--act" in sys.argv:
        # --act KEY [file.pdf]: add or refresh one Act from PDF_ACTS, keeping everything else.
        rest = sys.argv[sys.argv.index("--act") + 1:]
        if not rest or rest[0] not in PDF_ACTS:
            raise SystemExit(f"--act needs one of: {', '.join(PDF_ACTS)}")
        out = merge_into_statutes(build_pdf_act(rest[0], rest[1] if len(rest) > 1 else None), {rest[0]})
        print(f"Wrote statutes.json: {summary(out)}.")
        return
    if "--acts" in sys.argv:
        new = {}
        for key in PDF_ACTS:
            new.update(build_pdf_act(key))
        out = merge_into_statutes(new, set(PDF_ACTS))
        print(f"Wrote statutes.json: {summary(out)}.")
        return
    if "--all" in sys.argv:
        out = build_all()
        for key in PDF_ACTS:
            out.update(build_pdf_act(key))
        checked = build_listed()
        for key, want in checked.items():   # the hand-checked provisions must come out the same
            got = out.get(key)
            if not got or got["text"] != want["text"]:
                print(f"MISMATCH with hand-checked {key}; keeping the checked version", file=sys.stderr)
            out[key] = {**want, "note": out.get(key, want).get("note", want.get("note", ""))}
        Path("statutes.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote statutes.json: {summary(out)}.")
        return
    out = build_listed()
    Path("statutes.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWrote statutes.json with {len(out)} provisions.")


def build_listed():
    pages = {}
    out, today = {}, datetime.date.today().isoformat()
    for src, kind, number, title, nxt, cut in PROVISIONS:
        act, url = SOURCES[src]
        if src not in pages:
            pages[src] = page_text(url)
        found = extract(pages[src], number, title, nxt, cut)
        if not found:
            print(f"NOT FOUND: {act} {kind} {number}", file=sys.stderr)
            continue
        full_title, body = found
        key = ("ppc:" if src == "ppc" else "const:") + number.upper()
        out[key] = {"act": act, "kind": kind, "number": number, "title": full_title, "text": body,
                    "source": url, "note": PPC_NOTE if src == "ppc" else CONSTITUTION_NOTE, "retrieved": today}
        print(f"{key:12} {full_title[:50]:50} {len(body):5} chars")
    return out


if __name__ == "__main__":
    main()
