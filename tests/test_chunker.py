"""Tests for the Pakistani court judgment chunker and metadata extraction.

Run:  .venv/bin/python -m pytest tests/ -v
 or:  .venv/bin/python tests/test_chunker.py

Needs no API key, no Chroma and no embedding model.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wakeel_ai as W  # noqa: E402

HEADER = """Stereo. H C J D A-38.
JUDGMENT SHEET
IN THE LAHORE HIGH COURT, LAHORE
JUDICIAL DEPARTMENT
Writ Petition No.6404 of 2020
Aashiq Hussain
Versus
Fida Hussain & others
J U D G M E N T
Date of hearing: 24.01.2023.
MUHAMMAD SAJID MEHMOOD SETHI, J.: Through

"""

# Deliberately under min_chunk_size (400 chars) each: this is what the original
# chunker silently discarded.
SHORT_PARAS = [
    "1. Through this criminal appeal the appellant assails his conviction under Section 302 PPC.",
    "2. Learned counsel contended the prosecution failed, relying on 2016 SCMR 2073.",
    "3. Arguments heard and available record perused.",
    "4. The law favours adjudication on merits rather than technicalities.",
    "5. For the reasons recorded above this criminal appeal is dismissed.",
]

_FILLER = ("The record has been examined in considerable detail by this Court and the "
           "submissions advanced at the bar have been given due consideration. " * 4)

LONG_PARAS = [p + " " + _FILLER for p in SHORT_PARAS]

SHORT_JUDGMENT = HEADER + "\n\n".join(SHORT_PARAS) + "\n"
LONG_JUDGMENT = HEADER + "\n\n".join(LONG_PARAS) + "\n"

META = {"doc_id": "wp_6404_2020", "case_no": "", "title": ""}


def chunks_for(text, merge):
    return W.chunk_text(text, dict(META), merge_short_paragraphs=merge)


def body_of(chunks):
    return re.sub(r"\s+", " ", " ".join(c["raw_text"] for c in chunks)).lower()


# --------------------------------------------------------------------------
# The bug: short paragraphs must not vanish
# --------------------------------------------------------------------------

def test_legacy_drops_short_paragraphs():
    """Documents the original behaviour so the regression stays visible."""
    blob = body_of(chunks_for(SHORT_JUDGMENT, merge=False))
    assert "law favours adjudication on merits" not in blob
    assert "this criminal appeal is dismissed" not in blob


def test_merging_keeps_every_short_paragraph():
    blob = body_of(chunks_for(SHORT_JUDGMENT, merge=True))
    for para in SHORT_PARAS:
        snippet = re.sub(r"\s+", " ", para[3:60]).lower().strip()
        assert snippet in blob, f"lost: {snippet!r}"


def test_merging_keeps_the_final_paragraph():
    """The conclusion is last in the file and was dropped by the final flush."""
    blob = body_of(chunks_for(SHORT_JUDGMENT, merge=True))
    assert "this criminal appeal is dismissed" in blob


def test_no_text_is_lost_overall():
    chunks = chunks_for(SHORT_JUDGMENT, merge=True)
    kept = sum(len(c["raw_text"]) for c in chunks)
    assert kept > 0.9 * len(SHORT_JUDGMENT.strip())


def test_merged_chunk_reports_a_paragraph_range():
    """A chunk covering paragraphs 1-3 should say so, not claim to be just 1."""
    labels = [c["paragraph_number"] for c in chunks_for(SHORT_JUDGMENT, merge=True)]
    assert any("-" in str(lbl) for lbl in labels), labels


# --------------------------------------------------------------------------
# Existing chunking behaviour that must not regress
# --------------------------------------------------------------------------

def test_long_paragraphs_chunk_one_per_paragraph():
    """One chunk per numbered paragraph.

    No "0" header chunk here: "JUDGMENT SHEET" on the second line ends the
    header before 100 characters have accumulated, so there is nothing to emit.
    Real judgments with longer preambles do produce one.
    """
    paras = [c["paragraph_number"] for c in chunks_for(LONG_JUDGMENT, merge=True)]
    assert paras == ["1", "2", "3", "4", "5"], paras


def test_fix_preserves_structure_and_loses_nothing():
    """Above min_chunk_size the fix must not reshape the output.

    The one permitted difference is that the fixed chunker keeps text the legacy
    one threw away (the bare "JUDGMENT" marker line), so each legacy chunk must be
    contained in its counterpart rather than byte-equal to it.
    """
    legacy = chunks_for(LONG_JUDGMENT, False)
    fixed = chunks_for(LONG_JUDGMENT, True)

    assert len(legacy) == len(fixed)
    assert [c["paragraph_number"] for c in legacy] == [c["paragraph_number"] for c in fixed]
    assert [c["section_type"] for c in legacy] == [c["section_type"] for c in fixed]

    # Every legacy chunk survives intact inside its counterpart: the fix only
    # ever adds back text that was being discarded, never rewrites or drops.
    for a, b in zip(legacy, fixed):
        assert a["raw_text"] in b["raw_text"], (a["raw_text"][:60], b["raw_text"][:60])

    kept_legacy = sum(len(c["raw_text"]) for c in legacy)
    kept_fixed = sum(len(c["raw_text"]) for c in fixed)
    assert kept_fixed >= kept_legacy


def test_citations_and_provisions_extracted():
    chunks = chunks_for(LONG_JUDGMENT, merge=True)
    citations = [c for ch in chunks for c in ch["citations"]]
    provisions = [p for ch in chunks for p in ch["provisions"]]
    assert any("SCMR" in c for c in citations), citations
    assert any("302" in p for p in provisions), provisions


def test_section_types_assigned():
    types = {c["section_type"] for c in chunks_for(LONG_JUDGMENT, merge=True)}
    assert types
    assert types - {"header", "facts", "arguments", "analysis", "conclusion"} == set()


def test_contextual_prefix_added():
    for c in chunks_for(LONG_JUDGMENT, merge=True):
        assert c["text"] != c["raw_text"]
        assert c["raw_text"] in c["text"]


def test_fallback_chunker_for_non_judgments():
    chunks = W.chunk_text("Some generic text. " * 400, dict(META, doc_id="plain"))
    assert len(chunks) >= 1
    assert all(c["text"] for c in chunks)


def test_empty_and_tiny_input_do_not_crash():
    assert W.chunk_text("", dict(META)) == []
    assert isinstance(W.chunk_text("short", dict(META)), list)


# --------------------------------------------------------------------------
# Metadata extraction, against real header layouts
# --------------------------------------------------------------------------

def meta_for(text):
    return W.PakistaniCourtJudgmentChunker().extract_case_metadata(text, "doc")


def test_metadata_from_a_real_header():
    m = meta_for(LONG_JUDGMENT)
    assert m["court"] == "LAHORE HIGH COURT"
    assert m["case_number"] == "6404/2020"
    assert m["parties"]["appellant"] == "Aashiq Hussain"
    assert m["parties"]["respondent"] == "Fida Hussain"   # "& others" stripped
    assert m["judges"] == ["MUHAMMAD SAJID MEHMOOD SETHI"]
    assert m["date_of_hearing"] == "24.01.2023"


def test_court_does_not_swallow_following_lines():
    """\\s matches newlines, so the old pattern ran into JUDICIAL DEPARTMENT."""
    m = meta_for("IN THE LAHORE HIGH COURT LAHORE\nJUDICIAL DEPARTMENT\n"
                 "Writ Petition No.6917/2023\n")
    assert m["court"] == "LAHORE HIGH COURT"
    assert "\n" not in m["court"]
    assert "JUDICIAL" not in m["court"]


def test_court_bench_is_kept():
    m = meta_for("IN THE LAHORE HIGH COURT, MULTAN BENCH\nJUDICIAL DEPARTMENT\n")
    assert m["court"] == "LAHORE HIGH COURT, MULTAN BENCH"


def test_court_survives_spaced_out_pdf_extraction():
    """Some PDFs extract as "L A HOR E HIGH C OUR T"."""
    m = meta_for("JUDGMENT SHEET\nL A HOR E HIGH C OUR T, LAHOR E\nJUDICIAL DEPARTMENT\n")
    assert m["court"] == "LAHORE HIGH COURT"


def test_case_number_formats():
    cases = {
        "Writ Petition No.6404 of 2020": "6404/2020",
        "Writ Petition No.6917/2023": "6917/2023",
        "Crl. Misc. No.61551-M of 2022": "61551/2022",
        "R.S.A.No.46 of 2009": "46/2009",
        "C. R. No. 174-D of 2009": "174/2009",
        "W.P.No.50725-2022": "50725/2022",
        "EFA No. 18 of 2022": "18/2022",
        "Writ Petition No .36910 of 2022": "36910/2022",
        "Customs Reference No.52 of 2016": "52/2016",
    }
    for line, expected in cases.items():
        assert meta_for(line + "\n")["case_number"] == expected, line


def test_form_number_is_not_a_case_number():
    """Form No.HCJD/C-121 is the court's stationery code."""
    m = meta_for("Form No.HCJD/C-121\nORDER SHEET\nLAHORE HIGH COURT, LAHORE\n")
    assert m["case_number"] == ""


def test_redundant_case_prefix_is_stripped():
    m = meta_for("Case No. Crl.Misc.No. 261-B of 2023\n")
    assert m["case_number"] == "261/2023"
    assert m["judgment_type"] == "Crl.Misc"


def test_parties_inline_and_multiline():
    inline = meta_for("Hafiz Riaz Ahmad etc. Vs. Province of Punjab etc.\n")
    assert inline["parties"]["appellant"] == "Hafiz Riaz Ahmad"
    assert inline["parties"]["respondent"] == "Province of Punjab"

    lowercase_v = meta_for("Rana Muhammad Yousaf Khan v. The State, etc.\n")
    assert lowercase_v["parties"]["appellant"].startswith("Rana Muhammad")

    multiline = meta_for("Commissioner Inland Revenue\nVersus\nM/s Prime Commercial Bank Ltd.\n")
    assert multiline["parties"]["appellant"] == "Commissioner Inland Revenue"
    assert multiline["parties"]["respondent"] == "M/s Prime Commercial Bank Ltd"


def test_judge_name_formats():
    for line, expected in {
        "MUHAMMAD SAJID MEHMOOD SETHI, J.: Through": "MUHAMMAD SAJID MEHMOOD SETHI",
        "ABID AZIZ SHEIKH, J. In this Constitutional Petition,": "ABID AZIZ SHEIKH",
        "Shahid Karim, J:-. This": "Shahid Karim",
        "SHAHID BILAL HASSAN-J : Succinctly, a decree": "SHAHID BILAL HASSAN",
    }.items():
        assert meta_for(line + "\n")["judges"] == [expected], line


def test_hearing_date_formats():
    for line, expected in {
        "Date of hearing: 24.01.2023.": "24.01.2023",
        "Date of hearing 06-02-2023": "06-02-2023",
        "Date of Hearing. 12-01-2023": "12-01-2023",
    }.items():
        assert meta_for(line + "\n")["date_of_hearing"] == expected, line


def test_metadata_never_raises_or_returns_none():
    for text in ("", "nonsense", "\n\n\n", "IN THE"):
        m = meta_for(text)
        assert m["court"] == "" or isinstance(m["court"], str)
        assert isinstance(m["judges"], list)
        assert isinstance(m["parties"], dict)
        assert m["parties"]["appellant"] is not None


# --------------------------------------------------------------------------
# BM25 tokenization
# --------------------------------------------------------------------------

def tok(text):
    return W.BM25Index(None).tokenize(text)


def test_tokenizer_splits_punctuation_off_numbers():
    """A judgment writes "497(5)"; a user types "497". They must match."""
    assert "497" in tok("under section 497(5) Cr.P.C. the court may")
    assert "5" in tok("497(5)")
    assert "302" in tok("convicted under Section 302/34 PPC.")


def test_tokenizer_collapses_dotted_abbreviations():
    """Judgments write "Cr.P.C."; users type "CrPC"."""
    assert "crpc" in tok("an application under Cr.P.C. was filed")
    assert "cpc" in tok("Order VII Rule 11 C.P.C.")
    assert "ptr" in tok("P.T.R. No.173 of 2013")


def test_tokenizer_matches_query_spelling_to_judgment_spelling():
    judgment = tok("bail under Section 497(5) Cr.P.C. was cancelled")
    for term in tok("497 CrPC bail"):
        assert term in judgment, term


def test_tokenizer_drops_empty_tokens_and_handles_edges():
    assert tok("") == []
    assert tok("   ...   ") == []
    assert "" not in tok("a, b. c")


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
