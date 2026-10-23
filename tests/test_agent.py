"""Tests for the LangGraph agent: routing, memory, sources, error handling.

Run:  .venv/bin/python -m pytest tests/ -v
 or:  .venv/bin/python tests/test_agent.py

Uses a fake LLM and fake retrieval, so no API key, Chroma or embedding model.
"""
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wakeel_ai as W  # noqa: E402

PLAN = {}  # query -> QueryClassification kwargs, set per test


class _Structured:
    def invoke(self, prompt):
        for q, kwargs in PLAN.items():
            if f"Input Query:\n{q}" in prompt:
                return W.QueryClassification(**kwargs)
        raise AssertionError(f"no classification planned for: {prompt[-120:]!r}")


class FakeLLM:
    """Records prompts; returns a tag naming which prompt template was used."""

    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema):
        return _Structured()

    def invoke(self, prompt):
        self.calls.append(prompt)
        if "SIMILARITIES and DIFFERENCES" in prompt:
            tag = "COMPARE"
        elif "Document to summarize:" in prompt:
            tag = "SUMMARY"
        else:
            tag = "ANSWER"
        return types.SimpleNamespace(content=f"<{tag}>")


def fake_search_hybrid(query, top_k=10, alpha=0.5):
    return [{
        "score": 0.9,
        "doc": "[Case: 12/2019] [Paragraph 7]\n\n7. Judgment text. Section 302 PPC applied.",
        "meta": {"doc_id": "crl_12", "case_no": "12/2019", "judgment_type": "Criminal Appeal",
                 "court": "LAHORE HIGH COURT", "title": "A v B", "judges": '["Ali"]',
                 "parties": '{"appellant":"A","respondent":"B"}',
                 "citations": '["2016 SCMR 2073"]', "provisions": '["Section 302 PPC"]',
                 "section_type": "analysis", "paragraph_number": "7"},
    }]


def make_agent(llm=None, thread="t"):
    W.search_hybrid = fake_search_hybrid
    return W.WakeelAgent(llm or FakeLLM(), thread_id=thread)


# --------------------------------------------------------------------------
# A second opinion when the main model refuses to answer from retrieved passages
# --------------------------------------------------------------------------

class NotFoundLLM(FakeLLM):
    """Refuses every answer prompt, as an over-cautious model does."""
    def invoke(self, prompt):
        self.calls.append(prompt)
        return types.SimpleNamespace(content="Not found in the indexed judgments. Narrow the search.")


class AnswersLLM(FakeLLM):
    def invoke(self, prompt):
        self.calls.append(prompt)
        return types.SimpleNamespace(content="The court held that bail may be cancelled [S1].")


def _second_opinion_run(main, second, thread):
    PLAN.clear()
    PLAN["Find bail cases"] = {"route": "GQ", "restated_query": "bail cases"}
    W.search_hybrid = fake_search_hybrid
    agent = W.WakeelAgent(main, thread_id=thread, second_opinion=second)
    return agent.run_detailed("Find bail cases")


def test_second_opinion_answers_when_the_main_model_refuses():
    main, second = NotFoundLLM(), AnswersLLM()
    out = _second_opinion_run(main, second, "second-1")
    assert out["answer"].startswith("The court held") and out["grounded"] and out["sources"]
    assert len(second.calls) == 1 and "read every excerpt again" in second.calls[0]
    assert len([c for c in main.calls if "Retrieved excerpts" in str(c)]) == 2     # first try and the recheck


def test_second_opinion_keeps_the_refusal_if_it_also_refuses_or_fails():
    class Fails(FakeLLM):
        def invoke(self, prompt):
            raise RuntimeError("503 UNAVAILABLE")
    for second in (NotFoundLLM(), Fails(), None):
        out = _second_opinion_run(NotFoundLLM(), second, f"second-{type(second).__name__}")
        assert "Not found in the indexed judgments" in out["answer"] and out["grounded"] is False


def test_second_opinion_is_not_asked_when_the_main_model_answers():
    second = AnswersLLM()
    out = _second_opinion_run(AnswersLLM(), second, "second-unused")
    assert out["grounded"] and second.calls == []


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def test_general_query_routes_to_answer():
    PLAN.clear()
    PLAN["Find bail cases under Section 497 CrPC"] = {
        "route": "GQ", "restated_query": "bail cases under Section 497 CrPC"}
    agent = make_agent(thread="gq")
    out = agent.run_detailed("Find bail cases under Section 497 CrPC")
    assert out["answer"] == "<ANSWER>"
    assert out["route"] == "GQ"
    assert "Section 302 PPC applied" in agent.llm.calls[-1]


def test_summarize_query_routes_to_summarize():
    PLAN.clear()
    PLAN["Summarize the Lahore High Court judgment"] = {
        "route": "SQ", "restated_query": "Lahore High Court judgment"}
    out = make_agent(thread="sq").run_detailed("Summarize the Lahore High Court judgment")
    assert out["answer"] == "<SUMMARY>"
    assert out["route"] == "SQ"


# --------------------------------------------------------------------------
# Answer mode and language
# --------------------------------------------------------------------------

def test_default_answer_is_citizen_mode_in_english():
    PLAN.clear()
    PLAN["Find bail cases"] = {"route": "GQ", "restated_query": "bail cases"}
    agent = make_agent(thread="style-default")
    agent.run_detailed("Find bail cases")
    assert "member of the public" in agent.llm.calls[-1]
    assert "Write the answer in English." in agent.llm.calls[-1]


def test_lawyer_mode_and_urdu_reach_the_answer_prompt():
    PLAN.clear()
    PLAN["Find bail cases"] = {"route": "GQ", "restated_query": "bail cases"}
    agent = make_agent(thread="style-lawyer-ur")
    agent.run_detailed("Find bail cases", mode="lawyer", language="ur")
    prompt = agent.llm.calls[-1]
    assert "practising lawyer" in prompt and "member of the public" not in prompt
    assert "Urdu script" in prompt
    # The UI and evaluation detect citations and refusals by their English form.
    assert "[S1]" in prompt and W.NOT_FOUND_PHRASE in prompt


def test_mode_and_language_can_change_mid_conversation():
    PLAN.clear()
    PLAN["Find bail cases"] = {"route": "GQ", "restated_query": "bail cases"}
    PLAN["And the verdict?"] = {"route": "GQ", "restated_query": "verdict in the bail case"}
    agent = make_agent(thread="style-switch")
    agent.run_detailed("Find bail cases", mode="lawyer", language="en")
    agent.run_detailed("And the verdict?", mode="citizen", language="roman_ur")
    assert "member of the public" in agent.llm.calls[-1]
    assert "Roman Urdu" in agent.llm.calls[-1]


def test_summary_follows_the_mode():
    PLAN.clear()
    PLAN["Summarize the judgment"] = {"route": "SQ", "restated_query": "the judgment"}
    agent = make_agent(thread="style-sq")
    out = agent.run_detailed("Summarize the judgment", mode="lawyer")
    assert out["answer"] == "<SUMMARY>"
    assert "practising lawyer" in agent.llm.calls[-1]


def test_unknown_mode_or_language_falls_back_to_defaults():
    assert W.style_instructions("judge", "fr") == W.style_instructions()


def test_classifier_restates_urdu_queries_in_english():
    template = W.CLASSIFICATION_PROMPT.template
    assert "Always write it, and" in template and "in English" in template


def test_run_stream_reports_each_step_then_the_same_result():
    PLAN.clear()
    PLAN["Find bail cases"] = {"route": "GQ", "restated_query": "bail cases"}
    events = list(make_agent(thread="stream").run_stream("Find bail cases", mode="lawyer"))
    steps = [e for e in events if e["type"] == "step"]
    assert [s["node"] for s in steps] == ["classify", "retrieve", "answer"]
    assert steps[0]["route"] == "GQ" and steps[0]["restated_query"] == "bail cases"
    assert steps[1]["passages"] >= 1 and steps[1]["judgments"] == 1
    result = events[-1]
    assert result["type"] == "result" and result["answer"] == "<ANSWER>" and result["route"] == "GQ"
    assert events.count(result) == 1 and events.index(result) == len(events) - 1


def test_model_failure_while_writing_is_an_error_not_a_grounded_answer():
    class FailsOnAnswer(FakeLLM):
        def invoke(self, prompt):
            self.calls.append(prompt)
            raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
    PLAN.clear()
    PLAN["Find bail cases"] = {"route": "GQ", "restated_query": "bail cases"}
    out = make_agent(FailsOnAnswer(), thread="fail-write").run_detailed("Find bail cases")
    assert out["route"] == "ERROR" and out["grounded"] is False and out["sources"] == []
    events = list(make_agent(FailsOnAnswer(), thread="fail-write-2").run_stream("Find bail cases"))
    assert [e for e in events if e.get("node") == "answer"][0]["route"] == "ERROR"
    assert events[-1]["route"] == "ERROR"


def test_search_keys_chunks_by_id_not_by_text_prefix():
    """Chunks of one judgment share a long text prefix; they must stay separate hits."""
    import inspect
    module = inspect.getsource(W)      # other tests replace W.search_hybrid with a fake
    src = module[module.index("def search_hybrid"):module.index("def message_text")]
    assert "[:50]" not in src and 'semantic_results["ids"]' in src


class LegalCheckLLM(FakeLLM):
    """Answers the is-this-legal check with a fixed verdict, everything else as FakeLLM."""
    def __init__(self, verdict="yes"):
        super().__init__()
        self.verdict = verdict

    def invoke(self, prompt):
        if isinstance(prompt, str) and "Respond with **only** one word: yes or no" in prompt:
            self.calls.append(prompt)
            return types.SimpleNamespace(content=self.verdict)
        if isinstance(prompt, list):   # transcription request with page images
            self.calls.append(prompt)
            return types.SimpleNamespace(content="FIR No. 333/2021 under section 489-F PPC, Police Station City Bhakkar.")
        return super().invoke(prompt)


def test_explain_document_uses_the_text_and_similar_judgments():
    llm = LegalCheckLLM("yes")
    agent = make_agent(llm, thread="explain")
    out = agent.explain_document("FIR No. 333/2021 lodged under section 489-F PPC for a dishonoured cheque.",
                                 mode="citizen", language="roman_ur")
    prompt = llm.calls[-1]
    assert out["route"] == "DOC" and out["grounded"] and len(out["sources"]) == 1
    assert "489-F" in prompt and "Similar judgments" in prompt and "Roman Urdu" in prompt
    assert any("489" in p for p in out["provisions"])


def test_explain_document_rejects_non_legal_text():
    agent = make_agent(LegalCheckLLM("no"), thread="explain-no")
    out = agent.explain_document("Recipe: two cups of flour and a pinch of salt.")
    assert out["route"] == "REJECT" and out["answer"] == W.NOT_LEGAL_DOCUMENT and out["sources"] == []


def test_explain_document_reads_photos_with_the_model():
    llm = LegalCheckLLM("yes")
    agent = make_agent(llm, thread="explain-img")
    out = agent.explain_document("", images=[("image/png", b"\x89PNG fake")])
    msgs = [c for c in llm.calls if isinstance(c, list)]
    assert msgs and msgs[0][0].content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert out["route"] == "DOC" and out["document_chars"] > 0


def test_workflows_research_what_they_need():
    agent = make_agent(thread="workflows")
    out = agent.run_workflow("hearing", {"What the hearing is for": "post-arrest bail", "The facts": "Cheque bounced"})
    prompt = agent.llm.calls[-1]
    assert out["route"] == "HEARING" and out["sources"]                      # searched similar judgments
    assert "## Questions the bench may ask" in prompt and "- The facts: Cheque bounced" in prompt
    letter = agent.run_workflow("client_letter", {"Client name": "Aslam", "What happened at the hearing": "Adjourned"})
    assert letter["route"] == "CLIENT_LETTER" and letter["sources"] == []   # writes from the notes alone
    assert "Draft for review by the advocate" in agent.llm.calls[-1]
    own = agent.run_workflow("chronology", {"Documents": "FIR lodged on 3 March 2024. Notice served 10 March 2024."})
    assert own["sources"] == [] and "FIR lodged on 3 March 2024" in agent.llm.calls[-1].split("Text:")[1]
    assert agent.run_workflow("nope", {"x": "y"})["route"] == "ERROR"
    assert agent.run_workflow("hearing", {"x": "  "})["route"] == "ERROR"
    assert agent.strategy_memo({"What happened": "x"})["route"] == "STRATEGY"     # the memo is a workflow too


def test_chronology_of_a_library_judgment_cites_its_passages():
    saved, W.METADATA_DIR = W.METADATA_DIR, _fake_corpus()
    try:
        agent = make_agent(thread="timeline")
        out = agent.run_workflow("chronology", {}, doc_id="lhc_a")
        assert out["route"] == "CHRONOLOGY" and len(out["sources"]) == 10
        assert "Source 1 [S1]" in agent.llm.calls[-1]
        assert agent.run_workflow("chronology", {}, doc_id="missing")["route"] == "ERROR"
    finally:
        W.METADATA_DIR = saved


def test_draft_uses_only_the_given_facts_and_finds_supporting_judgments():
    agent = make_agent(thread="draft")
    out = agent.draft_document("cheque_notice", {"Your name": "Muhammad Aslam", "Amount": "Rs. 900,000", "Recipient": ""},
                               language="en")
    prompt = agent.llm.calls[-1]
    assert out["route"] == "DRAFT" and out["grounded"] and len(out["sources"]) == 1
    assert "- Your name: Muhammad Aslam" in prompt and "- Amount: Rs. 900,000" in prompt
    assert "Recipient" not in prompt.split("Facts provided:")[1].split("Judgment excerpts:")[0]   # blank fields are left out
    assert "placeholder in square" in prompt and "489-F" in prompt
    assert agent.draft_document("divorce_deed", {})["route"] == "ERROR"
    assert "style sample" not in prompt                       # no sample, no style rules


def test_draft_follows_the_users_style_sample_but_not_its_facts():
    agent = make_agent(thread="draft-style")
    sample = "Through Registered A.D.\n\nSir,\n\nUnder instructions from my client Mr. Zahid {old}, I serve you..."
    agent.draft_document("cheque_notice", {"Amount": "Rs. 900,000"}, sample=sample)
    prompt = agent.llm.calls[-1]
    assert "Through Registered A.D." in prompt and "Zahid {old}" in prompt     # braces in the sample are kept
    assert "Take nothing else from it" in prompt
    assert prompt.index("style sample") < prompt.index("Facts provided:")


def test_strategy_memo_researches_the_facts_and_never_predicts():
    agent = make_agent(thread="strategy")
    out = agent.strategy_memo({"Your side": "accused", "What happened": "Cheque bounced after a loan", "Other": ""},
                              mode="lawyer")
    prompt = agent.llm.calls[-1]
    assert out["route"] == "STRATEGY" and out["grounded"] and out["sources"]
    assert "- What happened: Cheque bounced after a loan" in prompt and "- Other:" not in prompt
    assert "Do not predict the outcome" in prompt and "## Arguments the other side will make" in prompt
    assert agent.strategy_memo({"What happened": "  "})["route"] == "ERROR"


def test_ask_document_answers_from_the_document_only():
    agent = make_agent(thread="docqa")
    out = agent.ask_document("Notice dated 1 May 2026 demanding payment within 15 days.", "What is the deadline?")
    prompt = agent.llm.calls[-1]
    assert out["route"] == "DOC" and "What is the deadline?" in prompt and "15 days" in prompt
    assert agent.ask_document("", "anything")["route"] == "ERROR"


def test_summarize_document_briefs_an_uploaded_judgment():
    llm = LegalCheckLLM("yes")
    agent = make_agent(llm, thread="sumdoc")
    out = agent.summarize_document("JUDGMENT. The petition is allowed. " * 20, mode="lawyer")
    assert out["route"] == "SQ" and out["answer"] == "<SUMMARY>" and "practising lawyer" in llm.calls[-1]
    out = agent.summarize_document("", images=[("image/png", b"\x89PNG fake")])
    assert out["route"] == "SQ" and any(isinstance(c, list) for c in llm.calls)


def _fake_corpus():
    import json as _json
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    for doc in ("lhc_a", "lhc_b"):
        chunks = [{"chunk_id": i, "text": f"[Case: {doc}] [Paragraph {i}] {doc} paragraph {i}. " + "x" * 50,
                   "doc_id": doc, "paragraph_number": str(i), "case_no": "1/2023", "title": doc}
                  for i in range(10)]
        (tmp / f"{doc}.json").write_text(_json.dumps(chunks), encoding="utf-8")
    return tmp


def test_summarize_judgment_reads_the_whole_judgment():
    saved, W.METADATA_DIR = W.METADATA_DIR, _fake_corpus()
    try:
        agent = make_agent(thread="sumj")
        out = agent.summarize_judgment("lhc_a", mode="lawyer")
        prompt = agent.llm.calls[-1]
        assert out["route"] == "SQ" and out["answer"] == "<SUMMARY>" and len(out["sources"]) == 10
        assert "[S10] (paragraph 9)" in prompt and "practising lawyer" in prompt
        assert "[Case:" not in prompt   # embedding-time prefixes are stripped
        assert agent.summarize_judgment("missing")["route"] == "ERROR"
        assert agent.summarize_judgment("../etc/passwd")["route"] == "ERROR"
    finally:
        W.METADATA_DIR = saved


class RowLLM(FakeLLM):
    """Answers review-table prompts with a JSON row citing the first source in the prompt."""
    def __init__(self, fail_on=None):
        super().__init__()
        self.fail_on = fail_on

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("503 UNAVAILABLE")
        if "filling one row of a review table" in prompt:
            import re as _re
            first = int(_re.search(r"Source (\d+) \[S", prompt).group(1))
            return types.SimpleNamespace(content='Here you go: {"Outcome": {"answer": "Bail cancelled", "source": %d}, '
                                                 '"Legal question": {"answer": "Locus standi", "source": 999}}' % first)
        return types.SimpleNamespace(content="Both cases <ANSWER> [S1]")


def test_review_table_cites_each_cell_with_sources_numbered_across_judgments():
    saved, W.METADATA_DIR = W.METADATA_DIR, _fake_corpus()
    try:
        agent = make_agent(RowLLM(), thread="table")
        out = agent.review_table(["lhc_a", "lhc_b", "lhc_a"], ["Outcome", "Legal question", "Relief"], mode="lawyer")
        assert out["route"] == "TABLE" and out["columns"] == ["Outcome", "Legal question", "Relief"]
        assert [r["doc_id"] for r in out["rows"]] == ["lhc_a", "lhc_b"]        # duplicates dropped
        a, b = out["rows"]
        assert a["cells"][0] == {"text": "Bail cancelled", "source": 1}
        assert b["cells"][0]["source"] == 11                                     # numbering continues across judgments
        assert a["cells"][1]["source"] is None                                   # out-of-range citation dropped
        assert a["cells"][2] == {"text": "Not stated", "source": None}           # column the model skipped
        assert len(out["sources"]) == 20 and "| Bail cancelled [S1] |" in out["answer"]
        assert agent.review_table([], ["Outcome"])["route"] == "ERROR"
        assert agent.review_table(["lhc_a", "missing"], ["Outcome"])["route"] == "ERROR"
    finally:
        W.METADATA_DIR = saved


def test_ask_judgment_answers_from_one_judgment_and_cites_its_paragraphs():
    saved, W.METADATA_DIR = W.METADATA_DIR, _fake_corpus()
    try:
        llm = AnswersLLM()
        agent = W.WakeelAgent(llm, thread_id="ask-judgment")
        out = agent.ask_judgment("lhc_a", "Why was bail cancelled?", mode="lawyer")
        prompt = llm.calls[-1]
        assert out["route"] == "JUDGMENT" and out["grounded"] and len(out["sources"]) == 10
        assert "Why was bail cancelled?" in prompt and "Answer only from this one judgment" in prompt
        assert "lhc_a paragraph 3" in prompt and "lhc_b" not in prompt          # that judgment's text, not another's
        assert agent.ask_judgment("lhc_a", "  ")["route"] == "ERROR"
        assert agent.ask_judgment("missing", "Why?")["route"] == "ERROR"
        refused = W.WakeelAgent(NotFoundLLM(), thread_id="ask-judgment-2").ask_judgment("lhc_a", "Unrelated?")
        assert refused["grounded"] is False and "Not found" in refused["answer"]
    finally:
        W.METADATA_DIR = saved


def test_review_table_takes_the_users_own_documents():
    saved, W.METADATA_DIR = W.METADATA_DIR, _fake_corpus()
    try:
        agent = make_agent(RowLLM(), thread="table-docs")
        notice = "LEGAL NOTICE\n\nYou issued cheque No. 123 for Rs. 900,000.\n\nPay within 15 days."
        out = agent.review_table(["lhc_a"], ["Outcome"], documents=[{"name": "notice.pdf", "text": notice},
                                                                    {"name": "empty.pdf", "text": "  "}])
        assert out["route"] == "TABLE" and [r["doc_id"] for r in out["rows"]] == ["lhc_a", "upload-1"]
        doc = out["rows"][1]
        assert doc["name"] == "notice.pdf" and doc["cells"][0]["source"] == 11   # numbered after the judgment
        src = out["sources"][10]
        assert src["court"] == "Your document" and src["title"] == "notice.pdf" and "Rs. 900,000" in src["excerpt"]
        assert "| notice.pdf | Bail cancelled [S11] |" in out["answer"]
        only_docs = agent.review_table([], ["Outcome"], documents=[{"name": "n", "text": notice}])
        assert only_docs["route"] == "TABLE" and only_docs["rows"][0]["cells"][0]["source"] == 1
    finally:
        W.METADATA_DIR = saved


def test_document_hits_split_long_text_into_passages():
    text = ". ".join(f"Sentence {i} about the agreement" for i in range(400))   # one paragraph, ~13,000 chars
    hits = W.document_hits("contract.pdf", text, 2)
    assert len(hits) > 5 and all(len(h["doc"]) <= 2 * W.UPLOAD_PASSAGE_CHARS for h in hits)
    assert hits[0]["meta"]["doc_id"] == "upload-2" and hits[0]["meta"]["paragraph_number"] == "1"
    assert W.document_hits("x", "a\n\n" * 3, 1)[0]["doc"].startswith("a")


def test_review_table_survives_one_failed_row():
    saved, W.METADATA_DIR = W.METADATA_DIR, _fake_corpus()
    try:
        out = make_agent(RowLLM(fail_on="lhc_b paragraph"), thread="table-fail").review_table(["lhc_a", "lhc_b"], ["Outcome"])
        assert out["route"] == "TABLE"
        assert out["rows"][1]["cells"][0]["text"] == "Could not be read"
        both = make_agent(RowLLM(fail_on="paragraph"), thread="table-fail2").review_table(["lhc_a", "lhc_b"], ["Outcome"])
        assert both["route"] == "ERROR"
    finally:
        W.METADATA_DIR = saved


def test_ask_table_answers_from_the_table():
    agent = make_agent(RowLLM(), thread="table-ask")
    out = agent.ask_table("Where was bail cancelled?", "| Judgment | Outcome |\n|---|---|\n| A | Bail cancelled [S1] |")
    assert out["route"] == "TABLE" and "[S1]" in out["answer"] and "Bail cancelled [S1]" in agent.llm.calls[-1]
    assert agent.ask_table("", "x")["route"] == "ERROR"


def test_long_judgments_keep_their_opening_and_end():
    saved, W.METADATA_DIR = W.METADATA_DIR, _fake_corpus()
    try:
        paras = [h["meta"]["paragraph_number"] for h in W.judgment_hits("lhc_a", max_chars=400)]
        assert "0" in paras and "9" in paras and len(paras) < 10
        assert paras == sorted(paras, key=int)
    finally:
        W.METADATA_DIR = saved


def test_compare_judgments_numbers_sources_across_both():
    saved, W.METADATA_DIR = W.METADATA_DIR, _fake_corpus()
    try:
        agent = make_agent(thread="cmpj")
        out = agent.compare_judgments("lhc_a", "lhc_b", language="ur")
        prompt = agent.llm.calls[-1]
        assert out["route"] == "CQ" and len(out["sources"]) == 20
        assert "[S11] (paragraph 0)" in prompt and "Urdu script" in prompt
        assert agent.compare_judgments("lhc_a", "nope")["route"] == "ERROR"
    finally:
        W.METADATA_DIR = saved


def test_compare_prompt_asks_for_a_table_in_the_chosen_language():
    PLAN.clear()
    PLAN["Compare case A with case B"] = {
        "route": "CQ", "restated_query": "Compare A and B", "compare_a": "A", "compare_b": "B"}
    agent = make_agent(thread="cq-style")
    agent.run_detailed("Compare case A with case B", mode="citizen", language="roman_ur")
    prompt = agent.llm.calls[-1]
    assert "## Side by side" in prompt and "Roman Urdu" in prompt


def test_compare_query_retrieves_both_sides():
    PLAN.clear()
    PLAN["Compare case A with case B"] = {
        "route": "CQ", "restated_query": "Compare case A with case B",
        "compare_a": "case A", "compare_b": "case B"}
    agent = make_agent(thread="cq")
    out = agent.run_detailed("Compare case A with case B")
    assert out["answer"] == "<COMPARE>"
    assert "doc1" in agent.llm.calls[-1] or "DOCS are" in agent.llm.calls[-1]


def test_non_legal_query_is_rejected_without_an_llm_call():
    PLAN.clear()
    PLAN["What's the weather in New York?"] = {
        "route": "REJECT",
        "clarification": "Sorry, I can't answer this query as it is not legal related."}
    agent = make_agent(thread="rj")
    out = agent.run_detailed("What's the weather in New York?")
    assert out["answer"].startswith("Sorry, I can't answer")
    assert agent.llm.calls == []  # no retrieval, no generation


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------

def test_history_accumulates_within_a_thread():
    PLAN.clear()
    PLAN["Find corruption cases"] = {"route": "GQ", "restated_query": "corruption cases"}
    PLAN["What was the verdict?"] = {"route": "GQ", "restated_query": "verdict"}
    agent = make_agent(thread="mem")
    agent.run("Find corruption cases")
    agent.run("What was the verdict?")
    state = agent.graph.get_state({"configurable": {"thread_id": "mem"}})
    assert len(state.values["messages"]) == 4


def test_threads_are_isolated():
    PLAN.clear()
    PLAN["Find corruption cases"] = {"route": "GQ", "restated_query": "corruption cases"}
    agent = make_agent(thread="main")
    agent.run("Find corruption cases")
    agent.run("Find corruption cases", thread_id="other")
    other = agent.graph.get_state({"configurable": {"thread_id": "other"}})
    assert len(other.values["messages"]) == 2


def test_render_history_excludes_the_current_turn():
    from langchain_core.messages import AIMessage, HumanMessage
    h = W._render_history([HumanMessage(content="q1"), AIMessage(content="a1"),
                           HumanMessage(content="q2")])
    assert "q1" in h and "a1" in h and "q2" not in h
    assert "(no previous conversation)" in W._render_history([HumanMessage(content="q")])


# --------------------------------------------------------------------------
# Clickable sources
# --------------------------------------------------------------------------

def test_run_detailed_returns_sources():
    PLAN.clear()
    PLAN["Find bail cases"] = {"route": "GQ", "restated_query": "bail cases"}
    out = make_agent(thread="src").run_detailed("Find bail cases")
    assert out["sources"], "no sources returned"
    s = out["sources"][0]
    assert s["n"] == 1
    assert s["doc_id"] == "crl_12"
    assert s["paragraph"] == "7"
    assert s["case_no"] == "12/2019"
    assert s["judgment_type"] == "Criminal Appeal"
    assert "Section 302 PPC" in s["provisions"]
    assert s["excerpt"]


def test_sources_are_numbered_for_citation_markers():
    PLAN.clear()
    PLAN["Find bail cases"] = {"route": "GQ", "restated_query": "bail cases"}
    agent = make_agent(thread="num")
    out = agent.run_detailed("Find bail cases")
    assert [s["n"] for s in out["sources"]] == list(range(1, len(out["sources"]) + 1))
    assert "[S1]" in agent.llm.calls[-1]  # the model saw the same numbering


def test_grounded_flag_false_on_not_found():
    class NotFoundLLM(FakeLLM):
        def invoke(self, prompt):
            self.calls.append(prompt)
            return types.SimpleNamespace(
                content=W.NOT_FOUND_PHRASE + ".\n\nTry narrowing the search.")

    PLAN.clear()
    PLAN["divorce law in Sweden"] = {"route": "GQ", "restated_query": "divorce Sweden"}
    out = make_agent(NotFoundLLM(), thread="nf").run_detailed("divorce law in Sweden")
    assert out["grounded"] is False
    assert out["sources"], "sources must still show, so the refusal is verifiable"


def test_not_found_is_rechecked_once_when_excerpts_were_retrieved():
    class FirstMissLLM(FakeLLM):
        def invoke(self, prompt):
            self.calls.append(prompt)
            if W.ANSWER_RECHECK.strip() in prompt:
                return types.SimpleNamespace(content="Found it in paragraph 7.")
            return types.SimpleNamespace(content=W.NOT_FOUND_PHRASE + ".")

    PLAN.clear()
    PLAN["section 302 penalty"] = {"route": "GQ", "restated_query": "Section 302 penalty"}
    llm = FirstMissLLM()
    out = make_agent(llm, thread="recheck").run_detailed("section 302 penalty")
    assert out["answer"] == "Found it in paragraph 7."
    assert out["grounded"] is True
    assert sum(W.ANSWER_RECHECK.strip() in c for c in llm.calls) == 1


def test_grounded_flag_true_on_normal_answer():
    PLAN.clear()
    PLAN["Find bail cases"] = {"route": "GQ", "restated_query": "bail cases"}
    assert make_agent(thread="g").run_detailed("Find bail cases")["grounded"] is True


def test_rejected_query_has_no_sources():
    PLAN.clear()
    PLAN["weather"] = {"route": "REJECT", "clarification": "Not legal."}
    out = make_agent(thread="nosrc").run_detailed("weather")
    assert out["sources"] == []
    assert out["grounded"] is False


def test_clean_excerpt_strips_embedding_context():
    raw = ("[Case: 12/2019] [Paragraph 7] [Section: Analysis]\n\n"
           "[Previous: ...earlier text]\n\n7. The real judgment text begins here.")
    assert W.clean_excerpt(raw).startswith("7. The real judgment text")
    assert "[Case:" not in W.clean_excerpt(raw)
    assert W.clean_excerpt("") == ""


def test_source_records_tolerate_json_string_metadata():
    hits = [{"score": 0.5, "doc": "text", "meta": {
        "doc_id": "d", "citations": '["2016 SCMR 2073"]',
        "provisions": '["Section 302"]', "judges": "not json", "parties": "{}"}}]
    r = W.source_records(hits)[0]
    assert r["citations"] == ["2016 SCMR 2073"]
    assert r["provisions"] == ["Section 302"]
    assert r["judges"] == []     # unparseable falls back to the default
    assert r["parties"] == {}


# --------------------------------------------------------------------------
# Model compatibility and error handling
# --------------------------------------------------------------------------

def test_handles_gemini_3_list_content_blocks():
    """Gemini 3.x returns content as a list of blocks, not a string."""
    class BlockLLM(FakeLLM):
        def invoke(self, prompt):
            self.calls.append(prompt)
            return types.SimpleNamespace(content=[
                {"type": "text", "text": "Bail was ", "extras": {"signature": "xx"}},
                {"type": "text", "text": "granted."},
            ])

    PLAN.clear()
    PLAN["Find bail cases"] = {"route": "GQ", "restated_query": "bail cases"}
    out = make_agent(BlockLLM(), thread="blocks").run_detailed("Find bail cases")
    assert out["answer"] == "Bail was granted.", out["answer"]


def test_message_text_accepts_every_response_shape():
    assert W.message_text(types.SimpleNamespace(content="plain")) == "plain"
    assert W.message_text(types.SimpleNamespace(
        content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])) == "ab"
    assert W.message_text(types.SimpleNamespace(
        content=[{"type": "reasoning", "x": 1}, {"type": "text", "text": "only"}])) == "only"
    assert W.message_text(types.SimpleNamespace(content=[])) == ""


def test_api_failure_is_reported_as_error_not_reject():
    """A 429 must not look like "your question isn't legal"."""
    class QuotaLLM(FakeLLM):
        def with_structured_output(self, schema):
            class Broken:
                def invoke(self, prompt):
                    raise RuntimeError("429 RESOURCE_EXHAUSTED. Quota exceeded, limit: 20")
            return Broken()

    agent = make_agent(QuotaLLM(), thread="quota")
    out = agent.run_detailed("Find bail cases under Section 497 CrPC")
    assert out["route"] == "ERROR", out["route"]
    assert "quota" in out["answer"].lower()
    assert agent.llm.calls == []   # no second doomed call


def test_bad_api_key_is_reported_clearly():
    class AuthLLM(FakeLLM):
        def with_structured_output(self, schema):
            class Broken:
                def invoke(self, prompt):
                    raise RuntimeError("401 UNAUTHENTICATED: API key not valid")
            return Broken()

    out = make_agent(AuthLLM(), thread="auth").run_detailed("Find bail cases")
    assert out["route"] == "ERROR"
    assert W.API_KEY_ENV in out["answer"]


def test_transient_error_still_falls_back_to_tag_parsing():
    class FlakyLLM(FakeLLM):
        def with_structured_output(self, schema):
            class Broken:
                def invoke(self, prompt):
                    raise RuntimeError("schema binding unavailable")
            return Broken()

        def invoke(self, prompt):
            self.calls.append(prompt)
            if "Classify the legal query" in prompt:
                return types.SimpleNamespace(
                    content="<answer><valid>[GQ] bail cases</valid></answer>")
            return types.SimpleNamespace(content="<ANSWER>")

    out = make_agent(FlakyLLM(), thread="flaky").run_detailed("Find bail cases")
    assert out["route"] == "GQ"
    assert out["answer"] == "<ANSWER>"


def test_retrieval_failure_does_not_crash_a_general_query():
    def boom(*a, **k):
        raise RuntimeError("chroma down")

    PLAN.clear()
    PLAN["Find bail cases"] = {"route": "GQ", "restated_query": "bail cases"}
    agent = make_agent(thread="err")
    W.search_hybrid = boom
    try:
        out = agent.run_detailed("Find bail cases")
    finally:
        W.search_hybrid = fake_search_hybrid
    assert isinstance(out["answer"], str) and out["answer"]


def test_compare_surfaces_a_retrieval_error():
    def boom(*a, **k):
        raise RuntimeError("chroma down")

    PLAN.clear()
    PLAN["Compare X and Y"] = {"route": "CQ", "restated_query": "Compare X and Y",
                               "compare_a": "X", "compare_b": "Y"}
    agent = make_agent(thread="err2")
    W.search_hybrid = boom
    try:
        out = agent.run_detailed("Compare X and Y")
    finally:
        W.search_hybrid = fake_search_hybrid
    assert "Error retrieving documents" in out["answer"]


def test_graph_has_every_node():
    nodes = set(make_agent(thread="shape").graph.get_graph().nodes)
    assert {"classify", "clarify", "retrieve", "retrieve_pair",
            "answer", "summarize", "compare"} <= nodes


def test_run_returns_a_plain_string():
    PLAN.clear()
    PLAN["Find corruption cases"] = {"route": "GQ", "restated_query": "corruption cases"}
    assert make_agent(thread="str").run("Find corruption cases") == "<ANSWER>"


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


def test_review_cells_keep_the_citation_out_of_the_text():
    raw = '{"Parties": {"answer": "Aslam and Tariq [S3].", "source": 0}, "Amount": {"answer": "Rs. 900,000 [S2] , unpaid", "source": 2}}'
    cells = W.WakeelAgent._parse_row(raw, ["Parties", "Amount"], 1, 3)
    assert cells == [{"text": "Aslam and Tariq.", "source": 3}, {"text": "Rs. 900,000, unpaid", "source": 2}]


def test_an_urdu_document_searches_english_judgments_with_english_terms():
    fir = ("ایف آئی آر زیر دفعہ 154 ضابطہ فوجداری۔ جرم: زیر دفعہ 489-ایف تعزیرات پاکستان۔ "
           "میں نے ملزم کو نو لاکھ روپے بطور قرض دیے، چیک واپس ہو گیا۔")
    terms = W.urdu_search_terms(fir)
    assert terms.startswith("section 154 CrPC section 489-F PPC")
    assert "cheque dishonoured" in terms and "loan" in terms
    assert W.urdu_search_terms("FIR under section 489-F PPC for a bounced cheque") == ""   # English: unchanged


class CaseLLM(FakeLLM):
    """Plays the agent room's model calls: the orchestrator's JSON, then a cited answer for every writer."""
    def __init__(self, kind="situation", draft="cheque_notice", judgments=None, fail_on=None):
        super().__init__()
        self.kind, self.draft, self.judgments, self.fail_on = kind, draft, judgments or [], fail_on

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.fail_on and self.fail_on in str(prompt):
            raise RuntimeError("503 UNAVAILABLE")
        if "You are the orchestrator" in str(prompt):
            return types.SimpleNamespace(content=json.dumps({
                "kind": self.kind, "reason": "You need help with a bounced cheque.", "judgments": self.judgments,
                "issue": "Remedies when a cheque given to repay a loan is dishonoured", "side": "complainant",
                "facts": ["Lent Rs. 900,000", "Cheque bounced on 1 March 2024"], "provisions": ["Section 489-F PPC"],
                "goal": "Recover the money", "dates": ["1 March 2024"], "draft": self.draft}))
        return types.SimpleNamespace(content="The court held the cheque was dishonestly issued [S1].")


def _room(llm, query="My cheque bounced", document="", doc_ids=None):
    W.search_hybrid = fake_search_hybrid
    agent = W.WakeelAgent(llm, thread_id="room")
    return list(agent.agent_room(query, document, doc_ids, mode="citizen", language="en"))


def _team(events):
    return next(e for e in events if e["type"] == "team")


def _done(events):
    return {e["id"]: e for e in events if e.get("type") == "agent" and e.get("status") == "done"}


def test_agent_room_situation_team_shares_one_set_of_passages():
    llm = CaseLLM()
    events = _room(llm, document="FIR No. 214/2024 under section 489-F PPC")
    team = _team(events)
    assert team["kind"] == "document"                                  # a document always gets the document team
    events = _room(CaseLLM())
    team = _team(events)
    assert team["kind"] == "situation" and [a["id"] for a in team["team"]] == W.TEAMS["situation"][0]
    assert ["research", "strategy"] in team["edges"] and all(len(e) == 2 for e in team["edges"])
    assert events[-1] == {"type": "done"} and events[0]["id"] == "orchestrator"
    done = _done(events)
    assert {"orchestrator", "research", "strategy", "draft"} <= set(done)
    assert done["research"]["result"]["sources"] == done["strategy"]["result"]["sources"] == done["draft"]["result"]["sources"]
    handoffs = {(e["from"], e["id"]) for e in events if e.get("from")}
    assert {("orchestrator", "research"), ("orchestrator", "statutes"), ("research", "precedents"),
            ("research", "strategy"), ("research", "draft")} <= handoffs
    assert all(tuple(h) in {tuple(e) for e in team["edges"]} for h in handoffs)   # every hand-off is an edge drawn


def test_agent_room_question_team_runs_the_explainer_after_the_research():
    events = _room(CaseLLM(kind="question"), "Can bail be cancelled on a compromise?")
    team = _team(events)
    assert team["kind"] == "question" and "counter" in [a["id"] for a in team["team"]]
    done = _done(events)
    assert {"research", "counter", "explainer"} <= set(done)
    order = [e["id"] for e in events if e.get("status") == "done"]
    assert order.index("research") < order.index("explainer")
    explainer_start = next(e for e in events if e.get("id") == "explainer" and e.get("status") == "working")
    assert explainer_start["from"] == "research"


def test_agent_room_reads_and_compares_chosen_judgments():
    saved, W.METADATA_DIR = W.METADATA_DIR, _fake_corpus()
    try:
        one = _room(CaseLLM(), "", doc_ids=["lhc_a"])
        assert _team(one)["kind"] == "judgment"
        assert {"reader", "timeline", "similar"} <= set(_done(one))
        assert _done(one)["similar"]["data"]["doc_ids"] and "lhc_a" not in _done(one)["similar"]["data"]["doc_ids"]
        two = _room(CaseLLM(), "", doc_ids=["lhc_a", "lhc_b"])
        assert _team(two)["kind"] == "compare"
        assert {"reader_a", "reader_b", "comparator"} <= set(_done(two))
        prec = next(e for e in two if e.get("id") == "precedents" and e.get("data"))
        assert prec["data"]["doc_ids"] == ["lhc_a", "lhc_b"]
    finally:
        W.METADATA_DIR = saved


def test_agent_room_falls_back_and_reports_failures():
    # Asked to compare, but the judgments named cannot be found: it answers as a question instead.
    W.WakeelAgent._resolve_judgment, saved = staticmethod(lambda name: ""), W.WakeelAgent._resolve_judgment
    try:
        events = _room(CaseLLM(kind="compare", judgments=["X v Y", "P v Q"]), "Compare X v Y with P v Q")
        assert _team(events)["kind"] == "question"
    finally:
        W.WakeelAgent._resolve_judgment = saved
    skipped = _room(CaseLLM(draft="not a kind"))
    assert [e for e in skipped if e.get("id") == "draft"][0]["status"] == "skipped"
    failed = _room(CaseLLM(fail_on="case strategy memo"))
    assert [e for e in failed if e.get("id") == "strategy"][-1]["status"] == "failed"
    assert "research" in _done(failed)                                      # the others still finish
    nothing = _room(CaseLLM(), "  ")
    assert nothing[1]["status"] == "failed" and nothing[-1] == {"type": "done"}
    broken = _room(CaseLLM(fail_on="You are the orchestrator"))
    assert broken[-2]["id"] == "orchestrator" and broken[-2]["status"] == "failed"


class StreamingCaseLLM(CaseLLM):
    """The same replies, given a few characters at a time, as a streaming chat model gives them."""
    def stream(self, prompt):
        text = self.invoke(prompt).content
        for i in range(0, len(text), 7):
            yield types.SimpleNamespace(content=text[i:i + 7])


def test_agent_room_streams_each_agents_words_before_its_result():
    saved, W.STREAM_FLUSH_CHARS = W.STREAM_FLUSH_CHARS, 10
    try:
        events = _room(StreamingCaseLLM(kind="question"), "Can bail be cancelled on a compromise?")
    finally:
        W.STREAM_FLUSH_CHARS = saved
    done = _done(events)
    for agent_id in ("research", "counter", "explainer"):
        chunks = [i for i, e in enumerate(events) if e["type"] == "chunk" and e["id"] == agent_id]
        assert len(chunks) > 1 and events[chunks[0]]["new"] and not events[chunks[1]]["new"]
        finished = events.index(done[agent_id])
        assert chunks[-1] < finished
        assert "".join(events[i]["text"] for i in chunks) == done[agent_id]["result"]["answer"]
    # The explainer's words only come after the research agent it works from has reported.
    first_explainer_word = next(i for i, e in enumerate(events) if e["type"] == "chunk" and e["id"] == "explainer")
    assert events.index(done["research"]) < first_explainer_word
    assert not [e for e in events if e["type"] == "chunk" and e["id"] == "orchestrator"]    # its JSON is not streamed


def test_a_watched_request_reports_each_agent_and_its_words_across_threads():
    saved, W.METADATA_DIR = W.METADATA_DIR, _fake_corpus()
    log = []
    watch, token = W.watch_begin(log.append)
    try:
        agent = make_agent(RowLLM(), thread="table")
        agent.llm.stream = lambda prompt: iter([types.SimpleNamespace(content=agent.llm.invoke(prompt).content)])
        agent.review_table(["lhc_a", "lhc_b"], ["Outcome"])
    finally:
        W.watch_end(token)
        W.METADATA_DIR = saved
    assert watch.log == log
    stages = {(e["id"], e["status"]) for e in log if e["type"] == "stage"}
    assert {("load-lhc_a", "done"), ("load-lhc_b", "done"), ("row-lhc_a", "working"), ("row-lhc_a", "done"),
            ("row-lhc_b", "done")} <= stages                         # each reviewer reports from its own thread
    assert {e["id"] for e in log if e["type"] == "chunk"} == {"row-lhc_a", "row-lhc_b"}
    assert next(e for e in log if e["id"] == "row-lhc_a")["name"].startswith("Reviewer · ")
    assert W.watching() is None
    log.clear()
    make_agent(RowLLM(), thread="t2").review_table(["lhc_a"], ["Outcome"])   # nobody watching: nothing reported
    assert log == []



class CitatorLLM(FakeLLM):
    def invoke(self, prompt):
        self.calls.append(prompt)
        if "You are a legal citator" in str(prompt):
            return types.SimpleNamespace(content=json.dumps([
                {"n": 1, "treatment": "followed", "reason": "Applied to grant bail.", "quote": "lhc_a paragraph 3"},
                {"n": 2, "treatment": "overruled-ish", "reason": "Unclear.", "quote": "words that are not there"}]))
        return super().invoke(prompt)


def test_citator_labels_each_judgment_and_drops_quotes_it_cannot_find():
    saved, W.METADATA_DIR = W.METADATA_DIR, _fake_corpus()
    try:
        agent = make_agent(CitatorLLM(), thread="cite")
        out = agent.citator("2013 SCMR 1205", {"lhc_a": ["3"], "lhc_b": ["7"]})
        assert out["route"] == "CITATOR" and len(out["results"]) == 2
        a, b = out["results"]
        assert a == {"doc_id": "lhc_a", "paragraph": "3", "treatment": "followed", "reason": "Applied to grant bail.",
                     "quote": "lhc_a paragraph 3", "quote_found": True}
        assert b["treatment"] == "mentioned" and b["quote"] == "" and not b["quote_found"]   # unknown label, unfound quote
        assert "lhc_b paragraph 7" in agent.llm.calls[-1]                                    # the citing paragraph was read
        assert out["counts"]["followed"] == 1 and "1 followed" in out["answer"]
        assert agent.citator("", {"lhc_a": ["1"]})["route"] == "ERROR"
        assert agent.citator("2013 SCMR 1205", {"missing": ["1"]})["route"] == "ERROR"
    finally:
        W.METADATA_DIR = saved


def test_opposing_counsel_and_custom_workflows_research_and_cite():
    agent = make_agent(thread="opp")
    out = agent.run_workflow("opponent", {"Your draft or argument": "The cheque was security, so 489-F does not apply."})
    prompt = agent.llm.calls[-1]
    assert out["route"] == "OPPONENT" and "You are opposing counsel" in prompt and "The cheque was security" in prompt
    assert out["sources"]                                                         # it searched the judgments
    out = agent.run_workflow("custom", {"Instructions": "List my bail grounds as numbered points", "Facts": "489-F, in custody"})
    assert out["route"] == "CUSTOM" and "do not apply" in agent.llm.calls[-1] and "List my bail grounds" in agent.llm.calls[-1]



def test_moot_court_asks_one_question_a_turn_then_gives_feedback():
    agent = make_agent(thread="moot")
    first = agent.moot_turn("Accused under 489-F; cheque was security.", "accused", "a bail application", [])
    prompt = agent.llm.calls[-1]
    assert first["route"] == "MOOT" and not first["done"] and first["question"] == 1 and first["of"] == 5
    assert "ask ONE new question" in prompt and "the hearing has not started" in prompt and first["sources"]
    history = [{"role": "judge", "text": "Why is this not within the prohibitory clause?"},
               {"role": "counsel", "text": "The maximum sentence is three years."}]
    agent.moot_turn("Accused under 489-F.", "accused", "a bail application", history)
    assert "COUNSEL: The maximum sentence is three years." in agent.llm.calls[-1]
    end = agent.moot_turn("Accused under 489-F.", "accused", "a bail application", history, finish=True)
    assert end["done"] and "## Score" in agent.llm.calls[-1]
    many = [{"role": r, "text": "x"} for _ in range(5) for r in ("judge", "counsel")]
    assert agent.moot_turn("Facts.", "accused", "bail", many)["done"]          # five questions, then feedback
    assert agent.moot_turn("", "accused", "bail", [])["route"] == "ERROR"


def test_evidence_check_reads_the_papers_whole():
    agent = make_agent(thread="ev")
    papers = "=== FIR ===\n" + "The cheque was dishonoured on 1 March 2024. " * 300
    out = agent.run_workflow("evidence", {"Draft": "The cheque bounced on 5 March 2024.", "Papers": papers})
    prompt = agent.llm.calls[-1]
    assert out["route"] == "EVIDENCE" and "Not in the papers" in prompt and "bounced on 5 March 2024" in prompt
    assert prompt.count("dishonoured on 1 March 2024") > 150                    # more than the 6,000 characters a fact gets
