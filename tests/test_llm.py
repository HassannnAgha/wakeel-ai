"""Tests for choosing and building the chat model: Gemini or OpenAI.

Run:  .venv/bin/python -m pytest tests/ -v
 or:  .venv/bin/python tests/test_llm.py

Builds the client objects only; no request is sent, so no real key is needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wakeel_ai as W  # noqa: E402


class fake_keys:
    """Set dummy API keys for the duration of a test, then restore the environment."""

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in ("GOOGLE_API_KEY", "OPENAI_API_KEY")}
        os.environ["GOOGLE_API_KEY"] = "test-google-key"
        os.environ["OPENAI_API_KEY"] = "test-openai-key"

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_both_providers_are_configured():
    assert set(W.PROVIDERS) == {"gemini", "openai"}
    assert W.PROVIDERS["gemini"]["key_env"] == "GOOGLE_API_KEY"
    assert W.PROVIDERS["openai"]["key_env"] == "OPENAI_API_KEY"
    assert W.API_KEY_ENV == W.PROVIDERS[W.LLM_PROVIDER]["key_env"]


def test_openai_provider_builds_chat_openai_with_default_model():
    with fake_keys():
        llm = W.make_llm(provider="openai", temperature=0)
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "gpt-4o-mini"
    assert llm.temperature == 0


def test_openai_reasoning_models_keep_the_default_temperature():
    with fake_keys():
        llm = W.make_llm(provider="openai", model="gpt-5-mini", temperature=0)
    assert llm.model_name == "gpt-5-mini"
    assert llm.temperature is None


def test_gemini_provider_builds_chat_google():
    with fake_keys():
        llm = W.make_llm(provider="gemini", temperature=0)
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert "gemini-3.5-flash" in llm.model


def test_explicit_model_overrides_the_default():
    with fake_keys():
        llm = W.make_llm(provider="openai", model="gpt-4.1-mini")
    assert llm.model_name == "gpt-4.1-mini"


def test_openai_client_supports_structured_classification():
    with fake_keys():
        llm = W.make_llm(provider="openai")
    # The classifier is built this way in WakeelAgent; it must not raise.
    assert llm.with_structured_output(W.QueryClassification) is not None


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


def test_chat_model_falls_back_to_the_other_provider():
    with fake_keys():
        llm = W.make_chat_model(temperature=0)
    assert type(llm).__name__ == "RunnableWithFallbacks"
    primary, (backup,) = llm.runnable, llm.fallbacks
    timeout = getattr(primary, "timeout", None) or getattr(primary, "request_timeout", None)
    assert primary.max_retries == 1 and timeout == W.PRIMARY_TIMEOUT_S   # fail over quickly
    assert {type(primary).__name__, type(backup).__name__} == {"ChatGoogleGenerativeAI", "ChatOpenAI"}
    # The router's structured output keeps the same fallback.
    with fake_keys():
        classifier = W.structured(llm, W.QueryClassification)
    assert type(classifier).__name__ == "RunnableWithFallbacks" and len(classifier.fallbacks) == 1


def test_fallback_can_be_turned_off():
    os.environ["WAKEEL_FALLBACK"] = "0"
    try:
        with fake_keys():
            llm = W.make_chat_model(temperature=0)
        assert type(llm).__name__ in ("ChatGoogleGenerativeAI", "ChatOpenAI")
    finally:
        os.environ.pop("WAKEEL_FALLBACK")
