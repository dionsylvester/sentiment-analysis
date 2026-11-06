import json
from types import SimpleNamespace
import numpy as np

import inference


def expect_error(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, messages, response_format, **kwargs):
        self.calls += 1
        user = messages[-1]["content"]
        if response_format["json_schema"]["name"] == "AspectsOutput":
            content = json.dumps({"aspects": [{"aspect": "food", "sentiment": "positive"},
                                              {"aspect": "service", "sentiment": "negative"}]})
        elif "garbage" in user:
            content = "not json"
        else:
            content = json.dumps({"sentiment": "positive" if "love" in user else "neutral"})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


def test_svc_and_rules():
    sents = ["I love this so much, best day ever!", "not good", "!!!"]
    plain = inference.predict(sents, "linearsvc")
    ruled = inference.predict(sents, "linearsvc_rules")
    assert plain[0]["sentiment"] == "positive"
    assert ruled[1] == {"sentence": "not good", "sentiment": "negative", "note": "Detected negation before positive word"}
    assert plain[2]["sentiment"] == "neutral" and plain[2]["note"] == "Empty after cleaning."
    assert all(r["sentiment"] in inference.LABELS for r in plain + ruled)


def test_rules_false_positives():
    from sentiment_rules import apply_sentiment_rules
    for text in ["I love it, never crashes", "Best app, zero bugs", "not only good but amazing", "ok I absolutely love this",
                 "zero bugs, love it", "no problem, great service", "never crashes, love it"]:
        assert apply_sentiment_rules(text, "positive") == ("positive", []), text
    assert apply_sentiment_rules("great scam", "positive")[0] == "negative"
    assert apply_sentiment_rules("it was ok", "positive")[0] == "neutral"
    assert apply_sentiment_rules("not good", "positive")[0] == "negative"
    assert apply_sentiment_rules("never good", "positive")[0] == "negative"


def test_validation():
    expect_error(ValueError, inference.predict, [], "linearsvc")
    expect_error(ValueError, inference.predict, ["ok"], "bert")
    expect_error(ValueError, inference.predict, [" "], "linearsvc")
    expect_error(ValueError, inference.predict, ["x" * 1001], "linearsvc")
    expect_error(ValueError, inference.predict, ["ok"] * 11, "llm") 
    expect_error(ValueError, inference.absa, "")


def test_llm_cache_and_invalid_output():
    fake = FakeLLM()
    inference._llm_client = lambda: fake
    inference._llm_label.cache_clear()
    out = inference.predict(["I love it", "meh", "garbage", "I love it"], "llm")
    assert [r["sentiment"] for r in out] == ["positive", "neutral", None, "positive"]
    assert out[2]["note"] == "Invalid model output."
    assert fake.calls == 3, "the repeated sentence must come from the cache"


def test_llm_quota():
    class FakeRateLimit(Exception):
        pass

    def raise_limit(**kwargs):
        raise FakeRateLimit()

    inference.RateLimitError = FakeRateLimit
    inference._llm_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=raise_limit)))
    inference._llm_label.cache_clear()
    expect_error(inference.QuotaExceeded, inference.predict, ["new sentence"], "llm")


def test_llm_down():
    class FakeTimeout(Exception):
        pass

    def raise_timeout(**kwargs):
        raise FakeTimeout()

    inference.APIError = FakeTimeout
    inference._llm_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=raise_timeout)))
    inference._llm_label.cache_clear()
    inference._absa.cache_clear()
    expect_error(inference.ModelUnavailable, inference.predict, ["another sentence"], "llm")
    expect_error(inference.ModelUnavailable, inference.absa, "The food was great.")


def test_absa():
    inference._llm_client = lambda: FakeLLM()
    inference._absa.cache_clear()
    aspects = inference.absa("The food was great but the service was slow.")
    assert aspects == [{"aspect": "food", "sentiment": "positive"}, {"aspect": "service", "sentiment": "negative"}]


def test_distilroberta_batch():
    class FakeTokenizer:
        def encode_batch(self, sentences):
            return [SimpleNamespace(ids=[0, 5, 2], attention_mask=[1, 1, 1]) for _ in sentences]

    class FakeSession:
        def run(self, _, feeds):
            assert feeds["input_ids"].dtype == np.int64 and feeds["input_ids"].shape == (2, 3)
            return [np.array([[0.1, 0.2, 3.0], [2.0, 0.1, 0.1]])]

    inference._distilroberta = lambda: (FakeTokenizer(), FakeSession())
    out = inference.predict(["great", "awful"], "distilroberta")
    assert [r["sentiment"] for r in out] == ["positive", "negative"]


def test_distilroberta_not_configured():
    import importlib
    fresh = importlib.reload(inference)
    fresh.HF_MODEL_REPO = None
    expect_error(fresh.ModelUnavailable, fresh.predict, ["ok"], "distilroberta")


if __name__ == "__main__":
    test_svc_and_rules()
    test_rules_false_positives()
    test_validation()
    test_llm_cache_and_invalid_output()
    test_llm_quota()
    test_llm_down()
    test_absa()
    test_distilroberta_batch()
    test_distilroberta_not_configured()
    print("ok")
