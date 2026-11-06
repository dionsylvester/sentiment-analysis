import json
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import absa
import llm_sentiment


def tag(content, name):
    return re.search(f"<{name}>(.*?)</{name}>", content, re.S).group(1)


class FakeClient:
    def __init__(self, gold, empty_text=None):
        self.polarity = {(r.text, r.aspect): r.sentiment for r in gold.itertuples()}
        self.aspects = gold.groupby("text")[["aspect", "sentiment"]].apply(lambda g: g.to_dict("records")).to_dict()
        self.empty_text = empty_text
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, messages, response_format, **kwargs):
        self.calls += 1
        user = messages[-1]["content"]
        if self.calls == 2:
            content = "not json"
        elif response_format["json_schema"]["name"] == "AspectsOutput":
            text = tag(user, "text")
            content = json.dumps({"aspects": [] if text == self.empty_text else self.aspects[text]})
        else:
            content = json.dumps({"sentiment": self.polarity[(tag(user, "sentence"), tag(user, "aspect"))]})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=10),
        )


def test_polarity_resumes_and_scores():
    gold = absa.load_semeval_test()
    client = FakeClient(gold)
    with tempfile.TemporaryDirectory() as tmp:
        llm_sentiment.RESULTS_DIR = Path(tmp)
        assert absa.evaluate_polarity("ollama", limit=4, client=client) is None
        metrics = absa.evaluate_polarity("ollama", client=client)

    assert client.calls == len(gold) + 2, "resume must not repeat rows (+1 warm-up per run)"
    assert metrics["invalid_rate"] == 1 / len(gold)
    assert metrics["accuracy"] == (len(gold) - 1) / len(gold)


def test_extraction_recall():
    gold = absa.load_semeval_test()
    sentences = gold["text"].unique()
    invalid_text = sentences[0]
    empty_text = sentences[5]
    client = FakeClient(gold, empty_text=empty_text)
    with tempfile.TemporaryDirectory() as tmp:
        llm_sentiment.RESULTS_DIR = Path(tmp)
        metrics = absa.evaluate_extraction("ollama", client=client)

    missed = (gold["text"] == invalid_text).sum() + (gold["text"] == empty_text).sum()
    assert metrics["invalid_rate"] == 1 / len(sentences)
    assert metrics["aspect_recall"] == (len(gold) - missed) / len(gold)
    assert metrics["polarity_accuracy_on_found"] == 1.0


def test_match_aspect():
    def asp(*names):
        return [{"aspect": n, "sentiment": "neutral"} for n in names]

    assert absa.match_aspect("spicy tuna roll", asp("Spicy Tuna Roll"))["aspect"] == "Spicy Tuna Roll"
    assert absa.match_aspect("tuna roll", asp("spicy tuna roll"))["aspect"] == "spicy tuna roll"
    assert absa.match_aspect("rice medley", asp("rice", "rice medley"))["aspect"] == "rice medley"
    assert absa.match_aspect("Portuguese cheese cart", asp("port")) is None
    assert absa.match_aspect("billed", asp("bill")) is None
    assert absa.match_aspect("service", asp("food", "")) is None


if __name__ == "__main__":
    test_polarity_resumes_and_scores()
    test_extraction_recall()
    test_match_aspect()
    print("ok")
