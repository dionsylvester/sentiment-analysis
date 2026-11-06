"""
Usage:
python absa.py --task polarity --provider ollama      # the benchmark
python absa.py --task extract --provider ollama       # the app feature
"""
import argparse
import json
import re
from typing import Literal
import pandas as pd
from pydantic import BaseModel, ConfigDict

import llm_sentiment
from llm_sentiment import PROVIDERS, SentimentOutput, call_llm, llm_metrics, load_prompt, make_client, run_name, run_resumable

SEMEVAL_LABELS = {-1: "negative", 0: "neutral", 1: "positive"}


class Aspect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aspect: str
    sentiment: Literal["negative", "neutral", "positive"]


class AspectsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aspects: list[Aspect]


def load_semeval_test() -> pd.DataFrame:
    df = pd.read_csv("dataset/semeval_test.csv").reset_index(names="id")
    df["sentiment"] = df["label"].map(SEMEVAL_LABELS)
    return df


def aspect_polarity(client, provider: str, system_prompt: str, sentence: str, aspect: str) -> dict:
    result = call_llm(client, provider, system_prompt, {"sentence": sentence, "aspect": aspect}, SentimentOutput)
    parsed = result.pop("parsed")
    return {"pred": parsed.sentiment if parsed else None, **result}


def extract_aspects(client, provider: str, system_prompt: str, text: str) -> dict:
    result = call_llm(client, provider, system_prompt, {"text": text}, AspectsOutput)
    parsed = result.pop("parsed")
    return {"aspects": [a.model_dump() for a in parsed.aspects] if parsed else None, **result}


def evaluate_polarity(provider: str, prompt_name: str = "absa_polarity_v1", limit=None, client=None) -> dict | None:
    client = client or make_client(provider)
    system_prompt = load_prompt(prompt_name)
    name = run_name(provider, prompt_name)

    preds = run_resumable(
        load_semeval_test(), llm_sentiment.RESULTS_DIR / f"{name}_predictions.csv",
        lambda row: {"aspect": row.aspect, "sentiment": row.sentiment,
                     **aspect_polarity(client, provider, system_prompt, row.text, row.aspect)},
        provider, limit,
    )
    if preds is None:
        return None

    metrics = {"provider": provider, "model": PROVIDERS[provider]["model"], "prompt": prompt_name, **llm_metrics(preds)}
    (llm_sentiment.RESULTS_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def match_aspect(gold: str, extracted: list[dict]) -> dict | None:
    gold = gold.lower().strip()
    names = [(a["aspect"].lower().strip(), a) for a in extracted if a["aspect"].strip()]
    exact = next((a for name, a in names if name == gold), None)
    if exact:
        return exact

    def contains(outer, inner):
        return re.search(rf"\b{re.escape(inner)}\b", outer) is not None

    return next((a for name, a in names if contains(name, gold) or contains(gold, name)), None)


def evaluate_extraction(provider: str, prompt_name: str = "absa_extract_v1", limit=None, client=None) -> dict | None:
    client = client or make_client(provider)
    system_prompt = load_prompt(prompt_name)
    name = run_name(provider, prompt_name)

    def predict(row):
        result = extract_aspects(client, provider, system_prompt, row.text)
        result["aspects"] = json.dumps(result["aspects"]) if result["aspects"] is not None else None
        return result

    gold = load_semeval_test()
    sentences = pd.DataFrame({"text": gold["text"].unique()}).reset_index(names="id")
    preds = run_resumable(sentences, llm_sentiment.RESULTS_DIR / f"{name}_predictions.csv", predict, provider, limit)
    if preds is None:
        return None

    decoded = preds["aspects"].map(lambda s: json.loads(s) if isinstance(s, str) else None)
    extracted = dict(zip(sentences["text"], decoded))
    found, correct = 0, 0
    for row in gold.itertuples():
        match = match_aspect(row.aspect, extracted[row.text] or [])
        found += match is not None
        correct += match is not None and match["sentiment"] == row.sentiment

    metrics = {
        "provider": provider,
        "model": PROVIDERS[provider]["model"],
        "prompt": prompt_name,
        "sentences": len(preds),
        "gold_aspects": len(gold),
        "invalid_rate": float(decoded.isna().mean()),
        "aspect_recall": found / len(gold), 
        "polarity_accuracy_on_found": correct / found if found else 0.0,
        "end_to_end_accuracy": correct / len(gold), 
        "aspects_per_sentence": float(decoded.dropna().map(len).mean()),
        "latency_ms": float(preds["latency_ms"].mean()),
        "latency_ms_median": float(preds["latency_ms"].median()),
        "prompt_tokens_mean": float(preds["prompt_tokens"].mean()),
        "completion_tokens_mean": float(preds["completion_tokens"].mean()),
    }
    (llm_sentiment.RESULTS_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["polarity", "extract"], required=True)
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--limit", type=int, help="max new predictions in this run")
    args = parser.parse_args()

    run = evaluate_polarity if args.task == "polarity" else evaluate_extraction
    metrics = run(args.provider, limit=args.limit)
    if metrics:
        print(json.dumps(metrics, indent=2))
