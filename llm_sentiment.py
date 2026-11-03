"""
Usage:
python llm_sentiment.py --provider ollama --prompt sentiment_zero_shot_v1
python llm_sentiment.py --provider groq --prompt sentiment_few_shot_v1 --limit 200
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Literal
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError

from model_training import load_test_data, summarize

PROVIDERS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b", "api_key_env": None, "extra": {}},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-20b",
             "api_key_env": "GROQ_API_KEY", "extra": {"reasoning_effort": "low"}, "pause_s": 6},
}
PROMPT_DIR = Path("prompts")
RESULTS_DIR = Path("results")


class SentimentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentiment: Literal["negative", "neutral", "positive"]


def make_client(provider: str) -> OpenAI:
    env = PROVIDERS[provider]["api_key_env"]
    api_key = os.environ[env] if env else "ollama"
    return OpenAI(base_url=PROVIDERS[provider]["base_url"], api_key=api_key, max_retries=5)


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def call_llm(client, provider: str, system_prompt: str, fields: dict, schema: type[BaseModel]) -> dict:
    def strip_tags(value: str) -> str:
        for tag in fields:
            value = value.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
        return value

    cfg = PROVIDERS[provider]
    user_content = "\n".join(f"<{tag}>{strip_tags(value)}</{tag}>" for tag, value in fields.items())

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=cfg["model"],
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True},
        },
        **cfg["extra"],
    )
    latency_ms = (time.perf_counter() - start) * 1000

    raw = response.choices[0].message.content or ""
    try:
        parsed = schema.model_validate_json(raw)
    except ValidationError:
        parsed = None
    return {
        "parsed": parsed,
        "raw": raw,
        "latency_ms": latency_ms,
        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
        "completion_tokens": response.usage.completion_tokens if response.usage else None,
    }


def classify(client, provider: str, system_prompt: str, text: str) -> dict:
    result = call_llm(client, provider, system_prompt, {"tweet": text}, SentimentOutput)
    parsed = result.pop("parsed")
    return {"pred": parsed.sentiment if parsed else None, **result}


def run_name(provider: str, prompt_name: str) -> str:
    return f"llm_{PROVIDERS[provider]['model'].replace(':', '-').replace('/', '-')}_{prompt_name}"


def run_resumable(rows: pd.DataFrame, out_csv: Path, predict, provider: str, limit: int | None = None):
    RESULTS_DIR.mkdir(exist_ok=True)
    done = set(pd.read_csv(out_csv)["id"]) if out_csv.exists() else set()
    todo = rows[~rows["id"].isin(done)]
    if limit is not None:
        todo = todo.head(limit)

    if len(todo):
        predict(next(todo.itertuples())) 
    for i, row in enumerate(todo.itertuples(), 1):
        pd.DataFrame([{"id": row.id, **predict(row)}]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        print(f"\r{len(done) + i}/{len(rows)}", end="", flush=True)
        time.sleep(PROVIDERS[provider].get("pause_s", 0)) 
    print()

    preds = pd.read_csv(out_csv)
    if len(preds) < len(rows):
        print(f"{len(preds)}/{len(rows)} done; run again to continue.")
        return None
    return preds.set_index("id").loc[rows["id"]]


def llm_metrics(preds: pd.DataFrame) -> dict:
    return {
        "test_size": len(preds),
        **summarize(preds["sentiment"], preds["pred"].fillna("invalid")),
        "invalid_rate": float(preds["pred"].isna().mean()),
        "latency_ms": float(preds["latency_ms"].mean()),
        "latency_ms_median": float(preds["latency_ms"].median()), 
        "prompt_tokens_mean": float(preds["prompt_tokens"].mean()),
        "completion_tokens_mean": float(preds["completion_tokens"].mean()),
    }


def evaluate(provider: str, prompt_name: str, limit: int | None = None, client=None) -> dict | None:
    client = client or make_client(provider)
    system_prompt = load_prompt(prompt_name)
    name = run_name(provider, prompt_name)

    preds = run_resumable(
        load_test_data(), RESULTS_DIR / f"{name}_predictions.csv",
        lambda row: {"sentiment": row.sentiment, **classify(client, provider, system_prompt, row.text)},
        provider, limit,
    )
    if preds is None:
        return None

    metrics = {"provider": provider, "model": PROVIDERS[provider]["model"], "prompt": prompt_name, **llm_metrics(preds)}
    (RESULTS_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--prompt", required=True, help="file name in prompts/ without .txt")
    parser.add_argument("--limit", type=int, help="max new predictions in this run")
    args = parser.parse_args()

    metrics = evaluate(args.provider, args.prompt, args.limit)
    if metrics:
        print(json.dumps(metrics, indent=2))
