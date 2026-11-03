import os
from functools import lru_cache
import joblib
import numpy as np
from openai import APIError, OpenAI, RateLimitError

from absa import extract_aspects
from llm_sentiment import PROVIDERS, classify, load_prompt
from model_training import LABELS, MODEL_PATH
from sentiment_rules import apply_sentiment_rules, clean_for_rules

MODELS = ("linearsvc", "linearsvc_rules", "distilroberta", "llm")
LLM_PROVIDER = "groq"
SENTIMENT_PROMPT = "sentiment_zero_shot_v1"
ABSA_PROMPT = "absa_extract_v1"

MAX_SENTENCES = 100
LLM_MAX_SENTENCES = 10
MAX_CHARS = 1000

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")
ONNX_FILE = "onnx/model.int8.onnx"
TOKENIZER_FILE = "onnx/tokenizer.json"
MAX_LENGTH = 128 


RATE_LIMITED = "The LLM is rate-limited. Try again in a minute, or use another model."
LLM_DOWN = "The LLM is not responding. Try again later, or use another model."


class QuotaExceeded(Exception):
    """The LLM provider's rate limit or daily quota is used up."""


class ModelUnavailable(Exception):
    """A model is not configured on this deployment."""


def _validate(texts: list[str], limit: int):
    if not texts:
        raise ValueError("Send at least one sentence.")
    if len(texts) > limit:
        raise ValueError(f"At most {limit} sentences per request for this model.")
    if any(not t.strip() for t in texts):
        raise ValueError("Sentences must not be empty.")
    if any(len(t) > MAX_CHARS for t in texts):
        raise ValueError(f"Each sentence must be at most {MAX_CHARS} characters.")


@lru_cache(maxsize=1)
def _svc():
    bundle = joblib.load(MODEL_PATH)
    return bundle["vectorizer"], bundle["model"]


def _load_onnx(tokenizer_path: str, model_path: str):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(MAX_LENGTH)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1 
    return tokenizer, ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _distilroberta():
    if not HF_MODEL_REPO:
        raise ModelUnavailable("DistilRoBERTa is not configured (HF_MODEL_REPO is not set).")
    from huggingface_hub import hf_hub_download

    cache = "/tmp/hf" 
    return _load_onnx(hf_hub_download(HF_MODEL_REPO, TOKENIZER_FILE, cache_dir=cache),
                      hf_hub_download(HF_MODEL_REPO, ONNX_FILE, cache_dir=cache))


@lru_cache(maxsize=1)
def _llm_client():
    cfg = PROVIDERS[LLM_PROVIDER]
    key = os.getenv(cfg["api_key_env"])
    if not key:
        raise ModelUnavailable(f"The LLM is not configured ({cfg['api_key_env']} is not set).")
    return OpenAI(base_url=cfg["base_url"], api_key=key, max_retries=1, timeout=10)


@lru_cache(maxsize=2048)
def _llm_label(sentence: str) -> str | None:
    try:
        return classify(_llm_client(), LLM_PROVIDER, load_prompt(SENTIMENT_PROMPT), sentence)["pred"]
    except RateLimitError as e:
        raise QuotaExceeded(RATE_LIMITED) from e
    except APIError as e:
        raise ModelUnavailable(LLM_DOWN) from e


def predict(sentences: list[str], model: str) -> list[dict]:
    """One {"sentence", "sentiment", "note"} per sentence; sentiment is None if the LLM output was invalid."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}. Choose one of: {', '.join(MODELS)}.")
    _validate(sentences, LLM_MAX_SENTENCES if model == "llm" else MAX_SENTENCES)

    if model in ("linearsvc", "linearsvc_rules"):
        vectorizer, svc = _svc()
        cleaned = [clean_for_rules(s) for s in sentences]
        raw = svc.predict(vectorizer.transform(cleaned))
        results = []
        for sentence, text, label in zip(sentences, cleaned, raw):
            note = ""
            if not text:
                label, note = "neutral", "Empty after cleaning."
            elif model == "linearsvc_rules":
                label, reasons = apply_sentiment_rules(sentence, label)
                note = "; ".join(reasons)
            results.append({"sentence": sentence, "sentiment": str(label), "note": note})
        return results

    if model == "distilroberta":
        tokenizer, session = _distilroberta()
        encodings = tokenizer.encode_batch(sentences)
        logits = session.run(None, {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        })[0]
        return [{"sentence": s, "sentiment": LABELS[i], "note": ""} for s, i in zip(sentences, logits.argmax(axis=1))]

    results = []
    for sentence in sentences:
        label = _llm_label(sentence)
        results.append({"sentence": sentence, "sentiment": label, "note": "" if label else "Invalid model output."})
    return results


@lru_cache(maxsize=512)
def _absa(text: str):
    try:
        aspects = extract_aspects(_llm_client(), LLM_PROVIDER, load_prompt(ABSA_PROMPT), text)["aspects"]
    except RateLimitError as e:
        raise QuotaExceeded(RATE_LIMITED) from e
    except APIError as e:
        raise ModelUnavailable(LLM_DOWN) from e
    return tuple(tuple(a.items()) for a in aspects) if aspects is not None else None 


def absa(text: str) -> list[dict] | None:
    _validate([text], 1)
    aspects = _absa(text.strip())
    return [dict(a) for a in aspects] if aspects is not None else None
