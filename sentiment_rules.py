import re
import pandas as pd
from typing import Optional, Tuple, List

NEUTRAL_PHRASES = {
    "ok",
    "okay",
    "acceptable",
    "average",
    "fair",
    "mediocre",
    "so so",
    "not bad not good",
    "neither good nor bad",
}

NEGATIVE_KEYWORDS = {
    "poison", "toxic", "toxicity", "scam", "fraud", "malware", "virus",
    "broken", "damage", "damaged", "dangerous", "unsafe", "disaster",
    "terrible", "awful", "bad", "worst", "hate", "pain", "problem",
    "bug", "bugs", "error", "errors", "failed", "failure", "crash",
    "crashes", "slow", "poor", "useless", "waste", "horrible"
}

POSITIVE_ADJECTIVES = {
    "good", "great", "nice", "excellent", "amazing", "perfect",
    "wonderful", "best", "beautiful", "love", "lovely"
}

NEGATION_TERMS = {
    "no", "not", "never", "hardly", "barely", "rarely", "without", "zero"
}


def clean_for_rules(text: str) -> str:
    text = str(text).lower().replace("’", "'")

    contractions = {
        "can't": "can not",
        "cannot": "can not",
        "won't": "will not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "couldn't": "could not",
    }

    for key, value in contractions.items():
        text = text.replace(key, value)

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def detect_neutral_phrase(cleaned_text: str) -> Optional[str]:
    padded = f" {cleaned_text} "
    for phrase in NEUTRAL_PHRASES:
        if f" {phrase} " in padded:
            rest = padded.replace(f" {phrase} ", " ").split()
            if any(w in POSITIVE_ADJECTIVES or w in NEGATIVE_KEYWORDS for w in rest):
                continue 
            return f"Detected neutral phrase: '{phrase}'"
    return None


def detect_negative_phrase(cleaned_text: str) -> Optional[str]:
    words = cleaned_text.split()

    for i, word in enumerate(words):
        if word in POSITIVE_ADJECTIVES:
            window = words[i + 1:i + 4]
            for candidate in window:
                if candidate in NEGATION_TERMS:
                    break
                if candidate in NEGATIVE_KEYWORDS:
                    return f"Positive adjective '{word}' appears near negative word '{candidate}'"
    return None


def detect_negation_flip(cleaned_text: str) -> Optional[str]:
    words = cleaned_text.split()

    for i, word in enumerate(words):
        if word in NEGATION_TERMS and words[i + 1:i + 2] != ["only"]:
            window = words[i + 1:i + 4]
            if any(w in NEGATIVE_KEYWORDS for w in window):
                window = window[:next(k for k, w in enumerate(window) if w in NEGATIVE_KEYWORDS)]

            if any(w in POSITIVE_ADJECTIVES for w in window):
                return "Detected negation before positive word"

            if "recommend" in window or "recommended" in window:
                return "Detected negation before recommendation"
    return None


def apply_sentiment_rules(original_text: str, model_prediction: str) -> Tuple[str, List[str]]:
    cleaned = clean_for_rules(original_text)
    reasons = []

    neutral_reason = detect_neutral_phrase(cleaned)
    if neutral_reason:
        reasons.append(neutral_reason)
        return "neutral", reasons

    negative_phrase_reason = detect_negative_phrase(cleaned)
    if negative_phrase_reason:
        reasons.append(negative_phrase_reason)
        return "negative", reasons

    negation_reason = detect_negation_flip(cleaned)
    if negation_reason:
        reasons.append(negation_reason)
        return "negative", reasons

    return model_prediction, reasons


def rule_based_training_examples() -> pd.DataFrame:
    examples = [
        ("good poison", "negative"),
        ("this is good poison", "negative"),
        ("what a nice scam", "negative"),
        ("great fraud service", "negative"),
        ("excellent malware", "negative"),
        ("good virus", "negative"),
        ("nice toxic product", "negative"),
        ("great disaster", "negative"),
        ("good damage", "negative"),
        ("not good", "negative"),
        ("not recommended", "negative"),
        ("not a good experience", "negative"),
        ("never good", "negative"),
        ("no good result", "negative"),
        ("the app is not bad", "positive"),
        ("not terrible at all", "positive"),
        ("no problem", "positive"),
        ("works without issues", "positive"),
    ]

    return pd.DataFrame(examples, columns=["text", "sentiment"])
