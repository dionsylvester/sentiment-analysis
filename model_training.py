import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from sentiment_rules import rule_based_training_examples, clean_for_rules, apply_sentiment_rules

LABELS = ["negative", "neutral", "positive"]

TRAIN_PATH = "dataset/sentiment_train.csv"
OFFICIAL_TEST_PATH = "dataset/sentiment_test.csv"
TEST_PATH = "dataset/sentiment_test_1000.csv" 
TEST_SIZE = 1000


def _new_pipeline():
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    model = LinearSVC(C=0.3, class_weight='balanced', random_state=42, max_iter=5000)
    return vectorizer, model


def _load_tweet_eval(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sentiment"] = df["label"].map(dict(enumerate(LABELS)))
    return df


def make_fixed_test_set() -> pd.DataFrame:
    df = _load_tweet_eval(OFFICIAL_TEST_PATH).reset_index(names="id")
    test, _ = train_test_split(df, train_size=TEST_SIZE, stratify=df["sentiment"], random_state=42)
    test = test.sort_values("id")[["id", "text", "label", "sentiment"]]
    test.to_csv(TEST_PATH, index=False)
    return test


def load_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH)


def load_training_data() -> pd.DataFrame:
    df_train = _load_tweet_eval(TRAIN_PATH)[["text", "sentiment"]]
    df_final = pd.concat([df_train, rule_based_training_examples()], ignore_index=True)
    df_final['text_clean'] = df_final['text'].apply(clean_for_rules)
    return df_final[df_final['text_clean'].str.len() > 2]


def summarize(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average='macro', zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall, 
        "f1_macro": f1,
    }


def predict(model, vectorizer, text: str, use_rules: bool = False) -> str:
    raw = model.predict(vectorizer.transform([clean_for_rules(text)]))[0]
    return apply_sentiment_rules(text, raw)[0] if use_rules else raw


def evaluate_model(model, vectorizer, use_rules: bool = False) -> dict:
    test = load_test_data()

    start = time.perf_counter()
    y_pred = [predict(model, vectorizer, t, use_rules) for t in test["text"]]
    latency_ms = (time.perf_counter() - start) / len(test) * 1000

    report = classification_report(test["sentiment"], y_pred, labels=LABELS, output_dict=True, zero_division=0)
    report.pop('accuracy', None)
    cm = confusion_matrix(test["sentiment"], y_pred, labels=LABELS)

    return {
        "test_size": len(test),
        **summarize(test["sentiment"], y_pred),
        "latency_ms": latency_ms,
        "predictions": y_pred,
        "report": pd.DataFrame(report).transpose(),
        "confusion_matrix": pd.DataFrame(
            cm,
            index=[f"actual_{l}" for l in LABELS],
            columns=[f"pred_{l}" for l in LABELS],
        ),
    }


def train_model():
    df_final = load_training_data()

    vectorizer, model = _new_pipeline()
    model.fit(vectorizer.fit_transform(df_final['text_clean']), df_final['sentiment'])

    return model, vectorizer, len(df_final), evaluate_model(model, vectorizer)


MODEL_PATH = "models/linearsvc.joblib"


def export_model():
    import json
    import os

    import joblib
    import sklearn

    model, vectorizer, data_count, metrics = train_model()
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "vectorizer": vectorizer, "sklearn_version": sklearn.__version__}, MODEL_PATH, compress=3)

    os.makedirs("results", exist_ok=True)
    for name, m in [("linearsvc", metrics), ("linearsvc_rules", evaluate_model(model, vectorizer, use_rules=True))]:
        summary = {k: float(v) if k != "test_size" else v for k, v in m.items()
                   if k in ("test_size", "accuracy", "precision_macro", "recall_macro", "f1_macro", "latency_ms")}
        with open(f"results/{name}_metrics.json", "w") as f:
            json.dump({"model": name, "train_size": data_count, **summary}, f, indent=2)
    return metrics


if __name__ == "__main__":
    m = export_model()
    print(f"Saved {MODEL_PATH}; test F1 macro {m['f1_macro']:.4f}")
