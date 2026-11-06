import html
import io
import json
import os
import re
from pathlib import Path
import docx
import pandas as pd
import requests
import streamlit as st
from pypdf import PdfReader

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("APP_API_KEY", "")
RESULTS = Path(__file__).resolve().parent.parent / "results"

LLM_MAX_SENTENCES = 10  # free LLM quota
MAX_SENTENCES = 100
MAX_CHARS = 1000

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

PALETTE = {
    "light": {"bg": "#fffaf1", "fg": "#1c1c1c", "accent": "#344f88", "accent2": "#66914a", "gold": "#b5803f"},
    "dark": {"bg": "#1c1c1c", "fg": "#fffaf1", "accent": "#87c0c8", "accent2": "#9cc7b8", "gold": "#d1b06f"},
}
theme = "light" if st.query_params.get("theme") == "light" else "dark"
c = PALETTE[theme]
SENTIMENT_COLOR = {"positive": c["accent2"], "negative": c["gold"], "neutral": c["fg"]}

st.markdown(
    f"""
    <style>
    :root {{
        --bg: {c["bg"]}; --fg: {c["fg"]}; --accent: {c["accent"]}; --accent-2: {c["accent2"]}; --gold: {c["gold"]};
        --line: color-mix(in srgb, var(--fg) 18%, transparent);
        --field: color-mix(in srgb, var(--fg) 5%, var(--bg));
    }}
    .stApp {{ background: var(--bg); color: var(--fg); font-family: Helvetica, Arial, sans-serif; }}
    header[data-testid="stHeader"], div[data-testid="InputInstructions"] {{ display: none !important; }}
    .stApp p, .stApp label, .stApp li, .stApp summary, .stApp span {{ color: var(--fg); }}
    h1, h2, h3 {{ font-family: Helvetica, Arial, sans-serif !important; color: var(--fg) !important; font-weight: 400 !important; }}
    .eyebrow {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }}
    .muted {{ opacity: .65; font-size: .85rem; }}

    [data-testid="stExpander"] details {{ border: 1px solid var(--line) !important; border-radius: 12px; background: transparent; }}
    [data-baseweb="textarea"], [data-baseweb="textarea"] textarea, [data-baseweb="select"] > div,
    [data-testid="stFileUploaderDropzone"] {{
        background: var(--field) !important; color: var(--fg) !important; border-color: var(--line) !important;
    }}
    [data-baseweb="textarea"]:focus-within, [data-baseweb="select"] > div:focus-within {{ border-color: var(--accent) !important; }}
    [data-baseweb="textarea"] textarea::placeholder {{ color: var(--fg); opacity: .4; }}
    [data-baseweb="popover"] ul, [data-baseweb="popover"] li {{ background: var(--bg) !important; color: var(--fg) !important; }}
    [data-baseweb="popover"] li:hover {{ background: var(--field) !important; }}
    [data-baseweb="select"] svg {{ fill: var(--fg); }}

    [data-baseweb="tab-list"] {{ gap: 1.5rem; }}
    [data-baseweb="tab-list"] button p {{ color: var(--fg); opacity: .7; }}
    [data-baseweb="tab-list"] button[aria-selected="true"] p {{ color: var(--accent); opacity: 1; }}
    [data-baseweb="tab-highlight"] {{ background: var(--accent) !important; }}
    [data-baseweb="tab-border"] {{ background: var(--line) !important; }}

    .stButton button, [data-testid="stFileUploaderDropzone"] button {{
        background: transparent !important; color: var(--accent) !important; border: 1px solid var(--accent) !important; border-radius: 999px;
    }}
    .stButton button:hover {{ color: var(--gold) !important; border-color: var(--gold) !important; }}
    .stButton button p {{ color: inherit !important; }}
    [data-baseweb="radio"]:has(input:checked) > div:first-child {{ background: var(--gold) !important; }}

    [data-testid="stAlert"] > div {{ background: color-mix(in srgb, var(--gold) 14%, var(--bg)) !important; border: 1px solid var(--gold); border-radius: 12px; }}
    [data-testid="stAlert"] p {{ color: var(--fg) !important; }}
    [data-testid="stMetricValue"] {{ color: var(--accent); font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}

    .tbl {{ width: 100%; border-collapse: collapse; font-size: .9rem; margin: .5rem 0 1rem; }}
    .tbl th {{ text-align: left; font-weight: 400; opacity: .65; border-bottom: 1px solid var(--line); padding: .4rem .5rem; }}
    .tbl td {{ border-bottom: 1px solid var(--line); padding: .45rem .5rem; vertical-align: top; color: var(--fg); }}
    .tbl td.num {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; white-space: nowrap; }}
    .tbl tr.hl td {{ color: var(--accent); }}
    .chip {{ display: inline-block; padding: .3rem .8rem; margin: .25rem .35rem .25rem 0; border-radius: 999px; border: 1px solid; font-size: .9rem; }}
    .chip small {{ opacity: .75; margin-left: .4rem; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
    .label {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .85rem; }}
    code {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; color: var(--gold) !important; background: var(--field) !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# RESULTS 
def load_metrics(name: str) -> dict | None:
    path = RESULTS / f"{name}_metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


SENTIMENT_ROWS = [
    ("linearsvc", "LinearSVC", "CPU"),
    ("linearsvc_rules", "LinearSVC + rules", "CPU"),
    ("llm_qwen2.5-7b_sentiment_zero_shot_v1", "qwen2.5:7b, zero-shot", "laptop GPU"),
    ("llm_qwen2.5-7b_sentiment_few_shot_v1", "qwen2.5:7b, few-shot", "laptop GPU"),
    ("llm_openai-gpt-oss-20b_sentiment_zero_shot_v1", "gpt-oss-20b, zero-shot", "Groq free API"),
    ("llm_openai-gpt-oss-20b_sentiment_few_shot_v1", "gpt-oss-20b, few-shot", "Groq free API"),
    ("distilroberta", "DistilRoBERTa, fine-tuned", "CPU, 2 threads"),
    ("distilroberta_onnx_int8", "DistilRoBERTa, ONNX int8", "CPU, 1 thread"),
]

DEMO_MODELS = {
    "LinearSVC": ("linearsvc", "linearsvc"),
    "LinearSVC + rules": ("linearsvc_rules", "linearsvc_rules"),
    "DistilRoBERTa (fine-tuned)": ("distilroberta", "distilroberta_onnx_int8"),
    "LLM · gpt-oss-20b via Groq": ("llm", "llm_openai-gpt-oss-20b_sentiment_zero_shot_v1"),
}


def latency(m: dict) -> float:
    return m.get("latency_ms_median", m["latency_ms"])


def html_table(columns: list[str], rows: list[list], numeric: set[int] = frozenset(), highlight=()) -> str:
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = "".join(
        f'<tr class="{"hl" if i in highlight else ""}">'
        + "".join(f'<td class="{"num" if j in numeric else ""}">{html.escape(str(v))}</td>' for j, v in enumerate(row))
        + "</tr>"
        for i, row in enumerate(rows)
    )
    return f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


@st.cache_data(show_spinner=False, max_entries=20)
def extract_text_from_file(name: str, data: bytes) -> str:
    name = name.lower()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
        return re.sub(r"\s+", " ", text).strip()
    if name.endswith(".docx"):
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    st.error("Unsupported file type. Please upload TXT, DOCX, or PDF.")
    return ""


def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


class ApiError(Exception):
    pass


@st.cache_data(ttl=24 * 3600, show_spinner=False) 
def call_api(path: str, payload: str) -> dict:
    try:
        res = requests.post(f"{API_URL}{path}", data=payload, timeout=90,
                            headers={"Content-Type": "application/json", "X-API-Key": API_KEY})
    except requests.RequestException:
        raise ApiError("The API is not responding. It may be waking up, so please try again in a minute.")
    if res.status_code in (422, 429, 503):
        try:
            detail = res.json()["detail"]
        except (ValueError, KeyError):
            detail = "The API is busy. Please try again in a minute."
        raise ApiError(detail)
    if not res.ok:
        raise ApiError(f"The API returned an error ({res.status_code}). Please try again later.")
    return res.json()


st.markdown('<div class="eyebrow">Live demo · LinearSVC · DistilRoBERTa · LLM</div>', unsafe_allow_html=True)
st.title("Sentiment Analysis")
st.markdown("Label each sentence as positive, neutral or negative with the model of your choice, "
            "or break a review into aspects with their own sentiment.")

tab_sentiment, tab_aspects, tab_results = st.tabs(["Sentiment", "Aspects", "Results"])

with tab_sentiment:
    choice = st.selectbox("Model", list(DEMO_MODELS))
    model_id, metrics_name = DEMO_MODELS[choice]
    m = load_metrics(metrics_name)
    if m:
        st.markdown(f'<div class="muted">F1 macro {m["f1_macro"]:.3f} on the 1,000-tweet test set · '
                    f'~{latency(m):,.0f} ms per sentence</div>', unsafe_allow_html=True)
    if model_id == "llm":
        st.markdown(f'<div class="muted">Uses a free API shared by every visitor: only the first '
                    f'{LLM_MAX_SENTENCES} sentences are analyzed.</div>', unsafe_allow_html=True)

    source = st.radio("Input", ["Paste text", "Upload TXT / DOCX / PDF"], horizontal=True, label_visibility="collapsed")
    if source == "Paste text":
        text = st.text_area("Text", height=180, label_visibility="collapsed",
                            placeholder="Paste a paragraph. Sentences are split at . ! ?")
    else:
        uploaded = st.file_uploader("File", type=["txt", "docx", "pdf"], label_visibility="collapsed")
        text = extract_text_from_file(uploaded.name, uploaded.getvalue()) if uploaded else ""
        if text:
            with st.expander("Preview extracted text"):
                st.text(text[:3000])

    if st.button("Analyze", key="analyze"):
        all_sentences = split_into_sentences(text)
        sentences = [s for s in all_sentences if len(s) <= MAX_CHARS]
        limit = LLM_MAX_SENTENCES if model_id == "llm" else MAX_SENTENCES
        if len(sentences) < len(all_sentences):
            st.info(f"{len(all_sentences) - len(sentences)} sentence(s) longer than {MAX_CHARS:,} characters were skipped.")
        if not sentences:
            st.warning("Please paste or upload some text first.")
        else:
            if len(sentences) > limit:
                st.info(f"Only the first {limit} of {len(sentences)} sentences are analyzed with this model.")
                sentences = sentences[:limit]
            try:
                with st.spinner("Asking the model…"):
                    out = call_api("/predict", json.dumps({"model": model_id, "sentences": sentences}))
            except ApiError as e:
                st.warning(str(e))
            else:
                results = out["results"]
                counts = pd.Series([r["sentiment"] for r in results]).value_counts()
                cols = st.columns(3)
                for col, label in zip(cols, ["positive", "neutral", "negative"]):
                    col.metric(label.capitalize(), int(counts.get(label, 0)))
                show_notes = any(r["note"] for r in results)
                rows = [[i, r["sentence"], r["sentiment"] or "—"] + ([r["note"]] if show_notes else [])
                        for i, r in enumerate(results, 1)]
                st.markdown(html_table(["#", "Sentence", "Sentiment"] + (["Note"] if show_notes else []), rows,
                                       numeric={0, 2}), unsafe_allow_html=True)

with tab_aspects:
    st.markdown("One sentence can be positive about one thing and negative about another. "
                "The LLM lists each aspect with its own sentiment.")
    review = st.text_area("Review", height=120, max_chars=MAX_CHARS, label_visibility="collapsed",
                          placeholder="The food was great but the service was slow.")
    if st.button("Find aspects", key="absa"):
        if not review.strip():
            st.warning("Please write a review first.")
        else:
            try:
                with st.spinner("Asking the LLM…"):
                    out = call_api("/absa", json.dumps({"text": review.strip()}))
            except ApiError as e:
                st.warning(str(e))
            else:
                if not out["valid"]:
                    st.warning("The model returned an invalid answer. Please try again with different wording.")
                elif not out["aspects"]:
                    st.info("No aspects found in this text.")
                else:
                    chips = "".join(
                        f'<span class="chip" style="border-color:{SENTIMENT_COLOR[a["sentiment"]]};'
                        f'color:{SENTIMENT_COLOR[a["sentiment"]]}">{html.escape(a["aspect"])}'
                        f'<small>{a["sentiment"]}</small></span>'
                        for a in out["aspects"]
                    )
                    st.markdown(chips, unsafe_allow_html=True)
    st.markdown('<div class="muted">Runs gpt-oss-20b via Groq. The Results tab shows SemEval-2014 Restaurants scores '
                'for every model evaluated on this task.</div>', unsafe_allow_html=True)

with tab_results:
    st.markdown("### Tweet sentiment")
    st.markdown('<div class="muted">Every model is scored on the same 1,000 tweets from the tweet_eval test split. '
                'F1 macro is the headline metric.</div>', unsafe_allow_html=True)
    rows, highlight = [], set()
    deployed = {name for _, name in DEMO_MODELS.values()}
    for name, label, hardware in SENTIMENT_ROWS:
        if r := load_metrics(name):
            if name in deployed:
                highlight.add(len(rows))
            rows.append([label, f'{r["f1_macro"]:.3f}', f'{r["accuracy"]:.3f}', f"{latency(r):,.1f} ms", hardware])
    st.markdown(html_table(["Model", "F1 macro", "Accuracy", "Latency (median)", "Hardware"], rows,
                           numeric={1, 2, 3}, highlight=highlight), unsafe_allow_html=True)

    st.markdown('<div class="muted">Latencies come from different hardware and are not like-for-like.</div>', unsafe_allow_html=True)

    st.markdown("### Aspect-based sentiment")
    st.markdown('<div class="muted">SemEval-2014 Restaurants, 1,120 aspects in 600 sentences. Baselines: always '
                '"positive" F1 0.263; sentence-level LinearSVC F1 0.549.</div>', unsafe_allow_html=True)
    absa_rows = []
    for model_name, label in [("qwen2.5-7b", "qwen2.5:7b"), ("openai-gpt-oss-20b", "gpt-oss-20b")]:
        if p := load_metrics(f"llm_{model_name}_absa_polarity_v1"):
            absa_rows.append([label, "Polarity of a given aspect", f'F1 {p["f1_macro"]:.3f}', f'accuracy {p["accuracy"]:.3f}'])
        if e := load_metrics(f"llm_{model_name}_absa_extract_v1"):
            absa_rows.append([label, "Find aspects + polarity", f'recall {e["aspect_recall"]:.3f}',
                              f'end-to-end {e["end_to_end_accuracy"]:.3f}'])
    st.markdown(html_table(["Model", "Task", "Score", ""], absa_rows, numeric={2, 3}), unsafe_allow_html=True)
