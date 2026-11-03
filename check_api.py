"""
APP_API_KEY=... python check_api.py [api-link]
"""
import os
import sys
import requests

url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
headers = {"X-API-Key": os.environ["APP_API_KEY"]}
sentences = ["I love this phone, the camera is amazing!", "Worst service ever, never again."]
expected = ["positive", "negative"]

assert requests.get(f"{url}/health", timeout=60).json()["status"] == "ok"
assert requests.post(f"{url}/predict", json={"model": "linearsvc", "sentences": sentences}, timeout=60).status_code == 401, \
    "requests without the key must be refused"

for model in ["linearsvc", "linearsvc_rules", "distilroberta", "llm"]:
    res = requests.post(f"{url}/predict", json={"model": model, "sentences": sentences}, headers=headers, timeout=120)
    assert res.ok, f"{model}: {res.status_code} {res.text}"
    labels = [r["sentiment"] for r in res.json()["results"]]
    print(f"{model:16s} {labels}  {res.json()['latency_ms']:>8.1f} ms  {'OK' if labels == expected else 'CHECK'}")

res = requests.post(f"{url}/absa", json={"text": "The food was great but the service was slow."}, headers=headers, timeout=120)
assert res.ok, f"absa: {res.status_code} {res.text}"
print("absa            ", res.json()["aspects"])
