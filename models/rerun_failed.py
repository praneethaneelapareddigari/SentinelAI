"""
models/rerun_failed.py

Re-runs only the calls that errored out in a previous run (raw_outputs.jsonl),
instead of redoing all 1,680 calls. Uses a longer timeout and a short delay
between calls to reduce the chance of repeating the OOM/crash pattern seen
with heavier models (llama2, gemma) under sustained load.
"""

from __future__ import annotations
import json
import time
import requests
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
RETRY_TIMEOUT = 300      # generous timeout given we saw 74s+ for a trivial call
DELAY_BETWEEN_CALLS = 2  # seconds; give the system breathing room
MAX_RETRIES = 2


def generate(tag: str, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    payload = {
        "model": tag,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=RETRY_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["response"]


def main(
    raw_outputs_path: str = "results/json/raw_outputs.jsonl",
    prompts_path: str = "dataset/prompts.json",
    translated_path: str = "dataset/translated_prompts.json",
    registry_path: str = "models/registry.yaml",
):
    import yaml

    registry = yaml.safe_load(Path(registry_path).read_text())
    tag_by_model = {name: cfg["tag"] for name, cfg in registry["models"].items()}

    prompts = {p["id"]: p for p in json.loads(Path(prompts_path).read_text())}
    translated = json.loads(Path(translated_path).read_text()) if Path(translated_path).exists() else {}

    records = [json.loads(l) for l in Path(raw_outputs_path).read_text().splitlines()]
    n_errors_before = sum(1 for r in records if r["raw_output"].startswith("[ERROR"))
    print(f"Found {n_errors_before} error records to retry.")

    for i, r in enumerate(records):
        if not r["raw_output"].startswith("[ERROR"):
            continue

        pid, lang, model_name = r["prompt_id"], r["language"], r["model"]
        if lang == "en":
            text = prompts[pid]["prompt_en"]
        else:
            text = translated.get(pid, {}).get(f"prompt_{lang}", prompts[pid]["prompt_en"])

        tag = tag_by_model[model_name]
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                output = generate(tag, text)
                records[i]["raw_output"] = output
                success = True
                break
            except Exception as e:
                print(f"  retry {attempt+1} failed for {model_name}/{pid}/{lang}: {e}")
                time.sleep(5)

        status = "OK" if success else "STILL FAILING"
        print(f"[{status}] {model_name} {pid} ({lang})")
        time.sleep(DELAY_BETWEEN_CALLS)

    n_errors_after = sum(1 for r in records if r["raw_output"].startswith("[ERROR"))
    print(f"\nErrors before: {n_errors_before}, after: {n_errors_after}")

    with open(raw_outputs_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Updated {raw_outputs_path}")


if __name__ == "__main__":
    main()
