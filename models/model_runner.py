"""
models/model_runner.py

Unified interface for running the same prompt against multiple locally-served
models via Ollama (https://ollama.com). Ollama is the default because it makes
"run 4 open models on 5 languages x 400 prompts" tractable on a single machine
without juggling four different inference stacks.

Swap `OllamaModel` for a `HFModel` / `VLLMModel` class with the same
`.generate()` signature if you'd rather run via transformers or vLLM.
"""

from __future__ import annotations
import json
import time
import requests
import yaml
from pathlib import Path
from dataclasses import dataclass


OLLAMA_HOST = "http://localhost:11434"


@dataclass
class GenerationResult:
    model: str
    prompt_id: str
    language: str
    run_index: int
    raw_output: str
    latency_s: float


class OllamaModel:
    def __init__(self, name: str, tag: str):
        self.name = name
        self.tag = tag

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        payload = {
            "model": self.tag,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["response"]


def load_registry(path: str = "models/registry.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def run_all(
    prompts_path: str = "dataset/prompts.json",
    translated_path: str = "dataset/translated_prompts.json",
    registry_path: str = "models/registry.yaml",
    out_path: str = "results/json/raw_outputs.jsonl",
):
    registry = load_registry(registry_path)
    decoding = registry["decoding"]
    prompts = json.loads(Path(prompts_path).read_text())
    translated = json.loads(Path(translated_path).read_text()) if Path(translated_path).exists() else {}

    models = {
        name: OllamaModel(name, cfg["tag"])
        for name, cfg in registry["models"].items()
        if cfg["backend"] == "ollama"
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # Model is the OUTERMOST loop deliberately: each Ollama model swap has real
        # cost (unload + reload, worse for larger models under tight memory budgets).
        # Looping prompt->language->model->run would force a swap on almost every
        # call; looping model->prompt->language->run swaps only once per model.
        for model_name, model in models.items():
            print(f"=== Loading/using model: {model_name} ===")
            for item in prompts:
                pid = item["id"]
                lang_variants = {"en": item["prompt_en"]}
                if pid in translated:
                    for lang in ["hi", "zh", "ta", "ms"]:
                        key = f"prompt_{lang}"
                        if key in translated[pid]:
                            lang_variants[lang] = translated[pid][key]

                for lang, text in lang_variants.items():
                    for run_idx in range(decoding["runs_per_item"]):
                        start = time.time()
                        try:
                            output = model.generate(
                                text,
                                temperature=decoding["temperature"],
                                max_tokens=decoding["max_tokens"],
                            )
                        except Exception as e:
                            output = f"[ERROR: {e}]"
                        result = GenerationResult(
                            model=model_name,
                            prompt_id=pid,
                            language=lang,
                            run_index=run_idx,
                            raw_output=output,
                            latency_s=round(time.time() - start, 3),
                        )
                        f.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
                        print(f"[{model_name}] {pid} ({lang}) run {run_idx} done")


if __name__ == "__main__":
    run_all()
