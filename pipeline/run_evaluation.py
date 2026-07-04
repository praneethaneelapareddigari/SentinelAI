"""
pipeline/run_evaluation.py

End-to-end orchestrator, following the protocol in docs/methodology.md §5:

  1. Translate dataset (skips if translated_prompts.json already exists)
  2. Run all models x languages x prompts (via Ollama)
  3. Score refusal / over-refusal / consistency
  4. Generate figures

Usage:
    python pipeline/run_evaluation.py --skip-translate --skip-run
    (useful for re-scoring/re-plotting without re-running expensive steps)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Ensure the project root (parent of this file's directory) is on sys.path so
# `python pipeline/run_evaluation.py` works the same as `python -m pipeline.run_evaluation`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from translation.translator import translate_dataset
from models.model_runner import run_all
from evaluation.scoring import load_records, save_summary
from visualization.plots import generate_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--backend", default="nllb", choices=["nllb", "argos"])
    parser.add_argument("--prompts", default="dataset/prompts.json")
    parser.add_argument("--translated", default="dataset/translated_prompts.json")
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    translated_path = Path(args.translated)

    if not args.skip_translate and not translated_path.exists():
        print("== Step 1: Translating dataset ==")
        translate_dataset(prompts_path, translated_path, backend=args.backend)
    else:
        print("== Step 1: Skipped (translated_prompts.json exists or --skip-translate) ==")

    if not args.skip_run:
        print("== Step 2: Running models (this can take a while) ==")
        run_all(
            prompts_path=str(prompts_path),
            translated_path=str(translated_path),
            out_path="results/json/raw_outputs.jsonl",
        )
    else:
        print("== Step 2: Skipped ==")

    print("== Step 3: Scoring ==")
    df = load_records("results/json/raw_outputs.jsonl", str(prompts_path))
    save_summary(df)

    print("== Step 4: Generating figures ==")
    generate_all()

    print("Done. See results/csv/ and reports/figures/.")


if __name__ == "__main__":
    main()
