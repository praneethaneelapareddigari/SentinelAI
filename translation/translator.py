"""
translation/translator.py

Translates base (English) prompts into the target languages and produces
back-translations for the QA gate described in docs/methodology.md §5.1.

Two backends are supported:
  - "nllb": Hugging Face's NLLB-200 model, run locally (no API key needed,
    but downloads a multi-GB model on first use).
  - "argos": argostranslate, fully offline, lighter weight, lower quality.

Swap backends via TRANSLATION_BACKEND in config.py or the CLI flag below.
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict

# ISO codes used throughout the project -> NLLB language codes
LANG_CODES_NLLB = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "zh": "zho_Hans",
    "ta": "tam_Taml",
    "ms": "zsm_Latn",
}

LANGUAGES = ["hi", "zh", "ta", "ms"]  # target languages; "en" is the source


class Translator:
    """Thin wrapper so the rest of the pipeline doesn't care which backend runs underneath."""

    def __init__(self, backend: str = "nllb"):
        self.backend = backend
        self._pipe = None  # sentinel: set to True once NLLB model/tokenizer are loaded
        self._tokenizer = None
        self._model = None

    def _load_nllb(self):
        # Loading tokenizer + model directly (rather than the pipeline() shorthand)
        # avoids depending on transformers registering a generic "translation" task,
        # which varies by version and broke on at least one tested setup.
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "facebook/nllb-200-distilled-600M"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._pipe = True  # sentinel marking "loaded"

    def _nllb_translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self._tokenizer.src_lang = src_lang
        inputs = self._tokenizer(text, return_tensors="pt")
        tgt_id = self._tokenizer.convert_tokens_to_ids(tgt_lang)
        input_len = inputs["input_ids"].shape[-1]
        max_new = min(512, max(64, input_len * 4))
        min_new = max(1, int(input_len * 1.0))
        generated = self._model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
        )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def _nllb_translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Batched version: translates a list of texts in one model.generate() call.
        Substantially faster than one-at-a-time, even on CPU, since it amortizes
        the fixed overhead of each generate() call across many sentences.

        NOTE on min_new_tokens + no_repeat_ngram_size: forcing a minimum length
        alone caused the model to fall into degenerate repetition loops ("It's a
        good way. It's a good way. ...") once it had already said everything
        meaningful -- confirmed via direct testing. no_repeat_ngram_size=3 blocks
        the model from repeating any 3-token sequence, which prevents the loop;
        the min_new_tokens multiplier was also reduced from 1.5x to 1.0x input
        length, since 1.5x was overshooting real translation length and forcing
        unnecessary padding in the first place, especially for short prompts
        batched alongside longer ones (batch padding inflates the shared
        input_len used to compute min_new_tokens for the whole batch)."""
        self._tokenizer.src_lang = src_lang
        inputs = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        tgt_id = self._tokenizer.convert_tokens_to_ids(tgt_lang)
        input_len = inputs["input_ids"].shape[-1]
        max_new = min(512, max(64, input_len * 4))
        min_new = max(1, int(input_len * 1.0))
        generated = self._model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
        )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)

    def translate_batch(self, texts: list[str], target_lang: str) -> list[str]:
        if self.backend == "nllb":
            if self._pipe is None:
                self._load_nllb()
            return self._nllb_translate_batch(
                texts, LANG_CODES_NLLB["en"], LANG_CODES_NLLB[target_lang]
            )
        else:
            return [self.translate(t, target_lang) for t in texts]

    def back_translate_batch(self, texts: list[str], source_lang: str) -> list[str]:
        if self.backend == "nllb":
            if self._pipe is None:
                self._load_nllb()
            return self._nllb_translate_batch(
                texts, LANG_CODES_NLLB[source_lang], LANG_CODES_NLLB["en"]
            )
        else:
            return [self.back_translate(t, source_lang) for t in texts]
        if self.backend == "nllb":
            if self._pipe is None:
                self._load_nllb()
            return self._nllb_translate(
                text, LANG_CODES_NLLB["en"], LANG_CODES_NLLB[target_lang]
            )

        elif self.backend == "argos":
            import argostranslate.translate as at

            return at.translate(text, "en", target_lang)

        else:
            raise ValueError(f"Unknown translation backend: {self.backend}")

    def back_translate(self, text: str, source_lang: str) -> str:
        """Translate FROM target_lang back TO English, for the QA gate."""
        if self.backend == "nllb":
            if self._pipe is None:
                self._load_nllb()
            return self._nllb_translate(
                text, LANG_CODES_NLLB[source_lang], LANG_CODES_NLLB["en"]
            )
        elif self.backend == "argos":
            import argostranslate.translate as at

            return at.translate(text, source_lang, "en")
        else:
            raise ValueError(f"Unknown translation backend: {self.backend}")


def check_completeness(entry: dict, original_en: str, threshold: float = 0.7) -> list[str]:
    """
    Flags languages whose back-translation word count is suspiciously short
    relative to the original -- the signature of the truncation bug documented
    in docs/tamil_quality_findings.md. Returns a list of flagged language codes.
    """
    original_words = len(original_en.split())
    flagged = []
    for lang in LANGUAGES:
        bt = entry.get(f"back_translation_en_{lang}", "")
        ratio = len(bt.split()) / original_words if original_words else 1.0
        if ratio < threshold:
            flagged.append(lang)
    return flagged


def translate_dataset(
    prompts_path: Path,
    output_path: Path,
    backend: str = "nllb",
    batch_size: int = 8,
) -> None:
    """
    Translates the full prompt dataset with batching (for speed) and incremental
    saving after every batch (for resumability). If output_path already exists
    from a previous partial/interrupted run, already-completed prompt IDs are
    skipped automatically -- safe to re-run the exact same command after any
    interruption (crash, power loss, Ctrl+C) without losing prior progress or
    re-paying translation time already spent.
    """
    translator = Translator(backend=backend)
    prompts = json.loads(prompts_path.read_text())

    translated: Dict[str, dict] = {}
    if output_path.exists():
        translated = json.loads(output_path.read_text())
        print(f"Resuming: {len(translated)} entries already done, "
              f"{len(prompts) - len(translated)} remaining.")

    remaining = [p for p in prompts if p["id"] not in translated]

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_ids = [item["id"] for item in batch]
        batch_texts = [item["prompt_en"] for item in batch]

        batch_entries = {pid: {"qa_status": "pending"} for pid in batch_ids}

        for lang in LANGUAGES:
            translated_texts = translator.translate_batch(batch_texts, lang)
            back_translated_texts = translator.back_translate_batch(translated_texts, lang)
            for pid, tt, bt in zip(batch_ids, translated_texts, back_translated_texts):
                batch_entries[pid][f"prompt_{lang}"] = tt
                batch_entries[pid][f"back_translation_en_{lang}"] = bt

        for item in batch:
            pid = item["id"]
            entry = batch_entries[pid]
            flagged = check_completeness(entry, item["prompt_en"])
            if flagged:
                entry["qa_status"] = f"FLAGGED: possible truncation in {', '.join(flagged)}"
                print(f"[translated] {pid}  WARNING possible truncation in: {', '.join(flagged)}")
            else:
                print(f"[translated] {pid}")
            translated[pid] = entry

        # Save after every batch, not just at the end -- this is the fix that
        # prevents losing all progress if a long run gets interrupted partway.
        output_path.write_text(json.dumps(translated, indent=2, ensure_ascii=False))
        done = min(batch_start + batch_size, len(remaining))
        print(f"  -- saved progress: {done}/{len(remaining)} remaining prompts done "
              f"({len(translated)}/{len(prompts)} total) --")

    print(f"Done. Wrote {len(translated)} translated entries -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate the SentinelAI prompt dataset.")
    parser.add_argument("--prompts", default="dataset/prompts.json")
    parser.add_argument("--out", default="dataset/translated_prompts.json")
    parser.add_argument("--backend", default="nllb", choices=["nllb", "argos"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    translate_dataset(Path(args.prompts), Path(args.out), backend=args.backend, batch_size=args.batch_size)
