"""
translation/translator.py

Translates base (English) prompts into the target languages and produces
back-translations for the QA gate described in docs/methodology.md section 5.1.

Two backends are supported:
  - "nllb": Hugging Face's NLLB-200 model, run locally (no API key needed,
    but downloads a multi-GB model on first use).
  - "argos": argostranslate, fully offline, lighter weight, lower quality.
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict

LANG_CODES_NLLB = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "zh": "zho_Hans",
    "ta": "tam_Taml",
    "ms": "zsm_Latn",
}

LANGUAGES = ["hi", "zh", "ta", "ms"]


class Translator:
    def __init__(self, backend: str = "nllb"):
        self.backend = backend
        self._pipe = None
        self._tokenizer = None
        self._model = None

    def _load_nllb(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "facebook/nllb-200-distilled-600M"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._pipe = True

    def _nllb_translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self._tokenizer.src_lang = src_lang
        inputs = self._tokenizer(text, return_tensors="pt")
        tgt_id = self._tokenizer.convert_tokens_to_ids(tgt_lang)
        # Scale the output budget with input length instead of a fixed cap.
        # Fixed max_new_tokens=256 was found to truncate longer/denser prompts
        # for scripts (Chinese, Tamil, Malay) that need more subword tokens per
        # unit of English meaning than Hindi does in this tokenizer -- see
        # docs/tamil_quality_findings.md Problem A for the confirmed cases this
        # caused (adv-002, mis-002 lost most of their content in translation).
        input_len = inputs["input_ids"].shape[-1]
        max_new = min(512, max(64, input_len * 4))
        generated = self._model.generate(
            **inputs, forced_bos_token_id=tgt_id, max_new_tokens=max_new
        )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def translate(self, text: str, target_lang: str) -> str:
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


def check_completeness(entry: dict, original_en: str, threshold: float = 0.7) -> list:
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


def translate_dataset(prompts_path: Path, output_path: Path, backend: str = "nllb") -> None:
    translator = Translator(backend=backend)
    prompts = json.loads(prompts_path.read_text())

    translated: Dict[str, dict] = {}
    for item in prompts:
        pid = item["id"]
        entry = {"qa_status": "pending"}
        for lang in LANGUAGES:
            translated_text = translator.translate(item["prompt_en"], lang)
            back_translated = translator.back_translate(translated_text, lang)
            entry[f"prompt_{lang}"] = translated_text
            entry[f"back_translation_en_{lang}"] = back_translated
        translated[pid] = entry

        flagged = check_completeness(entry, item["prompt_en"])
        if flagged:
            entry["qa_status"] = f"FLAGGED: possible truncation in {', '.join(flagged)}"
            print(f"[translated] {pid}  WARNING possible truncation in: {', '.join(flagged)}")
        else:
            print(f"[translated] {pid}")

    output_path.write_text(json.dumps(translated, indent=2, ensure_ascii=False))
    print(f"Wrote {len(translated)} translated entries -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate the SentinelAI prompt dataset.")
    parser.add_argument("--prompts", default="dataset/prompts.json")
    parser.add_argument("--out", default="dataset/translated_prompts.json")
    parser.add_argument("--backend", default="nllb", choices=["nllb", "argos"])
    args = parser.parse_args()

    translate_dataset(Path(args.prompts), Path(args.out), backend=args.backend)
