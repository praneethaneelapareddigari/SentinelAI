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
        generated = self._model.generate(
            **inputs, forced_bos_token_id=tgt_id, max_new_tokens=256
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


def translate_dataset(
    prompts_path: Path, output_path: Path, backend: str = "nllb"
) -> None:
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