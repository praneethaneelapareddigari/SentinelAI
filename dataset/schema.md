# Prompt Dataset Schema

Each entry in `prompts.json` follows this schema:

```json
{
  "id": "cyb-001",
  "category": "cybersecurity",
  "difficulty": "direct",
  "is_benign_control": false,
  "prompt_en": "English text of the base prompt.",
  "notes": "Optional annotation, e.g. what makes this a jailbreak variant.",
  "source": "original | adapted:<benchmark-name>",
  "verifiable_claim": false
}
```

## Field notes

- `id`: `<category-prefix>-<3-digit-number>`. Prefixes: `adv` (harmful advice), `cyb`
  (cybersecurity), `pri` (privacy), `mis` (misinformation), `med` (medical), `fin` (financial),
  `bia` (bias), `ill` (illegal activity), `inj` (prompt injection), `jlb` (jailbreak), `rol`
  (roleplay attack), `sde` (sensitive data extraction), `ctl` (benign control).
- `difficulty`: one of `direct`, `softened`, `roleplay`, `obfuscated`.
- `is_benign_control`: true for the ~10% negative-control prompts used to measure over-refusal.
- `verifiable_claim`: true if the prompt invites a factual claim that can be checked, so it's
  eligible for hallucination scoring (§4.3 of methodology.md).

## Translated prompts

`translated_prompts.json` mirrors `prompts.json` but keys each entry by `id` and adds
`prompt_hi`, `prompt_zh`, `prompt_ta`, `prompt_ms`, plus a `back_translation_en` field per
language used for the translation QA gate (see methodology §5.1).

```json
{
  "cyb-001": {
    "prompt_hi": "...",
    "back_translation_en_hi": "...",
    "prompt_zh": "...",
    "back_translation_en_zh": "...",
    "prompt_ta": "...",
    "back_translation_en_ta": "...",
    "prompt_ms": "...",
    "back_translation_en_ms": "...",
    "qa_status": "pending"
  }
}
```
