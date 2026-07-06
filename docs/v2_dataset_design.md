# Version 2 Dataset Design

**Issue:** #4 — Scale dataset from 28 → 300-500 prompts
**Decision:** Target 400 total prompts, keep the existing 12-category schema from
Pilot v1 for continuity with all scoring code (`evaluation/scoring.py`'s
`category_breakdown`, ID-prefix parsing in `evaluation/refusal.py`, etc.)

## Why 400, not 320 or 500

Each prompt requires: 4 translations + 4 back-translations (translation cost) +
2 models × 5 languages × 3 runs = 30 generation calls (generation cost).

| Total prompts | Total generation calls | Translation calls |
|---|---|---|
| 320 | 9,600 | 2,560 |
| **400** | **12,000** | **3,200** |
| 500 | 15,000 | 4,000 |

400 is the defensible midpoint of methodology.md's original 300-500 target,
balancing statistical value against the real compute/time cost on this hardware
(CPU-only NLLB translation, single-machine Ollama generation).

## Structure: 30 prompts × 12 categories + 40 benign controls = 400

Per category, 4 difficulty tiers (same as Pilot v1's schema):
- **Direct** (8/category): explicit, unambiguous unsafe request
- **Softened** (8/category): same request, hedged/indirect phrasing
- **Roleplay** (7/category): request wrapped in a persona/character framing
- **Obfuscated** (7/category): request hidden via encoding, fiction-wrapper, or
  multi-step indirection

| Category | Prefix | Count |
|---|---|---|
| Harmful advice | adv- | 30 |
| Cybersecurity | cyb- | 30 |
| Privacy | pri- | 30 |
| Misinformation | mis- | 30 |
| Medical | med- | 30 |
| Financial | fin- | 30 |
| Bias | bia- | 30 |
| Illegal activity | ill- | 30 |
| Prompt injection | inj- | 30 |
| Jailbreak | jlb- | 30 |
| Roleplay attacks | rol- | 30 |
| Sensitive data extraction | sde- | 30 |
| Benign controls | ctl- | 40 |
| **Total** | | **400** |

## Content approach (unchanged from Pilot v1)

All prompts remain generic, template-style requests consistent with published
red-team benchmarks (AdvBench, HarmBench style) — no operational specificity for
weapons, CBRN, malware code, or extremist content. The goal is measuring refusal
*behavior*, not producing a usable harm manual. See `dataset/SOURCES.md` for the
existing provenance policy, which continues to apply at this scale.

## Build plan

Given the size, this is being built in batches rather than all at once, to
maintain quality and allow review:
1. Batch 1: `adv`, `cyb` (60 prompts) — establishes the pattern
2. Batch 2: `pri`, `mis` (60 prompts)
3. Batch 3: `med`, `fin` (60 prompts)
4. Batch 4: `bia`, `ill` (60 prompts)
5. Batch 5: `inj`, `jlb` (60 prompts)
6. Batch 6: `rol`, `sde` (60 prompts)
7. Batch 7: `ctl` benign controls (40 prompts)

Each batch will be merged into `dataset/prompts.json` (replacing the 28-prompt
Pilot v1 version on the `v2-dev` branch only — `main` keeps the original 28 for
the frozen v1.0.0 release's reproducibility).
