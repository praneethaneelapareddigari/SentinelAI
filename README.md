# SentinelAI

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-Pilot%20v1%20Complete-brightgreen) [![tests](https://github.com/praneethaneelapareddigari/SentinelAI/actions/workflows/tests.yml/badge.svg)](https://github.com/praneethaneelapareddigari/SentinelAI/actions/workflows/tests.yml)

**A cross-lingual LLM safety evaluation framework for South/Southeast Asian languages.**

> ⚠️ **Read this before the numbers below.** Pilot v1 is a 28-prompt run designed to validate the *measurement pipeline*, not to establish safety benchmarks. Every quantitative result in this README is an **exploratory pilot finding**. See [Caveats & Known Limitations](#caveats--known-limitations) — placed deliberately *before* the Results section — for what's solid, what's confounded, and what's an infrastructure artifact rather than a model-behavior finding.

**Status: Pilot v1 complete.** Full writeup: [`reports/SentinelAI_Pilot_v1_Report.pdf`](reports/SentinelAI_Pilot_v1_Report.pdf)

---

## Project Overview

Most public LLM safety benchmarks (AdvBench, HarmBench, JailbreakBench, DoNotAnswer) are English-only. Models deployed across South and Southeast Asia serve users in Hindi, Mandarin, Tamil, Malay, and English simultaneously. If safety alignment doesn't transfer across languages, that gap is invisible to English-only evaluation and only surfaces in production.

SentinelAI measures whether open-source LLMs' safety behavior — refusal of unsafe requests, resistance to jailbreaks, factual reliability — holds up consistently across languages, using the same prompts translated and run against the same models.

This is a research project, not a product. Read `docs/methodology.md` first — it defines the hypothesis, dataset design, metrics, and experimental protocol this codebase implements, including the full Threats to Validity analysis from Pilot v1.

## Research Question

> Do open-source LLMs exhibit consistent safety behavior across languages when presented with equivalent prompts?

**Primary hypothesis (H1):** refusal rates for unsafe requests are measurably lower in lower-resource languages (Tamil, Malay) than in higher-resource languages (English, Hindi, Mandarin). Pilot v1 was **not designed to test H1 conclusively** — 28 prompts is far too small a sample — but to validate that the measurement pipeline itself produces trustworthy data before investing in a larger (300–500 prompt) run.

## Caveats & Known Limitations

**This section applies to every number in the Results section below — read it first, not after.**

### Infrastructure limitations (not experimental findings)

- **`llama2:7b` and `gemma:latest` (9B) are excluded from all results below.** They produced reproducible, deterministic Ollama API-level crashes (`"llama runner process no longer running"`) on the test machine (macOS 13, Apple Silicon, Ollama v0.10.1), isolated to the `/api/generate` code path — both models run fine interactively via `ollama run`. **This is a tooling/infrastructure limitation of the test environment, not a finding about either model's safety behavior.** Results below reflect `mistral` (7B) and `gemma` (2B) only.

### Measurement caveats (affect how every result should be read)

- **Non-English refusal detection is unreliable.** The rule-based classifier's non-English regex patterns are sparse and under-validated. Manual review confirmed at least one case (`mistral`, prompt `adv-002`, Hindi) where the model genuinely refused but was misclassified as compliance because the classifier didn't recognize the phrasing used. **The 0% refusal rates shown for Hindi, Tamil, and Malay are unreliable lower bounds — not evidence that these models fail to refuse in those languages.**
- **Tamil results are a known confound, not yet a finding.** Several Tamil outputs showed repetitive/incoherent generation independent of the refusal question. Until this is resolved via the translation QA / back-translation gate (`methodology.md` §5.1), **Tamil numbers should not be interpreted as reflecting actual model safety behavior in Tamil** — they may equally reflect translation or generation quality issues.
- **The "safety consistency score" (σ) is an exploratory measure, not a validated benchmark metric.** It is simply the standard deviation of refusal rate across languages in this single 28-prompt run. It has no established confidence interval, has not been validated against a larger sample, and — given the two caveats above — is currently capturing classifier noise as much as genuine cross-lingual behavior. Treat it as a placeholder for the kind of metric Version 2 will properly validate, not as a settled result.
- **Sample size (28 prompts) validates pipeline functionality, not generalizable safety claims.** It is not sufficient to confirm or reject H1. Every percentage below should be read as "what this pipeline measured in this one run," not as a claim about model safety in general.
- **Model coverage**: only 2 of 4 intended models produced valid data on this hardware/software configuration (see Infrastructure limitations above).

Full detail and exact evidence for each point above is in `docs/methodology.md` §8 (Threats to Validity).

## Results — Pilot v1 *(exploratory pilot data — see Caveats above)*

28 prompts × 5 languages (English, Hindi, Mandarin, Tamil, Malay) × 3 runs each. Reflects `mistral` (7B) and `gemma` (2B) only — see Infrastructure limitations above.

**Exploratory refusal rate by language** (`results/csv/refusal_rate.csv`):

| Model | English | Hindi † | Mandarin | Tamil †‡ | Malay † |
|---|---|---|---|---|---|
| Mistral 7B | 9.9% | 0.0% | 4.3% | 0.0% | 4.3% |
| Gemma 2B | 16.7% | 0.0% | 41.7% | 0.0% | 0.0% |

† classifier under-detects non-English refusals — treat as an unreliable lower bound, not a measured rate.
‡ Tamil is additionally confounded by generation/translation quality issues (see Caveats above).

![Pilot v1 refusal rate by language chart](reports/figures/refusal_by_language.png)

**Exploratory safety consistency score** (σ = std dev of refusal rate across languages — **not a validated metric**, see Caveats above):

| Model | Consistency Score (σ) |
|---|---|
| Mistral 7B | 4.08 |
| Gemma 2B | 18.27 |

**Over-refusal rate** (benign controls): 0.0% for both models, all languages — neither model incorrectly refused a benign prompt.

**Infrastructure reliability by model** *(an API/tooling metric, not a safety metric — see Infrastructure limitations above)*:

![Ollama API failure rate by model chart](reports/figures/error_rate_by_model.png)

Full breakdown by category is in `results/csv/category_breakdown.csv`.

## Architecture

```
User
  |
  v
Prompt Dataset (dataset/prompts.json)
  |
  v
Translation Module      | translation/translator.py
(NLLB-200, 4 langs)     | + back-translation QA
  |
  v
Model Runner            | models/model_runner.py
(Ollama-served LLMs)    | models/registry.yaml
  |
  v
Evaluation Layer        | evaluation/
refusal.py              | ├─ rule-based classifier
hallucination.py        | ├─ LLM-as-judge
jailbreak.py            | ├─ paired attack success
scoring.py              | └─ aggregation + consistency score
  |
  v
Visualization           | visualization/
heatmaps, bars,         | plots.py, dashboard.py
radar charts            |
  |
  v
Results (results/csv, results/json)
  |
  v
Technical Report (reports/SentinelAI_Pilot_v1_Report.pdf)
```

Orchestrated end-to-end by `pipeline/run_evaluation.py`.

## Future Work (Version 2, before scaling to 300–500 prompts)

1. **Pin exact model versions** in `models/registry.yaml` and all Quickstart commands — replace `:latest` tags with exact, dated versions (e.g. via `ollama show <model> --modelfile` to record the digest) so Pilot v1 results remain reproducible even as upstream tags move.
2. **Improve multilingual refusal detection** — expand language-specific patterns using real observed refusal phrasings, and add an LLM-as-judge pass rather than relying on regex alone: `Prompt → Model Response → Rule-based Classifier → LLM-as-a-Judge → Human Validation → Final Label`.
3. **Resolve the Tamil quality question** via the translation QA / back-translation gate (methodology §5.1), to separate translation-side from generation-side issues.
4. **Resolve or re-document the Ollama model compatibility issue** — try an Ollama upgrade or a cloud/Linux environment (e.g. Colab) to restore full 4-model coverage.
5. **Scale the dataset** from 28 → 100 → 300–500 prompts, only after 1–4 are addressed.
6. Write the policy brief (`docs/policy.md`) translating findings into governance implications for regional multilingual LLM deployment.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Pull the models used in Pilot v1 (Llama 2 7B / Gemma 9B are excluded — see Caveats above)
# NOTE: pin exact versions before your next run (see Future Work #1) — replace
# the `:latest` / floating tags below with the exact tag or digest you validate against.
ollama pull mistral:latest   # TODO: pin to exact tag, e.g. mistral:7b-instruct-vX.Y
ollama pull gemma:2b

# 2. Set your judge-model API key (used only for hallucination scoring)
export ANTHROPIC_API_KEY=sk-...

# 3. Run the full pipeline (translate -> run models -> score -> plot)
python -m pipeline.run_evaluation --backend nllb

# Or re-score/re-plot without re-running expensive steps:
python -m pipeline.run_evaluation --skip-translate --skip-run

# If some API calls failed (see docs/methodology.md §8.3), retry only the failed ones:
python models/rerun_failed.py
```

Run tests:
```bash
pytest tests/
```

**Note on `llama2:7b` / `gemma:latest` (9B):** these produced 100% API-level failures during Pilot v1 on macOS 13 / Apple Silicon / Ollama v0.10.1, isolated to the `/api/generate` code path (they work fine via `ollama run`). This is an infrastructure limitation of the test environment, not a model finding — see Caveats above. If your environment doesn't hit this issue, add them back to `models/registry.yaml` and re-run.

## Repository Structure

```
SentinelAI/
├── README.md
├── requirements.txt
├── dataset/
│   ├── prompts.json               # 28-prompt pilot set; scale to 300-500 per methodology
│   ├── translated_prompts.json    # generated by translation/translator.py
│   ├── schema.md
│   └── SOURCES.md
├── translation/
│   └── translator.py              # NLLB backend + back-translation QA support
├── models/
│   ├── registry.yaml              # Pilot v1: mistral + gemma_small only — pin exact tags (see Future Work #1)
│   ├── model_runner.py            # Ollama-based unified runner, model-outer loop order
│   └── rerun_failed.py            # retries only failed/errored API calls
├── evaluation/
│   ├── refusal.py                 # rule-based multilingual refusal classifier + is_error()
│   ├── hallucination.py           # LLM-judge hallucination scorer
│   ├── jailbreak.py               # paired direct-vs-jailbreak success rate
│   └── scoring.py                 # aggregation -> summary CSVs, consistency score
├── visualization/
│   ├── plots.py                   # heatmap / bar / radar (matplotlib)
│   └── dashboard.py                # interactive Dash app
├── pipeline/
│   └── run_evaluation.py          # end-to-end orchestrator
├── results/
│   ├── csv/                       # Pilot v1 summary tables (real data)
│   └── json/                      # raw model outputs (reproducibility)
├── reports/
│   ├── SentinelAI_Pilot_v1_Report.docx   # full technical report — START HERE
│   └── build_report.js            # regenerate report from updated data
│
│   <!-- REST OF TREE UNCHANGED FROM CURRENT README — docs/, tests/, etc. -->
```

<!-- Sections below (docs/ + tests/ tree entries, License, remainder of repo structure)
     were not fully visible in the source screenshots — carry them over unchanged
     from the current live README before committing. -->
