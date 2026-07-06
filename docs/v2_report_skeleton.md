# SentinelAI v2.0.0 — Full Study Report (SKELETON — placeholders marked with [ ])

## Title
[Working title, e.g.: "SentinelAI v2: A 400-Prompt Cross-Lingual Safety Benchmark
for Open-Source LLMs — Findings and Methodological Corrections"]

## Abstract
[ ] 150-200 word summary. Draft after Results/Discussion are final. Should state:
    the research question, dataset scale (400 prompts, 12 categories, 5 languages,
    2 models), the two headline quantitative findings (classifier validation
    kappa, refusal-rate-by-category/language table), and the one-sentence
    takeaway for a non-specialist reader.

## 0. Version Contribution Framing (write this first, it anchors everything else)

| Version | Main Contribution |
|---|---|
| Pilot v1 | Established and validated the evaluation pipeline; surfaced (but did not yet resolve) the classifier-reliability question |
| Version 2 | Scaled the benchmark 28→400 prompts; resolved the classifier-reliability question with quantified human validation (kappa~0.02); found and fixed a real translation-pipeline bug; added statistical rigor (Wilson CIs) |

[ ] Use this framing explicitly in the Introduction and Conclusion -- it tells
    reviewers this is a deliberate, evidence-driven progression, not just "more
    data."

## 0.1 Figures Inventory (decide now, generate exactly these once data is ready)

| # | Figure | Purpose | Source script |
|---|---|---|---|
| 1 | Refusal rate by language (bar, per model) | Core result | visualization/plots.py:grouped_bar |
| 2 | Refusal rate by model (bar, per language) | Model comparison | visualization/plots.py:grouped_bar |
| 3 | Refusal rate by category (heatmap, category x language, per model) | Safety-capability breakdown -- NEW for v2 | visualization/plots.py:heatmap |
| 4 | Rule-based vs. human/LLM-judge agreement (bar, per language) | Methodology validation callback | new -- build from docs/human_validation_v1_findings.md data |
| 5 | Refusal rate with 95% CI error bars (per language) | Statistical uncertainty -- NEW for v2 | new -- built on scoring.py:refusal_table_with_ci |
| 6 | Translation completeness summary (before/after fix, if re-running Pilot v1's 28 for comparison) | Data quality discussion | new -- built on check_completeness() output |

## 0.2 Tables Inventory (decide now)

1. Dataset summary (category x difficulty tier counts -- pull directly from
   docs/v2_dataset_design.md's table)
2. Model configuration (name, size, quantization, context length -- from
   models/registry.yaml)
3. Overall refusal rate by model x language, with 95% CI
   (results/csv/refusal_rate_with_ci.csv)
4. Per-category refusal rate by model, with 95% CI
   (results/csv/category_breakdown_with_ci.csv)
5. Classifier agreement statistics (from Issue #1's human validation --
   already have this data, just needs restating in report format)
6. Threats to validity (narrative table, reused/updated from Pilot v1)

## 1. Introduction
- Motivation: do open-source LLMs refuse unsafe requests consistently across
  languages? (unchanged from Pilot v1, restate briefly)
- What's new in v2 vs Pilot v1: 400 vs 28 prompts, validated classifier
  (LLM-judge + human agreement, kappa~0.02 finding), fixed translation pipeline,
  documented model-coverage constraint.
- Contribution statement: [ ] 2-3 sentences on what this study adds to the
  public record (dataset + findings + open validation methodology).

## 2. Related Work
[ ] Reuse Pilot v1's section largely as-is (AdvBench, HarmBench, JailbreakBench,
    DoNotAnswer). Add one paragraph situating this work relative to any newer
    cross-lingual safety benchmarks published since Pilot v1 (check for recent
    papers before finalizing).

## 3. Methodology
### 3.1 Dataset (UPDATE from Pilot v1)
- 400 prompts, 12 categories x 30 + 40 benign controls
- 4 difficulty tiers per category: direct, softened, roleplay, obfuscated
- 28 original Pilot v1 prompts preserved at original IDs for continuity
- See docs/v2_dataset_design.md for full rationale

### 3.2 Translation (UPDATE from Pilot v1)
- NLLB-200, 5 languages (en source + hi/zh/ta/ms targets)
- [ ] Document the translation bug found + fixed (2 rounds -- max_new_tokens
      alone was insufficient; min_new_tokens + no_repeat_ngram_size was the
      real fix). Cite docs/tamil_quality_findings.md.
- [ ] Report final back-translation completeness check results across all 400
      prompts once translation finishes (should show ~0 flagged, vs. 2 flagged
      in the 28-prompt pilot).

### 3.3 Models (UPDATE from Pilot v1)
- mistral:latest, gemma:2b (via Ollama, local inference)
- [ ] State plainly: llama2:7b and gemma:latest (9B) could not be evaluated on
      this hardware (macOS 13, Apple Silicon) -- cite
      docs/ollama_compatibility_findings.md. This is a disclosed limitation,
      not a silent omission.

### 3.4 Refusal Classification (UPDATE from Pilot v1 -- major change)
- [ ] Report the human-validation finding prominently here, not buried in
      limitations: rule-based classifier vs. human-equivalent judgment,
      Cohen's kappa ~ 0.02 (chance-level agreement), 100% one-directional
      error (classifier under-detects, never over-detects). Cite
      docs/human_validation_v1_findings.md.
- [ ] State which classifier's output is used for the v2 headline numbers
      (rule-based, human-validated subset, or LLM-judge if the API run
      completed) and justify the choice given the above finding.

### 3.5 Statistical Methods (NEW section for v2)
- [ ] Wilson score 95% confidence intervals for all refusal-rate estimates
      (evaluation/scoring.py:wilson_ci) -- justify choice over normal
      approximation given n=30/category.
- [ ] State how ties/errors are handled (excluded from denominator, per
      Pilot v1's approach).

## 4. Results
[ ] All tables/figures below are placeholders -- fill in once
    generation + scoring complete.

### 4.1 Overall refusal rate by model x language (with 95% CI)
[ TABLE -- results/csv/refusal_rate_with_ci.csv ]

### 4.2 Refusal rate by category (with 95% CI)
[ TABLE -- results/csv/category_breakdown_with_ci.csv ]
[ ] Highlight any categories with notably low refusal (candidates for
    discussion: harmful_advice, illegal_activity tend to run lower in Pilot v1)

### 4.3 Over-refusal rate on benign controls
[ TABLE -- results/csv/over_refusal_rate.csv ]

### 4.4 Safety Consistency Score (cross-language std dev)
[ TABLE -- results/csv/safety_consistency.csv ]

### 4.5 Figures
[ ] reports/figures/refusal_by_language_v2.png
[ ] reports/figures/refusal_by_category_v2.png (NEW for v2 -- heatmap,
    category x language, per model)
[ ] reports/figures/error_rate_by_model_v2.png

## 5. Discussion
[ ] Write after results are in. Should address:
    - Does the larger, validated dataset change the Pilot v1 conclusions, or
      just add precision/confidence to them?
    - Which categories show the widest cross-language gaps, and any pattern
      (e.g., persona/roleplay-framed prompts vs. direct requests)?
    - How much of Pilot v1's apparent "0% Hindi/Tamil/Malay refusal" finding
      survives now that translation is fixed and classification is validated?

## 6. Threats to Validity
[ ] Reuse Pilot v1's structure (construct validity, internal validity,
    external validity, reliability), updating each subsection for what's
    changed:
    - Construct validity: classifier reliability now has real evidence,
      not just a caveat.
    - External validity: still limited to 2 open-source 7B-9B-class models;
      state this remains a real limitation even at 400 prompts.

## 7. Limitations -> Future Work
- [ ] Model coverage (2 of originally-planned 4 models; see 3.3)
- [ ] Automated LLM-judge cross-validation still pending API access
      (evaluation/llm_judge_refusal.py ready, not yet run at scale)
- [ ] Tamil generation-quality limitation (model capability, not fixable in
      pipeline; see docs/tamil_quality_findings.md Problem B)
- [ ] Single-turn evaluation only; no multi-turn/conversational jailbreak testing

## 8. Conclusion
[ ] 1 paragraph. Restate the two headline findings and the overall verdict on
    the original research question, calibrated to what 400 validated prompts
    actually support (vs. Pilot v1's necessarily hedged conclusions).

## Appendix
- [ ] Full category list with example prompts (1 per category, redacted/
      generalized if needed for the writeup medium)
- [ ] Reproducibility table (same format as Pilot v1: command, input, output,
      runtime)
