# Benchmark Card (fill in once dataset is scaled to 300–500 prompts)

Following the "datasheets for datasets" convention, document:

- **Motivation:** see docs/methodology.md §1
- **Composition:** category counts, language counts, difficulty tier counts (generate this
  table programmatically from `dataset/prompts.json` once scaled — don't hand-maintain it)
- **Collection process:** original + adapted (see `dataset/SOURCES.md`)
- **Preprocessing:** translation + back-translation QA gate (methodology.md §5.1)
- **Uses:** cross-lingual safety evaluation; explicitly NOT intended as training data for
  making models "more jailbreak-resistant to these exact prompts" (that would overfit to the
  benchmark rather than improving general safety)
- **Distribution:** license TBD — recommend CC-BY-NC-4.0 given harmful-prompt content, with the
  harmfulness≥3 model outputs excluded from any public release (methodology.md §6)
- **Maintenance:** who updates it, how versioning works (`dataset/CHANGELOG.md`, add when first
  update happens)
