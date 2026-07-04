# Policy Brief (draft — fill in after running the full evaluation)

**Audience:** policymakers / trust & safety teams deploying LLMs in multilingual South/Southeast
Asian markets.

## Structure to fill in once results exist

1. **One-paragraph summary** — do safety behaviors hold up across languages? (headline finding)
2. **Key numbers** — safety consistency score per model (from `results/csv/safety_consistency.csv`)
3. **Which languages are most at risk** — ranked, with the over-refusal caveat (a language
   with a "high refusal rate" that's actually just over-refusing everything is not evidence of
   good safety).
4. **Which attack patterns transfer** — summarized at the pattern level only, per the
   responsible-disclosure note in methodology.md §6 (no working exploit recipes).
5. **Recommendations** — e.g. mandating multilingual red-teaming before regional deployment,
   requiring disclosure of which languages were included in safety fine-tuning data.
6. **Limitations** — carry over methodology.md §7 in plain language.

Keep this document to 2 pages; it's meant to be read by someone without ML background.
