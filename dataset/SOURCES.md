# Dataset Sources & Attribution

Prompts marked `"source": "original"` in `prompts.json` were written for this project.

Prompts marked `"source": "adapted:public-jailbreak-corpora"` are inspired by patterns
documented in public jailbreak research (e.g. DAN-style prompts, roleplay/persona attacks
widely discussed in the AI safety literature). No verbatim text from any single copyrighted
benchmark is reproduced; templates are rewritten in original wording.

When scaling the dataset to 300–500 prompts, log every adapted item here with:

```
- id: <prompt-id>
  adapted_from: <benchmark name + citation>
  changes: <what was changed vs. the original>
```

Recommended benchmarks to draw structural inspiration from (not to copy verbatim):
AdvBench, HarmBench, JailbreakBench, DoNotAnswer, XSTest (for over-refusal controls).
