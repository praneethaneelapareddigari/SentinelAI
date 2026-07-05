import json
import tempfile
from pathlib import Path

from evaluation.llm_judge_refusal import compute_agreement


def test_compute_agreement_perfect_match(tmp_path):
    raw_outputs = tmp_path / "raw_outputs.jsonl"
    judge_labels = tmp_path / "judge_labels.jsonl"
    prompts = tmp_path / "prompts.json"

    prompts.write_text(json.dumps([{"id": "p1", "prompt_en": "test prompt"}]))

    raw_records = [
        {"model": "m1", "prompt_id": "p1", "language": "en", "run_index": 0,
         "raw_output": "I can't help with that."},
        {"model": "m1", "prompt_id": "p1", "language": "en", "run_index": 1,
         "raw_output": "Sure, here you go."},
    ]
    raw_outputs.write_text("\n".join(json.dumps(r) for r in raw_records))

    judge_records = [
        {"model": "m1", "prompt_id": "p1", "language": "en", "run_index": 0,
         "refused": True, "confidence": 0.95, "reasoning": "declines"},
        {"model": "m1", "prompt_id": "p1", "language": "en", "run_index": 1,
         "refused": False, "confidence": 0.9, "reasoning": "complies"},
    ]
    judge_labels.write_text("\n".join(json.dumps(r) for r in judge_records))

    report = compute_agreement(str(raw_outputs), str(judge_labels), str(prompts))

    assert report["overall"]["n"] == 2
    assert report["overall"]["agreement_rate"] == 1.0
    assert report["n_disagreements"] == 0


def test_compute_agreement_flags_disagreement(tmp_path):
    raw_outputs = tmp_path / "raw_outputs.jsonl"
    judge_labels = tmp_path / "judge_labels.jsonl"
    prompts = tmp_path / "prompts.json"

    prompts.write_text(json.dumps([{"id": "p1", "prompt_en": "test prompt"}]))

    # Rule-based classifier will NOT flag this as a refusal (no matching pattern),
    # but the judge says it IS a refusal -- simulates the Pilot v1 finding.
    raw_records = [
        {"model": "m1", "prompt_id": "p1", "language": "hi", "run_index": 0,
         "raw_output": "यह एक असामान्य अनुरोध है।"},
    ]
    raw_outputs.write_text(json.dumps(raw_records[0]))

    judge_records = [
        {"model": "m1", "prompt_id": "p1", "language": "hi", "run_index": 0,
         "refused": True, "confidence": 0.8, "reasoning": "declines in Hindi"},
    ]
    judge_labels.write_text(json.dumps(judge_records[0]))

    report = compute_agreement(str(raw_outputs), str(judge_labels), str(prompts))

    assert report["n_disagreements"] == 1
    assert report["sample_disagreements"][0]["rule_based_label"] is False
