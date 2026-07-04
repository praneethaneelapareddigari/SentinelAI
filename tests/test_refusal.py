from evaluation.refusal import is_refusal, refusal_rate


def test_english_refusal_detected():
    assert is_refusal("I'm sorry, but I can't help with that request.", "en")


def test_english_compliance_not_flagged():
    assert not is_refusal("Sure, here's a sourdough recipe: start with a starter...", "en")


def test_refusal_rate_computation():
    records = [
        {"language": "en", "category": "cybersecurity", "raw_output": "I can't help with that.", "is_benign_control": False},
        {"language": "en", "category": "cybersecurity", "raw_output": "Sure, here's how...", "is_benign_control": False},
        {"language": "en", "category": "cybersecurity", "raw_output": "Here's a sourdough recipe", "is_benign_control": True},
    ]
    rate = refusal_rate(records, language="en")
    assert rate == 0.5  # 1 of 2 non-benign items refused
