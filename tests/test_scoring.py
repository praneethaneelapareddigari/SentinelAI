import pandas as pd
from evaluation.scoring import refusal_table, safety_consistency_score


def _sample_df():
    return pd.DataFrame(
        [
            {"model": "llama3", "language": "en", "refused": 1.0, "is_benign_control": False, "is_error": False},
            {"model": "llama3", "language": "en", "refused": 1.0, "is_benign_control": False, "is_error": False},
            {"model": "llama3", "language": "ta", "refused": 0.0, "is_benign_control": False, "is_error": False},
            {"model": "llama3", "language": "ta", "refused": 1.0, "is_benign_control": False, "is_error": False},
        ]
    )


def test_refusal_table_shape():
    df = _sample_df()
    table = refusal_table(df)
    assert "en" in table.columns
    assert "ta" in table.columns
    assert table.loc["llama3", "en"] == 100.0
    assert table.loc["llama3", "ta"] == 50.0


def test_safety_consistency_lower_is_more_consistent():
    df = _sample_df()
    table = refusal_table(df)
    scores = safety_consistency_score(table)
    assert scores["llama3"] > 0  # en=100, ta=50 -> nonzero std dev
