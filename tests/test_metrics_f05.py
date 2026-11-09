"""Checks for the F0.5 scorer against hand-computed values.

Run either way:
    python tests/test_metrics_f05.py
    pytest tests/test_metrics_f05.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import f05_entity, macro_f05

TOL = 1e-12


def test_official_example():
    """Worked example from the problem statement: P=2/3, R=1 -> 0.714."""
    score = f05_entity(["S2-00047", "S3-00812"], ["S2-00047", "S2-00193", "S3-00812"])
    # 1.25 * (2/3) * 1 / (0.25 * (2/3) + 1) = 0.8333... / 1.1666... = 0.7142857142857143
    assert abs(score - 0.7142857142857143) < TOL, score


def test_one_false_positive_costs_more_than_one_miss():
    """Truth {a}, predict {a,b}: P=1/2, R=1 -> 0.5555..."""
    score = f05_entity(["a"], ["a", "b"])
    # 1.25 * 0.5 * 1 / (0.25 * 0.5 + 1) = 0.625 / 1.125 = 0.5555555555555556
    assert abs(score - 0.5555555555555556) < TOL, score


def test_one_miss_costs_less_than_one_false_positive():
    """Truth {a,b}, predict {a}: P=1, R=1/2 -> 0.8333..."""
    score = f05_entity(["a", "b"], ["a"])
    # 1.25 * 1 * 0.5 / (0.25 * 1 + 0.5) = 0.625 / 0.75 = 0.8333333333333334
    assert abs(score - 0.8333333333333334) < TOL, score
    assert score > f05_entity(["a"], ["a", "b"])  # precision is weighted 2x


def test_singletons():
    """An entity with no true matches scores 1.0 only for an empty prediction."""
    assert f05_entity([], []) == 1.0
    assert f05_entity("", "") == 1.0
    assert f05_entity([], ["S2-1"]) == 0.0


def test_empty_prediction_on_a_matched_entity():
    """Assumed 0.0: the problem statement does not define this case."""
    assert f05_entity(["a"], []) == 0.0


def test_no_overlap():
    assert f05_entity(["a"], ["b"]) == 0.0


def test_perfect():
    assert f05_entity(["a", "b"], ["b", "a"]) == 1.0


def test_parses_comma_separated_strings():
    """On-disk format: one comma-separated field, no quoting."""
    assert abs(f05_entity("S2-00047,S3-00812",
                          "S2-00047,S2-00193,S3-00812") - 0.7142857142857143) < TOL


def test_macro_average_includes_singletons():
    """Mean over entities: (1.0 + 1.0 + 0.8333...) / 3."""
    truth = {"S1-1": ["a"], "S1-2": [], "S1-3": ["a", "b"]}
    pred = {"S1-1": ["a"], "S1-2": [], "S1-3": ["a"]}
    expected = (1.0 + 1.0 + 0.8333333333333334) / 3
    assert abs(macro_f05(truth, pred) - expected) < TOL


def test_macro_counts_missing_predictions_as_empty():
    truth = {"S1-1": ["a"], "S1-2": []}
    assert abs(macro_f05(truth, {}) - 0.5) < TOL  # 0.0 for S1-1, 1.0 for S1-2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
