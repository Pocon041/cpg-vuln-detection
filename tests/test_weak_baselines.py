from __future__ import annotations

from cpg_vuln.mining.motif import RiskMotif
from cpg_vuln.mining.weak_baselines import feature_matrix, fit_and_score_group


def test_feature_matrix_selects_named_feature_group() -> None:
    motifs = [
        RiskMotif("a", 0, {"known_alloc_api": 1.0, "num_nodes": 2.0}),
        RiskMotif("b", 1, {"known_alloc_api": 0.0, "num_nodes": 3.0}),
    ]

    matrix, labels, sample_ids = feature_matrix(motifs, group="api")

    assert matrix.shape[0] == 2
    assert labels.tolist() == [0, 1]
    assert sample_ids == ["a", "b"]


def test_fit_and_score_group_fits_only_training_split() -> None:
    class FakeClassifier:
        fit_rows = None

        def fit(self, matrix, labels):
            FakeClassifier.fit_rows = matrix.shape[0]
            return self

        def predict_proba(self, matrix):
            import numpy as np

            probabilities = np.linspace(0.2, 0.8, matrix.shape[0])
            return np.column_stack([1 - probabilities, probabilities])

    train = [
        RiskMotif("train-0", 0, {"known_alloc_api": 0.0}),
        RiskMotif("train-1", 1, {"known_alloc_api": 1.0}),
    ]
    val = [
        RiskMotif("val-0", 0, {"known_alloc_api": 0.0}),
        RiskMotif("val-1", 1, {"known_alloc_api": 1.0}),
    ]
    test = [
        RiskMotif("test-0", 0, {"known_alloc_api": 0.0}),
        RiskMotif("test-1", 1, {"known_alloc_api": 1.0}),
    ]

    report = fit_and_score_group(
        train,
        val,
        test,
        group="api",
        classifier_factory=FakeClassifier,
    )

    assert FakeClassifier.fit_rows == 2
    assert report["train_samples"] == 2
    assert report["val_samples"] == 2
    assert report["test_samples"] == 2


def test_weak_baseline_group_scoring_reports_progress(monkeypatch) -> None:
    import cpg_vuln.mining.weak_baselines as weak_baselines

    calls: list[dict[str, object]] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    def fake_fit_and_score_group(train, val, test, *, group, classifier_factory):
        return {"group": group}

    monkeypatch.setattr(weak_baselines, "tqdm", fake_tqdm)
    monkeypatch.setattr(weak_baselines, "fit_and_score_group", fake_fit_and_score_group)

    results = weak_baselines.score_weak_baseline_groups(
        [],
        [],
        [],
        classifier_factory=object,
    )

    assert calls[0]["desc"] == "weak baseline groups"
    assert calls[0]["unit"] == "group"
    assert set(results) == {"api", "motif", "scale", "relation"}
