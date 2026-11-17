from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.food11.promote import promote, promotion_threshold


def registry(candidate_accuracy):
    client = Mock()
    client.get_model_version.return_value = SimpleNamespace(
        version="2", run_id="candidate", aliases=[]
    )
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="1", run_id="champion")
    client.get_run.side_effect = [
        SimpleNamespace(
            info=SimpleNamespace(status="FINISHED"),
            data=SimpleNamespace(
                metrics={"test_accuracy": accuracy},
                params={"test_size": "1000", "evaluation_dataset": "mini"},
            ),
        )
        for accuracy in (candidate_accuracy, 0.8)
    ]
    return client


@pytest.mark.parametrize(
    "accuracy, alias",
    [(0.81, "challenger"), (0.85, "champion"), (promotion_threshold(0.8, 1000), "champion")],
)
def test_gate_decision_and_tags(accuracy, alias):
    client = registry(accuracy)
    result = promote(client, "2", 1000)
    client.set_registered_model_alias.assert_called_once_with("food11", alias, "2")
    assert result["promotion.threshold"] == pytest.approx(0.8252982213)
    assert result["promotion.candidate_accuracy"] == accuracy
    assert client.set_model_version_tag.call_count == 7


@pytest.mark.parametrize("accuracy", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_metric_does_not_move_alias(accuracy):
    client = registry(accuracy)
    with pytest.raises(ValueError):
        promote(client, "2", 1000)
    client.set_registered_model_alias.assert_not_called()


def test_test_size_must_match():
    client = registry(0.9)
    with pytest.raises(ValueError, match="test size"):
        promote(client, "2", 20)
    client.set_registered_model_alias.assert_not_called()


def test_missing_metric():
    client = registry(0.9)
    client.get_run.side_effect = [
        SimpleNamespace(
            info=SimpleNamespace(status="FINISHED"), data=SimpleNamespace(metrics={}, params={})
        )
    ] * 2
    with pytest.raises(ValueError, match="test_accuracy"):
        promote(client, "2", 1000)
    client.set_registered_model_alias.assert_not_called()
