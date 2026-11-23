
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src_validation import (
    select_best_threshold,
    evaluate_validation,
    select_best_model
)

# 1
def test_threshold_range():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = select_best_threshold(y, scores)
    assert 0 <= threshold <= 1

# 2
def test_threshold_is_float():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    assert isinstance(select_best_threshold(y, scores), float)

# 3
def test_validation_returns_metrics():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    result = evaluate_validation(y, scores, 0.5)
    assert "Average Precision" in result

# 4
def test_precision_exists():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    result = evaluate_validation(y, scores, 0.5)
    assert "Precision" in result

# 5
def test_recall_exists():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    result = evaluate_validation(y, scores, 0.5)
    assert "Recall" in result

# 6
def test_f1_exists():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    result = evaluate_validation(y, scores, 0.5)
    assert "F1 Score" in result

# 7
def test_metrics_between_zero_and_one():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    result = evaluate_validation(y, scores, 0.5)
    assert all(0 <= v <= 1 for v in result.values())

# 8
def test_best_model_selected():
    data = pd.DataFrame({
        "Model": ["LR", "RF"],
        "Average Precision": [0.7, 0.9]
    })
    assert select_best_model(data)["Model"] == "RF"

# 9
def test_empty_results_rejected():
    data = pd.DataFrame(columns=["Model", "Average Precision"])
    with pytest.raises(ValueError):
        select_best_model(data)

# 10
def test_best_model_not_worse_than_other():
    data = pd.DataFrame({
        "Model": ["A", "B", "C"],
        "Average Precision": [0.6, 0.8, 0.7]
    })
    assert select_best_model(data)["Average Precision"] == 0.8

# 11
def test_model_file_exists():
    assert Path("artifacts/best_ml_model.joblib").exists()

# 12
def test_validation_results_file_exists():
    assert Path("artifacts/ml_validation_comparison.csv").exists()

# 13
def test_test_predictions_file_exists():
    assert Path("artifacts/ml_test_predictions.csv").exists()
