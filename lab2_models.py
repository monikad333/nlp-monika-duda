from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier

MODEL_NAMES = ["nb", "rf", "mlp", "logreg"]

PARAM_GRIDS = {
    "nb": {"classifier__alpha": [0.1, 0.5, 1.0]},
    "rf": {"classifier__n_estimators": [100, 300], "classifier__max_depth": [None, 10, 20]},
    "logreg": {"classifier__C": [0.1, 1, 10]},
    "mlp": {"classifier__hidden_layer_sizes": [(128,), (256, 128)]},
}


def build_estimator(method: str, is_dense: bool, seed: int) -> Any:
    normalized = (method or "").strip().lower()

    if normalized == "nb":
        return GaussianNB() if is_dense else MultinomialNB()

    if normalized == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=None, random_state=seed)

    if normalized == "logreg":
        return LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=seed)

    if normalized == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=seed)

    raise ValueError(f"Unknown method '{method}'. Allowed: {MODEL_NAMES}, all")


def build_model(method: str, is_dense: bool, seed: int, use_gridsearch: bool) -> Any:
    estimator = build_estimator(method, is_dense, seed)

    if not use_gridsearch:
        return estimator

    normalized = (method or "").strip().lower()
    param_grid = PARAM_GRIDS.get(normalized)
    if not param_grid:
        return estimator

    wrapped_grid = {key.replace("classifier__", ""): value for key, value in param_grid.items()}
    return GridSearchCV(estimator, wrapped_grid, cv=3, scoring="f1_macro", n_jobs=1)


def resolve_methods(method_arg: str) -> list[str]:
    normalized = (method_arg or "").strip().lower()
    if normalized == "all":
        return list(MODEL_NAMES)

    methods = [item.strip() for item in normalized.split(",") if item.strip()]
    invalid = [item for item in methods if item not in MODEL_NAMES]
    if invalid:
        raise ValueError(f"Unknown method(s) {invalid}. Allowed: {MODEL_NAMES}, all")

    return methods
