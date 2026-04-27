import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, request
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "har.csv"

LABEL_CANDIDATES = {"activity", "label", "target", "class", "y"}

app = Flask(__name__)

model: Optional[DecisionTreeClassifier] = None
feature_names: List[str] = []
label_column: Optional[str] = None
training_error: Optional[str] = None


def _pick_label_column(columns: List[str]) -> str:
    by_lower = {c.lower(): c for c in columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in by_lower:
            return by_lower[candidate]
    return columns[-1]


def _load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series, List[str], str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected: {BASE_DIR / 'dataset' / 'har.csv'}"
        )

    df = pd.read_csv(path)
    if df.shape[0] < 1 or df.shape[1] < 2:
        raise ValueError("Dataset must have at least 1 row and 2 columns (features + label).")

    label_col = _pick_label_column(list(df.columns))
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # HAR features are typically numeric; coerce and fail fast on non-numeric input.
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        nan_count = int(X.isna().sum().sum())
        raise ValueError(f"Found {nan_count} non-numeric feature values (NaN after coercion).")

    return X, y, list(X.columns), label_col


def train() -> None:
    global model, feature_names, label_column, training_error
    try:
        X, y, cols, label_col = _load_dataset(DATASET_PATH)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        model = clf
        feature_names = cols
        label_column = label_col
        training_error = None
        app.logger.info(
            "Trained DecisionTreeClassifier on %d rows, %d features (label=%s).",
            X.shape[0],
            X.shape[1],
            label_col,
        )
    except Exception as exc:  # noqa: BLE001 - surface training failures via health/predict
        model = None
        feature_names = []
        label_column = None
        training_error = f"{type(exc).__name__}: {exc}"
        app.logger.exception("Model training failed: %s", training_error)


def _coerce_features_row(row: pd.DataFrame) -> pd.DataFrame:
    coerced = row.apply(pd.to_numeric, errors="coerce")
    if coerced.isna().any().any():
        bad_cols = [c for c in coerced.columns if coerced[c].isna().any()]
        raise ValueError(f"Non-numeric values for features: {bad_cols}")
    return coerced


def _parse_features(payload: Any) -> pd.DataFrame:
    if not feature_names:
        raise RuntimeError("Model is not trained (no feature schema available).")

    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")

    features_obj = payload.get("features", payload)

    if isinstance(features_obj, list):
        if len(features_obj) != len(feature_names):
            raise ValueError(
                f"Expected {len(feature_names)} features, got {len(features_obj)}."
            )
        row = pd.DataFrame([features_obj], columns=feature_names)
        return _coerce_features_row(row)

    if isinstance(features_obj, dict):
        missing = [c for c in feature_names if c not in features_obj]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        row = pd.DataFrame([{c: features_obj.get(c) for c in feature_names}])
        return _coerce_features_row(row)

    raise ValueError("`features` must be a list or an object.")


@app.get("/health")
def health() -> Any:
    return jsonify(
        {
            "status": "ok" if model is not None else "not_ready",
            "dataset_path": str(DATASET_PATH),
            "label_column": label_column,
            "num_features": len(feature_names),
            "training_error": training_error,
        }
    )


@app.post("/predict")
def predict() -> Any:
    if model is None:
        return (
            jsonify(
                {
                    "error": "Model is not ready.",
                    "training_error": training_error,
                }
            ),
            503,
        )

    try:
        payload = request.get_json(force=True)
        X_row = _parse_features(payload)
        pred = model.predict(X_row)[0]
        return jsonify({"prediction": str(pred)})
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400


train()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").strip() in {"1", "true", "True", "yes", "YES"}
    app.run(host="0.0.0.0", port=port, debug=debug)
