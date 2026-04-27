from __future__ import annotations

from typing import Any, Mapping, Sequence, Union, cast

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


ArrayLike2D = Union[np.ndarray, pd.DataFrame]
ArrayLike1D = Union[np.ndarray, pd.Series, Sequence[Any]]


def train_model(X: ArrayLike2D, y: ArrayLike1D) -> DecisionTreeClassifier:
    """
    Train and return a DecisionTreeClassifier.

    Parameters
    ----------
    X:
        2D features (pandas DataFrame or numpy array).
    y:
        1D labels/targets.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model


def predict(model: DecisionTreeClassifier, features: Any) -> Any:
    """
    Predict a single label from one feature row.

    `features` may be:
    - list/tuple/1D numpy array (ordered feature values)
    - dict (feature_name -> value); if the model exposes `feature_names_in_`,
      that order is used.
    - pandas Series (treated like a dict with index as column names)
    - pandas DataFrame / 2D numpy array (uses first row)
    """
    if isinstance(features, pd.DataFrame):
        X_row = features.iloc[[0]]
    elif isinstance(features, np.ndarray):
        arr = features
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        X_row = arr
    elif isinstance(features, pd.Series):
        X_row = _dict_to_row(model, features.to_dict())
    elif isinstance(features, Mapping):
        X_row = _dict_to_row(model, cast(Mapping[str, Any], features))
    elif isinstance(features, (list, tuple)):
        X_row = np.asarray(features, dtype=float).reshape(1, -1)
    else:
        raise ValueError("Unsupported features type for prediction.")

    return model.predict(X_row)[0]


def _dict_to_row(model: DecisionTreeClassifier, d: Mapping[str, Any]) -> pd.DataFrame:
    order = getattr(model, "feature_names_in_", None)
    if order is None:
        # Fall back to a stable order when column metadata is unavailable.
        columns = sorted(d.keys())
    else:
        columns = list(order)

    row = pd.DataFrame([{c: d.get(c) for c in columns}])
    row = row.apply(pd.to_numeric, errors="coerce")
    if row.isna().any().any():
        bad_cols = [c for c in row.columns if row[c].isna().any()]
        raise ValueError(f"Non-numeric values for features: {bad_cols}")
    return row

