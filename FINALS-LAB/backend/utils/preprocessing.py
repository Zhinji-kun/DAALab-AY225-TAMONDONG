from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LABEL_CANDIDATES = ("activity", "label", "target", "class", "y")


def detect_sensor_columns(
    df: pd.DataFrame,
    accel_keywords: Sequence[str] = ("acc", "accel"),
    gyro_keywords: Sequence[str] = ("gyro", "gy", "gyroscope"),
    label_candidates: Sequence[str] = LABEL_CANDIDATES,
) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Best-effort detection of accelerometer + gyroscope columns and (optional) label column.

    Returns (accel_cols, gyro_cols, label_col).
    """
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    label_col = None
    for c in label_candidates:
        if c.lower() in lower_map:
            label_col = lower_map[c.lower()]
            break

    numeric_cols = [
        c for c in cols if c != label_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    def has_any_kw(col: str, keywords: Sequence[str]) -> bool:
        s = col.lower()
        return any(kw.lower() in s for kw in keywords)

    accel_cols = [c for c in numeric_cols if has_any_kw(c, accel_keywords)]
    gyro_cols = [c for c in numeric_cols if has_any_kw(c, gyro_keywords)]

    # If heuristics fail, fall back to "all numeric feature columns".
    if not accel_cols and not gyro_cols:
        accel_cols = numeric_cols
        gyro_cols = []

    return accel_cols, gyro_cols, label_col


def zscore_normalize(
    df: pd.DataFrame, columns: Sequence[str]
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Z-score normalize selected columns: (x - mean) / std.

    Returns (normalized_df, stats) where stats[col] = (mean, std).
    """
    out = df.copy()
    stats: Dict[str, Tuple[float, float]] = {}

    for c in columns:
        col = pd.to_numeric(out[c], errors="coerce")
        mean = float(col.mean())
        std = float(col.std(ddof=0))
        if std == 0.0 or np.isnan(std):
            std = 1.0
        out[c] = (col - mean) / std
        stats[c] = (mean, std)

    if out[columns].isna().any().any():
        bad_cols = [c for c in columns if out[c].isna().any()]
        raise ValueError("Non-numeric values encountered in columns: %s" % bad_cols)

    return out, stats


def segment_windows(
    df: pd.DataFrame,
    window_size: int,
    step_size: int,
    feature_cols: Sequence[str],
    label_col: Optional[str] = None,
) -> Iterable[Tuple[pd.DataFrame, Optional[Any]]]:
    """
    Segment a time-ordered DataFrame into sliding windows.

    Yields (window_df, window_label). `window_label` is the majority label in the window
    when `label_col` is provided; otherwise None.
    """
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive integers.")
    if window_size > len(df):
        return

    n = len(df)
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        win = df.iloc[start:end]
        win_x = win.loc[:, list(feature_cols)]

        win_y: Optional[Any] = None
        if label_col is not None and label_col in win.columns:
            # Majority vote label for the window.
            vc = win[label_col].value_counts(dropna=False)
            win_y = vc.index[0] if len(vc) else None

        yield win_x, win_y


def extract_mean_variance_features(
    window: pd.DataFrame, prefix: str = ""
) -> Dict[str, float]:
    """
    Compute per-column mean and variance features for one window.

    Feature names are: {prefix}{col}_mean and {prefix}{col}_var
    """
    feats: Dict[str, float] = {}
    for c in window.columns:
        col = pd.to_numeric(window[c], errors="coerce")
        if col.isna().any():
            raise ValueError("Non-numeric values encountered in window column: %s" % c)
        feats["%s%s_mean" % (prefix, c)] = float(col.mean())
        feats["%s%s_var" % (prefix, c)] = float(col.var(ddof=0))
    return feats


def preprocess_har(
    df: pd.DataFrame,
    window_size: int = 128,
    step_size: int = 64,
    accel_cols: Optional[Sequence[str]] = None,
    gyro_cols: Optional[Sequence[str]] = None,
    label_col: Optional[str] = None,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
    """
    End-to-end preprocessing for accelerometer/gyroscope style HAR data:
    - detect accel/gyro/label columns (optional)
    - z-score normalize sensor channels (optional)
    - segment into sliding windows
    - extract mean/variance features per channel

    Returns (X, y, metadata).
    """
    if accel_cols is None or gyro_cols is None or label_col is None:
        detected_acc, detected_gyro, detected_label = detect_sensor_columns(df)
        if accel_cols is None:
            accel_cols = detected_acc
        if gyro_cols is None:
            gyro_cols = detected_gyro
        if label_col is None:
            label_col = detected_label

    accel_cols = list(accel_cols or [])
    gyro_cols = list(gyro_cols or [])
    feature_cols = accel_cols + gyro_cols

    if not feature_cols:
        raise ValueError("No feature columns provided/detected for preprocessing.")

    work = df.copy()
    norm_stats: Dict[str, Tuple[float, float]] = {}
    if normalize:
        work, norm_stats = zscore_normalize(work, feature_cols)

    rows: List[Dict[str, float]] = []
    labels: List[Any] = []

    for win_x, win_y in segment_windows(
        work, window_size=window_size, step_size=step_size, feature_cols=feature_cols, label_col=label_col
    ):
        feats = extract_mean_variance_features(win_x)
        rows.append(feats)
        if label_col is not None:
            labels.append(win_y)

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name=label_col) if label_col is not None else None

    metadata = {
        "window_size": window_size,
        "step_size": step_size,
        "accel_cols": accel_cols,
        "gyro_cols": gyro_cols,
        "label_col": label_col,
        "normalize": normalize,
        "normalization_stats": norm_stats,
        "num_windows": int(len(X)),
    }
    return X, y, metadata

