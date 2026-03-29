"""
model_engine.py

Fits a Ridge regression of H-day forward returns on z-scored alpha signals.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import duckdb
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr


SIGNAL_COLS = [
      "momentum_z",
      "momentum_med_z",
      "reversal_z",
      "volatility_z",
      "downside_vol_z",
      "turnover_z",
      "spread_z",
      "amihud_illiquidity_z",
      "high52w_z",
      "max_ret_z",
      "rel_volume_z",
      "skewness_z",
      "range_vol_z",
  ]


def fit_model(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    start_date: str,
    end_date: str,
    horizon: int = 5,
    alpha: float = 1.0,
    signal_cols: list[str] = SIGNAL_COLS,
) -> tuple:
    """
    Fit Ridge regression of H-day forward returns on z-scored signals
    over the training window [start_date, end_date).

    Parameters
    ----------
    conn        : DuckDB connection
    table       : table produced by create_forward_return()
    start_date  : start of training window (inclusive)
    end_date    : end of training window (exclusive) — model predicts AFTER this date
    horizon     : forward return horizon in trading days
    alpha       : Ridge regularization strength
    signal_cols : feature columns to use

    Returns
    -------
    model       : fitted Ridge model
    signal_cols : feature columns used (pass to predict_returns)
    ic_mean     : mean per-date rank IC on training data
    ic_ir       : IC / std(IC) — quality of the signal
    """
    target_col   = f"fwd_ret_{horizon}d"
    cols_sql     = ", ".join(signal_cols + [target_col])
    null_filters = " AND ".join(f"{col} IS NOT NULL" for col in signal_cols)

    df = conn.execute(f"""
        SELECT cusip, date, {cols_sql}
        FROM {table}
        WHERE date >= DATE '{start_date}'
        AND date <  DATE '{end_date}'
        AND {target_col} IS NOT NULL
        AND {null_filters}
    """).fetch_df()

    # Winsorize returns at 1%/99% to reduce outlier influence
    lo = df[target_col].quantile(0.01)
    hi = df[target_col].quantile(0.99)
    df[target_col] = df[target_col].clip(lo, hi)

    X = df[signal_cols].to_numpy()
    y = df[target_col].to_numpy()

    model = Ridge(alpha=alpha)
    model.fit(X, y)

    # Per-date IC: rank correlation between predicted and realized returns
    df["predicted"] = model.predict(X)
    ic_series = (
        df.groupby("date")[["predicted", target_col]]
        .apply(lambda g: spearmanr(g["predicted"], g[target_col])[0])
        .dropna()
    )
    ic_mean = float(ic_series.mean())
    ic_ir   = float(ic_mean / ic_series.std()) if ic_series.std() > 0 else 0.0

    return model, signal_cols, ic_mean, ic_ir


def predict_returns(
    conn: duckdb.DuckDBPyConnection,
    model: Ridge,
    signal_cols: list[str],
    as_of_date: str,
    table: str,
) -> pd.DataFrame:
    """
    Generate expected return µ_{i,t} for all stocks at as_of_date.

    Returns DataFrame with columns: cusip, permno, mu
    """
    cols_sql     = ", ".join(signal_cols)
    null_filters = " AND ".join(f"{col} IS NOT NULL" for col in signal_cols)

    df = conn.execute(f"""
        SELECT cusip, permno, {cols_sql}
        FROM {table}
        WHERE date = DATE '{as_of_date}'
        AND {null_filters}
    """).fetch_df()

    if df.empty:
        raise ValueError(f"No signal data found for date {as_of_date}.")

    df["mu"] = model.predict(df[signal_cols].to_numpy())

    return df[["cusip", "permno", "mu"]]