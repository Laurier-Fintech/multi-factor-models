"""
alpha_score.py

Produces final cross-sectionally normalized alpha scores from model predictions.
"""
from __future__ import annotations
import duckdb
import pandas as pd
from model_engine import predict_returns, SIGNAL_COLS


def score_signals(
    conn: duckdb.DuckDBPyConnection,
    model,
    as_of_date: str,
    table: str,
    signal_cols: list[str] = SIGNAL_COLS,
) -> pd.DataFrame:
    """
    Generate a ranked alpha score for all stocks at as_of_date.

    Applies a cross-sectional z-score to µ so the scores are comparable
    across dates and ready for the portfolio optimizer.

    Returns DataFrame sorted descending by alpha_score:
        cusip | permno | mu | alpha_score
    """
    df = predict_returns(conn, model, signal_cols, as_of_date, table)

    mu_mean = df["mu"].mean()
    mu_std  = df["mu"].std()

    df["alpha_score"] = (
        (df["mu"] - mu_mean) / mu_std if mu_std > 0 else 0.0
    )

    return df.sort_values("alpha_score", ascending=False).reset_index(drop=True)