import sys
import os

_here        = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(_here)
db_path      = os.path.join(project_root, "new_factor_data.duckdb")

sys.path.append(os.path.join(project_root, 'alpha_model', 'src'))
sys.path.append(os.path.join(project_root, 'risk_model'))

import duckdb
import numpy as np
import pandas as pd

from create_forward_return import create_forward_return
from model_engine import fit_model
from alpha_score import score_signals
from matrix_construction import (
    B_matrix_construction,
    factor_covariance_matrix,
    specific_risk_diagonal_matrix,
    implied_asset_covariance_factor_model,
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(project_root, "new_factor_data.duckdb")

sys.path.append(os.path.join(project_root, 'alpha_model', 'src'))
sys.path.append(os.path.join(project_root, 'alpha_model', 'src', 'signal_lib'))
sys.path.append(os.path.join(project_root, 'risk_model'))

def select_universe(scores, quantile=0.1, min_price=5.0):
    """
    Select long and short candidates from alpha scores.
    - quantile: top/bottom fraction to select (0.1 = decile)
    - min_price: liquidity filter applied before scoring
    Returns combined df of selected stocks with a 'side' column (1=long, -1=short).
    """
    n = int(len(scores) * quantile)
    longs  = scores.head(n).copy()
    shorts = scores.tail(n).copy()
    longs["side"]  =  1
    shorts["side"] = -1
    return pd.concat([longs, shorts], ignore_index=True)


def get_portfolio_weights(start_date, end_date, horizon=5, quantile=0.1, lambda_risk=1.0):
    """
    Full pipeline: alpha model → universe selection → risk model → weights.

    Parameters
    ----------
    start_date  : training window start
    end_date    : training window end / signal date
    horizon     : forward return horizon in trading days
    quantile    : top/bottom fraction of scored universe to trade
    lambda_risk : risk aversion parameter in mean-variance optimisation
    """
    conn = duckdb.connect(db_path)

    # ── Step 1: Alpha model ───────────────────────────────────────────────────
    create_forward_return(conn, source_table="alpha_signals", horizon=horizon)

    train_end = conn.execute(f"""
        SELECT date FROM alpha_signals_with_fwd
        WHERE date >= DATE '{start_date}'
        AND date <  DATE '{end_date}'
        ORDER BY date
    """).fetch_df()["date"].values
    train_end = str(train_end[int(len(train_end) * 0.8) - 1])[:10]

    model, signal_cols, ic_mean, ic_ir = fit_model(
        conn,
        table="alpha_signals_with_fwd",
        start_date=start_date,
        end_date=train_end,
        horizon=horizon,
    )
    # print(f"  Alpha model  —  IC mean: {ic_mean:.4f}  |  IC IR: {ic_ir:.4f}")

    scores = score_signals(conn, model, end_date, "alpha_signals_with_fwd", signal_cols)

    # ── Step 2: Universe selection ────────────────────────────────────────────
    selected = select_universe(scores, quantile=quantile)
    cusips   = selected["cusip"].tolist()
    # print(f"  Universe     —  {len(cusips)} stocks  ({len(cusips)//2} long / {len(cusips)//2} short)")

    # ── Step 3: Risk model (full universe, then subset for covariance) ────────
    B_matrix_construction(conn, start_date, end_date)
    factor_covariance_matrix(conn, start_date, end_date, table_name="factor_covariance_matrix")
    specific_risk_diagonal_matrix(conn, start_date, end_date)

    Sigma, cusip_list = implied_asset_covariance_factor_model(
        conn,
        start_date=start_date,
        end_date=end_date,
        cusips=cusips,          # ← only build N×N for selected stocks
    )
    # print(f"  Risk model   —  Σ shape: {Sigma.shape}")

    # ── Step 4: Mean-variance optimisation ────────────────────────────────────
    mu_series = selected.set_index("cusip")["mu"].reindex(cusip_list).values
    weights   = mean_variance_weights(mu_series, Sigma, lambda_risk)

    result = pd.DataFrame({
        "cusip":  cusip_list,
        "mu":     mu_series,
        "weight": weights,
        "side":   selected.set_index("cusip")["side"].reindex(cusip_list).values,
    }).sort_values("weight", ascending=False)

    # print(f"scores shape: {scores.shape}")
    # print(f"selected shape: {selected.shape}")
    # print(f"cusips len: {len(cusips)}")
    # print(f"cusip_list len: {len(cusip_list)}")
    # print(f"unique cusips in result: {result['cusip'].nunique()}")
    conn.close()
    return result


def mean_variance_weights(mu, Sigma, lambda_risk=1.0):
    """
    Unconstrained mean-variance: w = (1/lambda) * Sigma^{-1} * mu, rescaled to sum to 1.
    """
    Sigma_inv = np.linalg.inv(Sigma)
    w = (1.0 / lambda_risk) * Sigma_inv @ mu
    # Rescale so absolute weights sum to 1 (dollar-neutral long/short)
    w = w / np.abs(w).sum()
    return w

if __name__ == "__main__":
    start_date  = "2005-01-03"
    end_date    = "2006-01-04"
    weights_df = get_portfolio_weights(start_date, end_date, horizon=5, quantile=0.1, lambda_risk=1.0)
    print(weights_df)