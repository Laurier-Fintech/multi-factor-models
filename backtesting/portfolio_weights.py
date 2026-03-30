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

def select_universe(scores, quantile=0.1, long_only=False):
      n      = int(len(scores) * quantile)
      longs  = scores.head(n).copy()
      longs["side"] = 1

      if long_only:
          return longs

      shorts = scores.tail(n).copy()
      shorts["side"] = -1
      return pd.concat([longs, shorts], ignore_index=True)


def get_portfolio_weights(
    start_date, end_date, horizon=5, quantile=0.1, lambda_risk=1.0, long_only=False
):
    conn = duckdb.connect(db_path)

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

    scores   = score_signals(conn, model, end_date, "alpha_signals_with_fwd", signal_cols)
    selected = select_universe(scores, quantile=quantile, long_only=long_only)
    cusips   = selected["cusip"].tolist()

    B_matrix_construction(conn, start_date, end_date)
    factor_covariance_matrix(conn, start_date, end_date, table_name="factor_covariance_matrix")
    specific_risk_diagonal_matrix(conn, start_date, end_date)

    Sigma, cusip_list = implied_asset_covariance_factor_model(
        conn, start_date=start_date, end_date=end_date, cusips=cusips,
    )

    mu_series = selected.set_index("cusip")["mu"].reindex(cusip_list).values
    weights   = mean_variance_weights(mu_series, Sigma, lambda_risk, long_only=long_only)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    pred_var  = float(weights @ Sigma @ weights)
    n_longs   = int((selected["side"] ==  1).sum())
    n_shorts  = int((selected["side"] == -1).sum())

    diagnostics = {
        "ic_mean":  ic_mean,
        "ic_ir":    ic_ir,
        "pred_vol": float(np.sqrt(max(pred_var, 0))),
        "n_longs":  n_longs,
        "n_shorts": n_shorts,
    }

    result = pd.DataFrame({
        "cusip":  cusip_list,
        "mu":     mu_series,
        "weight": weights,
        "side":   selected.set_index("cusip")["side"].reindex(cusip_list).values,
    }).sort_values("weight", ascending=False)

    conn.close()
    return result, diagnostics


def mean_variance_weights(mu, Sigma, lambda_risk=1.0, long_only=False):
      """
      Unconstrained MV: w = (1/lambda) * Sigma^{-1} * mu.
      Long-only: negative weights are clipped to zero before normalising.
      """
      Sigma_inv = np.linalg.inv(Sigma)
      w = (1.0 / lambda_risk) * Sigma_inv @ mu

      if long_only:
          w = np.clip(w, 0, None)        # drop any negative-weight suggestions
          total = w.sum()
          if total <= 0:
              w = np.ones(len(w)) / len(w)   # fallback: equal weight if all clipped
          else:
              w = w / total              # weights sum to 1
      else:
          w = w / np.abs(w).sum()        # dollar-neutral: sum(|w|) = 1

      return w

if __name__ == "__main__":
    start_date  = "2005-01-03"
    end_date    = "2006-01-04"
    weights_df = get_portfolio_weights(start_date, end_date, horizon=5, quantile=0.1, lambda_risk=1.0)
    print(weights_df)