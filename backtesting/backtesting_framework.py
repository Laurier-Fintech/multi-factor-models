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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dateutil.relativedelta import relativedelta
from portfolio_weights import get_portfolio_weights


def get_trading_dates(conn, start, end):
    """Return all trading dates between start and end from the database."""
    return conn.execute(f"""
        SELECT DISTINCT date FROM cusip_daily_retx
        WHERE date >= DATE '{start}'
        AND date <= DATE '{end}'
        ORDER BY date
    """).fetch_df()["date"].values

def get_realised_return(conn, weights_df, signal_date, horizon):
    """
    Compute portfolio return from signal_date over the next horizon trading days.
    Uses compounded daily returns per cusip, weighted by portfolio weights.
    """
    cusips  = weights_df["cusip"].tolist()
    quoted  = ", ".join(f"'{c}'" for c in cusips)

    # Get the next `horizon` trading days after signal_date
    future_dates = conn.execute(f"""
        SELECT DISTINCT date FROM cusip_daily_retx
        WHERE date > DATE '{signal_date}'
        ORDER BY date
        LIMIT {horizon}
    """).fetch_df()["date"].values

    if len(future_dates) < horizon:
        return None  # not enough data

    period_end = str(future_dates[-1])[:10]

    # Compound daily returns over the holding period per cusip
    returns = conn.execute(f"""
        SELECT cusip, EXP(SUM(LN(1 + retx))) - 1 AS period_ret
        FROM cusip_daily_retx
        WHERE cusip IN ({quoted})
        AND date >  DATE '{signal_date}'
        AND date <= DATE '{period_end}'
        AND retx IS NOT NULL
        AND retx > -1
        GROUP BY cusip
    """).fetch_df()

    merged = weights_df[["cusip", "weight"]].merge(returns, on="cusip", how="inner")

    if merged.empty:
        return None

    portfolio_return = float((merged["weight"] * merged["period_ret"]).sum())
    return portfolio_return, period_end

def save_backtest_results(returns_df, weights_history, diagnostics_df, backtest_start, backtest_end):
    results_dir = os.path.join(_here, "results")
    os.makedirs(results_dir, exist_ok=True)

    tag = f"{backtest_start}_to_{backtest_end}"

    returns_path     = os.path.join(results_dir, f"returns_{tag}.csv")
    weights_path     = os.path.join(results_dir, f"weights_{tag}.csv")
    diagnostics_path = os.path.join(results_dir, f"diagnostics_{tag}.csv")

    returns_df.to_csv(returns_path, index=False)
    weights_history.to_csv(weights_path, index=False)
    diagnostics_df.to_csv(diagnostics_path, index=False)

    print(f"  Returns saved     → {returns_path}")
    print(f"  Weights saved     → {weights_path}")
    print(f"  Diagnostics saved → {diagnostics_path}")

def run_backtest(
    backtest_start,
    backtest_end,
    lookback_days=252,
    rebalance_days=5,
    horizon=5,
    quantile=0.1,
    lambda_risk=1.0,
    long_only=False,
    save_results=True,
):
    conn = duckdb.connect(db_path)
    all_dates = get_trading_dates(conn, backtest_start, backtest_end)
    conn.close()

    rebalance_dates  = all_dates[::rebalance_days]
    total            = len(rebalance_dates)
    records          = []
    weights_records  = []
    diagnostics_records = []

    for i, signal_date in enumerate(rebalance_dates):
        signal_date_str = str(signal_date)[:10]
        train_start     = str(pd.Timestamp(signal_date_str) - pd.DateOffset(days=lookback_days))[:10]

        print_progress(i + 1, total, signal_date_str)

        try:
            weights_df, diagnostics = get_portfolio_weights(
                start_date=train_start,
                end_date=signal_date_str,
                horizon=horizon,
                quantile=quantile,
                lambda_risk=lambda_risk,
                long_only=long_only,
            )

            conn   = duckdb.connect(db_path)
            result = get_realised_return(conn, weights_df, signal_date_str, horizon)
            conn.close()

            if result is None:
                continue

            port_ret, period_end = result
            print_progress(i + 1, total, signal_date_str, port_ret)

            records.append({
                "date":             signal_date_str,
                "period_end":       period_end,
                "portfolio_return": port_ret,
                "n_positions":      len(weights_df),
            })

            diagnostics_records.append({
                "date":     signal_date_str,
                "ic_mean":  diagnostics["ic_mean"],
                "ic_ir":    diagnostics["ic_ir"],
                "pred_vol": diagnostics["pred_vol"],
                "n_longs":  diagnostics["n_longs"],
                "n_shorts": diagnostics["n_shorts"],
            })

            w = weights_df.copy()
            w["date"] = signal_date_str
            weights_records.append(w[["date", "cusip", "weight", "side", "mu"]])

        except Exception as e:
            print(f"\n    ERROR on {signal_date_str}: {e}")
            continue

    print()

    returns_df = pd.DataFrame(records)
    returns_df["date"]       = pd.to_datetime(returns_df["date"])
    returns_df["cumulative"] = (1 + returns_df["portfolio_return"]).cumprod() - 1

    weights_history  = pd.concat(weights_records, ignore_index=True) if weights_records else pd.DataFrame()
    diagnostics_df   = pd.DataFrame(diagnostics_records)
    diagnostics_df["date"] = pd.to_datetime(diagnostics_df["date"])

    if save_results:
        save_backtest_results(returns_df, weights_history, diagnostics_df, backtest_start, backtest_end)

    return returns_df, weights_history, diagnostics_df

def compute_metrics(returns_df, rebalance_days=5):
      r = returns_df["portfolio_return"]

      total_return     = float((1 + r).prod() - 1)
      n_periods        = len(r)
      periods_per_year = 252 / rebalance_days

      ann_return = float((1 + total_return) ** (periods_per_year / n_periods) - 1)
      ann_vol    = float(r.std() * np.sqrt(periods_per_year))
      sharpe     = ann_return / ann_vol if ann_vol > 0 else 0.0

      cum         = (1 + r).cumprod()
      rolling_max = cum.cummax()
      drawdown    = (cum - rolling_max) / rolling_max
      max_dd      = float(drawdown.min())

      hit_rate = float((r > 0).mean())

      return {
          "Total return":       f"{total_return:.2%}",
          "Annualised return":  f"{ann_return:.2%}",
          "Annualised vol":     f"{ann_vol:.2%}",
          "Sharpe ratio":       f"{sharpe:.2f}",
          "Max drawdown":       f"{max_dd:.2%}",
          "Hit rate":           f"{hit_rate:.2%}",
          "N periods":          n_periods,
      }

def plot_backtest(returns_df, rolling_window=12, rebalance_days=5):
    r   = returns_df["portfolio_return"]
    cum = (1 + r).cumprod()

    # Log cumulative return
    log_cum  = np.log(cum)

    # Drawdown
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max

    # Rolling Sharpe (annualised)
    periods_per_year = 252 / rebalance_days
    roll_mean = r.rolling(rolling_window).mean()
    roll_std  = r.rolling(rolling_window).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(periods_per_year)

    # Rolling volatility (annualised)
    roll_vol = roll_std * np.sqrt(periods_per_year)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Walk-Forward Backtest Results", fontsize=13)

    # 1) Log cumulative return
    ax = axes[0, 0]
    ax.plot(returns_df["date"], log_cum, color="steelblue", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Log Cumulative Return")
    ax.set_ylabel("log(1 + cumulative return)")

    # 2) Drawdown
    ax = axes[0, 1]
    ax.fill_between(returns_df["date"], drawdown, 0, color="tomato", alpha=0.6)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")

    # 3) Return distribution
    ax = axes[0, 2]
    ax.hist(r, bins=30, color="steelblue", edgecolor="white", density=True)
    ax.axvline(r.mean(), color="red",   linestyle="--", linewidth=1, label=f"Mean: {r.mean():.4f}")
    ax.axvline(0,        color="black", linestyle="--", linewidth=0.5)
    ax.set_title("Return Distribution")
    ax.set_xlabel("Period return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # 4) Rolling Sharpe
    ax = axes[1, 0]
    ax.plot(returns_df["date"], roll_sharpe, color="steelblue", linewidth=1.2)
    ax.axhline(0, color="black",  linewidth=0.5, linestyle="--")
    ax.axhline(1, color="green",  linewidth=0.5, linestyle="--", alpha=0.6)
    ax.axhline(-1, color="tomato", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_title(f"Rolling Sharpe ({rolling_window}-period)")
    ax.set_ylabel("Sharpe ratio")

    # 5) Rolling volatility
    ax = axes[1, 1]
    ax.plot(returns_df["date"], roll_vol, color="orange", linewidth=1.2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(f"Rolling Annualised Volatility ({rolling_window}-period)")
    ax.set_ylabel("Volatility")

    # 6) Number of positions over time
    ax = axes[1, 2]
    ax.plot(returns_df["date"], returns_df["n_positions"], color="steelblue", linewidth=1.2)
    ax.set_title("Number of Positions")
    ax.set_ylabel("N positions")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()

def print_progress(current, total, signal_date, port_ret=None):
      pct      = current / total
      bar_len  = 30
      filled   = int(bar_len * pct)
      bar      = "█" * filled + "░" * (bar_len - filled)
      ret_str  = f"  last return: {port_ret:+.4f}" if port_ret is not None else ""
      print(f"\r  [{bar}] {current}/{total}  {signal_date}{ret_str}   ", end="", flush=True)

def plot_model_diagnostics(diagnostics_df, rolling_window=12):
    d = diagnostics_df.copy()

    roll_ic     = d["ic_mean"].rolling(rolling_window).mean()
    cum_ic      = d["ic_mean"].cumsum()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Walk-Forward Model Diagnostics", fontsize=13)

    # 1) Per-period IC with rolling mean
    ax = axes[0, 0]
    ax.bar(d["date"], d["ic_mean"], color=["steelblue" if v >= 0 else "tomato" for v in d["ic_mean"]],
            alpha=0.5, width=5)
    ax.plot(d["date"], roll_ic, color="steelblue", linewidth=1.5,
            label=f"Rolling {rolling_window}-period mean")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Alpha Model — Per-Period IC")
    ax.set_ylabel("IC (Spearman rank corr)")
    ax.legend(fontsize=8)

    # 2) Cumulative IC
    ax = axes[0, 1]
    ax.plot(d["date"], cum_ic, color="steelblue", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Alpha Model — Cumulative IC")
    ax.set_ylabel("Cumulative IC")

    # 3) IC IR over time
    ax = axes[1, 0]
    roll_ir = (d["ic_mean"].rolling(rolling_window).mean()
                / d["ic_mean"].rolling(rolling_window).std())
    ax.plot(d["date"], roll_ir, color="steelblue", linewidth=1.2)
    ax.axhline(0,    color="black",  linewidth=0.5, linestyle="--")
    ax.axhline(0.5,  color="green",  linewidth=0.5, linestyle="--", alpha=0.6)
    ax.axhline(-0.5, color="tomato", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_title(f"Alpha Model — Rolling IC IR ({rolling_window}-period)")
    ax.set_ylabel("IC IR")

    # 4) Risk model predicted vol over time
    ax = axes[1, 1]
    ax.plot(d["date"], d["pred_vol"] * 100, color="orange", linewidth=1.2)
    ax.set_title("Risk Model — Predicted Portfolio Vol (ex-ante)")
    ax.set_ylabel("Predicted vol (%)")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    returns_df, weights_history, diagnostics_df = run_backtest(
        backtest_start="2013-01-06",
        backtest_end="2023-11-23",
        lookback_days=252,
        rebalance_days=5,
        horizon=5,
        quantile=0.05,
        lambda_risk=1.0,
        long_only=False,
        save_results=True,
    )

    print("\n── PERFORMANCE METRICS ──────────────────────────")
    for k, v in compute_metrics(returns_df, rebalance_days=5).items():
        print(f"  {k:<22}: {v}")

    plot_backtest(returns_df, rebalance_days=5)
    plot_model_diagnostics(diagnostics_df)