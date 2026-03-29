"""
build_signals.py

DuckDB-based alpha signal generation library.
"""
from __future__ import annotations
import os

from signal_lib.amihud        import compute_amihud_expr
from signal_lib.vol           import compute_volatility_expr
from signal_lib.momentum      import compute_momentum_expr
from signal_lib.momentum_med  import compute_momentum_med_expr
from signal_lib.reversal      import compute_reversal_expr
from signal_lib.spread        import compute_spread_expr
from signal_lib.turnover      import compute_turnover_expr
from signal_lib.high52w       import compute_high52w_expr
from signal_lib.max_ret       import compute_max_ret_expr
from signal_lib.rel_volume    import compute_rel_volume_expr
from signal_lib.downside_vol  import compute_downside_vol_expr
from signal_lib.range_vol     import compute_range_vol_expr

import duckdb
import pandas as pd


def add_cross_sectional_zscores(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    output_table: str,
    signal_cols: list[str],
) -> None:
    zscore_exprs = []
    for col in signal_cols:
        zscore_exprs.append(f"""
        CASE
            WHEN STDDEV_SAMP({col}) OVER (PARTITION BY date) IS NOT NULL
            AND STDDEV_SAMP({col}) OVER (PARTITION BY date) <> 0
            THEN (
                {col} - AVG({col}) OVER (PARTITION BY date)
            ) / STDDEV_SAMP({col}) OVER (PARTITION BY date)
            ELSE NULL
        END AS {col}_z
        """)

    zscore_sql = ",\n".join(zscore_exprs)

    con.execute(f"""
    CREATE OR REPLACE TABLE {output_table} AS
    SELECT
        *,
        {zscore_sql}
    FROM {source_table}
    """)


def build_signals(
    con: duckdb.DuckDBPyConnection,
    source_table: str = "crsp_clean",
    output_table: str = "alpha_signals",
    start_date: str | None = None,
    end_date: str | None = None,
    momentum_window: int = 20,
    momentum_skip_recent: int = 0,
    momentum_med_window: int = 120,
    momentum_med_skip: int = 5,
    reversal_window: int = 1,
    volatility_window: int = 20,
    downside_vol_window: int = 20,
    turnover_window: int = 20,
    amihud_window: int = 20,
    high52w_window: int = 252,
    max_ret_window: int = 20,
    rel_vol_short: int = 5,
    rel_vol_long: int = 60,
    skewness_window: int = 20,
    range_vol_window: int = 20,
    add_zscores: bool = True,
) -> None:

    date_filter = "1=1"
    if start_date and end_date:
        date_filter = f"date BETWEEN DATE '{start_date}' AND DATE '{end_date}'"
    elif start_date:
        date_filter = f"date >= DATE '{start_date}'"
    elif end_date:
        date_filter = f"date <= DATE '{end_date}'"

    momentum_expr     = compute_momentum_expr(window=momentum_window, skip_recent=momentum_skip_recent)
    momentum_med_expr = compute_momentum_med_expr(window=momentum_med_window, skip_recent=momentum_med_skip)
    reversal_expr     = compute_reversal_expr(window=reversal_window)
    volatility_expr   = compute_volatility_expr(window=volatility_window)
    downside_vol_expr = compute_downside_vol_expr(window=downside_vol_window)
    turnover_expr     = compute_turnover_expr(window=turnover_window)
    spread_expr       = compute_spread_expr()
    amihud_expr       = compute_amihud_expr(window=amihud_window)
    high52w_expr      = compute_high52w_expr(window=high52w_window)
    max_ret_expr      = compute_max_ret_expr(window=max_ret_window)
    rel_volume_expr   = compute_rel_volume_expr(short_window=rel_vol_short, long_window=rel_vol_long)
    range_vol_expr    = compute_range_vol_expr(window=range_vol_window)

    raw_table = f"{output_table}_raw"

    # ── Pass 1: all signals except skewness ───────────────────────────────────
    con.execute(f"""
    CREATE OR REPLACE TABLE {raw_table} AS
    WITH base AS (
        SELECT
            date, permno, permco, cusip,
            ret, retx, prc, vol, shrout,
            ask, bid, askhi, bidlo
        FROM {source_table}
        WHERE {date_filter}
    )
    SELECT
        date, permno, permco, cusip,
        ret, retx, prc, vol, shrout,
        ask, bid, askhi, bidlo,

        {momentum_expr}     AS momentum,
        {momentum_med_expr} AS momentum_med,
        {reversal_expr}     AS reversal,
        {volatility_expr}   AS volatility,
        {downside_vol_expr} AS downside_vol,
        {turnover_expr}     AS turnover,
        {spread_expr}       AS spread,
        {amihud_expr}       AS amihud_illiquidity,
        {high52w_expr}      AS high52w,
        {max_ret_expr}      AS max_ret,
        {rel_volume_expr}   AS rel_volume,
        {range_vol_expr}    AS range_vol

    FROM base
    """)

    # ── Pass 2: skewness via CTE (avoids nested window functions) ─────────────
    # Pre-compute the rolling mean first, then use it as a plain column
    # to compute E[(r - mu)^3] / sigma^3 in the outer query.
    w = skewness_window
    con.execute(f"""
    CREATE OR REPLACE TABLE {raw_table} AS
    WITH means AS (
        SELECT *,
            AVG(CASE WHEN ret IS NOT NULL THEN ret ELSE NULL END)
                OVER (PARTITION BY permno ORDER BY date
                    ROWS BETWEEN {w} PRECEDING AND 1 PRECEDING) AS _ret_mean
        FROM {raw_table}
    )
    SELECT
        * EXCLUDE (_ret_mean),
        AVG(
            POWER(
                CASE WHEN ret IS NOT NULL THEN ret - _ret_mean ELSE NULL END,
                3
            )
        ) OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN {w} PRECEDING AND 1 PRECEDING)
        / NULLIF(
            POWER(
                STDDEV_SAMP(CASE WHEN ret IS NOT NULL THEN ret ELSE NULL END)
                    OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN {w} PRECEDING AND 1 PRECEDING),
                3
            ),
        0) AS skewness
    FROM means
    """)

    signal_cols = [
        "momentum",
        "momentum_med",
        "reversal",
        "volatility",
        "downside_vol",
        "turnover",
        "spread",
        "amihud_illiquidity",
        "high52w",
        "max_ret",
        "rel_volume",
        "skewness",
        "range_vol",
    ]

    if add_zscores:
        add_cross_sectional_zscores(
            con=con,
            source_table=raw_table,
            output_table=output_table,
            signal_cols=signal_cols,
        )
    else:
        con.execute(f"CREATE OR REPLACE TABLE {output_table} AS SELECT * FROM {raw_table}")


def preview_signals(
    con: duckdb.DuckDBPyConnection,
    table_name: str = "alpha_signals",
    limit: int = 10,
):
    return con.execute(f"""
        SELECT *
        FROM {table_name}
        ORDER BY date DESC, permno
        LIMIT {limit}
    """).df()


def cleanup_signal_tables(con: duckdb.DuckDBPyConnection, tables_to_remove: list[str]) -> None:
    existing = set(con.execute("SHOW TABLES").df()["name"].tolist())
    for table in tables_to_remove:
        if table in existing:
            con.execute(f"DROP TABLE {table}")
            print(f"Dropped: {table}")
        else:
            print(f"Skipped (not found): {table}")


if __name__ == "__main__":
    print("Connecting to DB...")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    conn = duckdb.connect(os.path.join(project_root, "new_factor_data.duckdb"))
    print("Connected.")

    build_signals(con=conn, output_table="alpha_signals")

    print(preview_signals(con=conn))

    cleanup_signal_tables(conn, ["alpha_signals_raw"])
    conn.close()
    print("Done.")