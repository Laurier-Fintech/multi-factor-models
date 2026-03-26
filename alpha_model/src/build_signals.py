"""
signal_lib.py

DuckDB-based alpha signal generation library.

This module computes trailing alpha signals directly inside DuckDB tables
instead of materializing massive pandas DataFrames.

Expected source table columns:
    date, permno, ret, prc, vol, shrout, ask, bid

Main entry point:
    build_signals(con, source_table, output_table, ...)

Outputs:
    DuckDB table with one row per (date, permno) containing:
        - raw signals
        - optionally z-scored signals
"""
from __future__ import annotations

from signal_lib.amihud import compute_amihud_expr
from signal_lib.vol import compute_volatility_expr
from signal_lib.momentum import compute_momentum_expr
from signal_lib.reversal import compute_reversal_expr
from signal_lib.spread import compute_spread_expr
from signal_lib.turnover import compute_turnover_expr 

import duckdb
import pandas as pd


def add_cross_sectional_zscores(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    output_table: str,
    signal_cols: list[str],
) -> None:
    """
    Create a new table with cross-sectional z-scores by date for each signal.
    """
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
    reversal_window: int = 1,
    volatility_window: int = 20,
    turnover_window: int = 20,
    amihud_window: int = 20,
    add_zscores: bool = True,
) -> None:
    """
    Build all alpha signals directly in DuckDB and write to a table.

    Parameters
    ----------
    con : duckdb connection
    source_table : input clean CRSP-style table
    output_table : output signal table
    start_date / end_date : optional date filter
    add_zscores : whether to create date-wise standardized versions
    """

    date_filter = "1=1"
    if start_date and end_date:
        date_filter = f"date BETWEEN DATE '{start_date}' AND DATE '{end_date}'"
    elif start_date:
        date_filter = f"date >= DATE '{start_date}'"
    elif end_date:
        date_filter = f"date <= DATE '{end_date}'"

    momentum_expr = compute_momentum_expr(
        window=momentum_window,
        skip_recent=momentum_skip_recent
    )
    reversal_expr = compute_reversal_expr(window=reversal_window)
    volatility_expr = compute_volatility_expr(window=volatility_window)
    turnover_expr = compute_turnover_expr(window=turnover_window)
    spread_expr = compute_spread_expr()
    amihud_expr = compute_amihud_expr(window=amihud_window)

    raw_table = f"{output_table}_raw"

    con.execute(f"""
    CREATE OR REPLACE TABLE {raw_table} AS
    WITH base AS (
        SELECT
            date,
            permno,
            permco,
            cusip,
            ret,
            prc,
            vol,
            shrout,
            ask,
            bid
        FROM {source_table}
        WHERE {date_filter}
    )
    SELECT
        date,
        permno,
        permco,
        cusip,
        ret,
        prc,
        vol,
        shrout,
        ask,
        bid,

        {momentum_expr}   AS momentum,
        {reversal_expr}   AS reversal,
        {volatility_expr} AS volatility,
        {turnover_expr}   AS turnover,
        {spread_expr}     AS spread,
        {amihud_expr}     AS amihud_illiquidity

    FROM base
    """)

    signal_cols = [
        "momentum",
        "reversal",
        "volatility",
        "turnover",
        "spread",
        "amihud_illiquidity",
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
    """
    Convenience preview for debugging.
    """
    return con.execute(f"""
        SELECT *
        FROM {table_name}
        ORDER BY date DESC, permno
        LIMIT {limit}
    """).df()

def test_signal_building():
    print("Building Signals...")
    build_signals(con=conn,output_table="alpha_signals_test1")
    print("Built Signals.")


if __name__ == "__main__":
    print("Attempting to connect to DB.")
    new_db_path = r"C:\Users\agand\Desktop\Personal_Coding\Finance_Projects\MultiFactorModel_LFQA\multi-factor-models\venv\crsp\new_factor_data.duckdb"
    conn = duckdb.connect(new_db_path)
    print("Connected to DB")
    tables = pd.DataFrame(conn.execute("SHOW TABLES;").fetch_df())
    print(tables)

    
    test_signal_building()

    table_name = "alpha_signals_test1_raw"
    print(f"Displaying preview of {table_name}...")
    print(preview_signals(con=conn, table_name=table_name))
    print("Done.")

    print("------------------------------------------------------------------------------------------------------")

    table_name = "alpha_signals_test1"
    print(f"Displaying preview of {table_name}...")
    print(preview_signals(con=conn, table_name=table_name))
    print("Done.")
    

    conn.close()
    print("Connection Closed.")



