from pathlib import Path
import duckdb
import pandas as pd
import os

DB_PATH = Path(__file__).parent / "crsp_dataset.duckdb"
# Use repo-level outputs directory (moved location)
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_connection(read_only: bool = True):
    return duckdb.connect(database=str(DB_PATH), read_only=read_only)

def list_tables(con=None):
    close = False
    if con is None:
        con = get_connection()
        close = True
    try:
        df = con.execute("SHOW TABLES").df()
        return df
    finally:
        if close:
            con.close()

def run_sample_query(sql: str, con=None) -> pd.DataFrame:
    close = False
    if con is None:
        con = get_connection()
        close = True
    try:
        return con.execute(sql).df()
    finally:
        if close:
            con.close()

def _write_txt(df: pd.DataFrame, filename: str):
    """Write a DataFrame to a readable txt file in OUTPUT_DIR."""
    p = OUTPUT_DIR / filename
    # Pretty text for quick inspection
    p.write_text(df.to_string(index=False))
    return p

def _write_text_block(text: str, filename: str):
    p = OUTPUT_DIR / filename
    p.write_text(text)
    return p

def _main_print_examples():
    # kept for API compatibility; prefer write_examples_to_file for script usage
    write_examples_to_file(Path(__file__).parent / "crsp_query_output.txt")

# -----------------------------
# New: write multiple query outputs to separate files
# -----------------------------
def write_queries_to_files():
    if not DB_PATH.exists():
        raise FileNotFoundError("DuckDB file not found. Run create_crsp_database.py first.")

    con = get_connection()
    try:
        # Get the list of all unique CUSIP codes
        cusip_codes = run_sample_query("SELECT DISTINCT cusip FROM crsp_table", con)
        _write_txt(cusip_codes, "unique_cusip_codes.txt")

        

        # Monthly universe summary with simple market cap stats
        # - ME = abs(prc) * shrout * 1000
        # - month_end snapshot: last trading day each month per permno
        universe_summary = run_sample_query("""
            WITH base AS (
                SELECT
                    permno,
                    date::date AS date,
                    ABS(prc) AS prc_abs,
                    shrout
                FROM crsp_table
                WHERE prc IS NOT NULL AND shrout IS NOT NULL AND shrout > 0
            ),
            pe AS (
                SELECT
                    permno,
                    DATE_TRUNC('month', date) AS m,
                    date,
                    prc_abs,
                    shrout,
                    ROW_NUMBER() OVER (PARTITION BY permno, DATE_TRUNC('month', date) ORDER BY date DESC) AS rn
                FROM base
            ),
            month_end AS (
                SELECT permno, m AS date, prc_abs, shrout
                FROM pe
                WHERE rn = 1
            ),
            month_me AS (
                SELECT
                    date,
                    permno,
                    prc_abs * shrout * 1000.0 AS ME
                FROM month_end
                WHERE prc_abs > 0 AND shrout > 0
            )
            SELECT
                date,
                COUNT(*) AS n_names,
                SUM(ME) AS total_me,
                MEDIAN(ME) AS median_me
            FROM month_me
            GROUP BY 1
            ORDER BY 1;
        """, con)
        _write_txt(universe_summary, "universe_summary_by_month.txt")

        # Factor starter pack for the latest month:
        #    Size (lnME), Momentum 12-1, 1M Reversal, Amihud, Turnover, plus z-scores.
        factors_latest = run_sample_query("""
            -- Base daily with helper columns
            WITH d AS (
                SELECT
                    permno,
                    date::date AS date,
                    ABS(prc) AS prc_abs,
                    shrout,
                    ret,
                    retx,
                    vol,
                    CASE WHEN vol IS NOT NULL AND vol > 0 THEN ABS(prc) * vol ELSE NULL END AS dollar_vol,
                    bid,
                    ask
                FROM crsp_table
            ),

            -- Month end snapshot for ME and shrout
            pe AS (
                SELECT
                    permno,
                    DATE_TRUNC('month', date) AS m,
                    date,
                    prc_abs,
                    shrout,
                    ROW_NUMBER() OVER (PARTITION BY permno, DATE_TRUNC('month', date) ORDER BY date DESC) AS rn
                FROM d
            ),
            mend AS (
                SELECT permno, m AS date, prc_abs, shrout
                FROM pe
                WHERE rn = 1
            ),

            -- Monthly returns from daily compounding: ret_m = product(1+ret) - 1
            daily_ret AS (
                SELECT permno, date, ret
                FROM d
                WHERE ret IS NOT NULL
            ),
            monthly_ret AS (
                SELECT
                    permno,
                    DATE_TRUNC('month', date) AS date,
                    EXP(SUM(LN(1 + ret))) - 1 AS ret_m
                FROM daily_ret
                GROUP BY 1, 2
            ),

            -- Amihud per month: mean( |ret| / dollar_vol ) from daily
            amihud_m AS (
                SELECT
                    permno,
                    DATE_TRUNC('month', date) AS date,
                    AVG(CASE WHEN dollar_vol IS NOT NULL AND dollar_vol > 0
                             THEN ABS(ret) / dollar_vol
                             ELSE NULL END) AS amihud
                FROM d
                GROUP BY 1, 2
            ),

            -- Turnover per month: sum(daily vol) / month-end shrout
            vol_m AS (
                SELECT permno,
                       DATE_TRUNC('month', date) AS date,
                       SUM(vol) AS vol_month
                FROM d
                GROUP BY 1, 2
            ),

            -- Assemble monthly panel
            panel_m AS (
                SELECT
                    me.date,
                    me.permno,
                    me.prc_abs,
                    me.shrout,
                    (me.prc_abs * me.shrout * 1000.0) AS ME,
                    mr.ret_m,
                    ah.amihud,
                    vm.vol_month
                FROM mend me
                LEFT JOIN monthly_ret mr USING (permno, date)
                LEFT JOIN amihud_m ah USING (permno, date)
                LEFT JOIN vol_m vm USING (permno, date)
            ),

            -- Momentum 12-1 and 1M reversal using monthly returns
            panel_with_mom AS (
                SELECT
                    *,
                    -- 1M reversal
                    LAG(ret_m, 1) OVER (PARTITION BY permno ORDER BY date) AS rev_1m,
                    -- Momentum 12-1: exp(sum log(1+ret_m) from t-12 to t-2) - 1
                    EXP(
                        SUM(CASE WHEN ret_m IS NOT NULL THEN LN(1 + ret_m) ELSE 0 END)
                        OVER (PARTITION BY permno ORDER BY date
                              ROWS BETWEEN 12 PRECEDING AND 2 PRECEDING)
                    ) - 1 AS mom_12_1
                FROM panel_m
            ),

            -- Compute turnover: sum of daily volume in month divided by month-end shrout
            panel_final AS (
                SELECT
                    date,
                    permno,
                    ME,
                    CASE WHEN ME > 0 THEN LN(ME) ELSE NULL END AS size_lnME,
                    ret_m,
                    mom_12_1,
                    rev_1m,
                    amihud,
                    CASE WHEN shrout IS NOT NULL AND shrout > 0 AND vol_month IS NOT NULL
                         THEN vol_month / shrout
                         ELSE NULL END AS turnover
                FROM panel_with_mom
            ),

            -- Pick latest month available
            latest_month AS (
                SELECT MAX(date) AS date FROM panel_final
            ),

            latest_panel AS (
                SELECT pf.*
                FROM panel_final pf
                JOIN latest_month lm ON pf.date = lm.date
            ),

            -- Cross-section z-scores for the latest month
            stats AS (
                SELECT
                    AVG(size_lnME) AS m_size,
                    STDDEV_POP(size_lnME) AS s_size,
                    AVG(mom_12_1) AS m_mom,
                    STDDEV_POP(mom_12_1) AS s_mom,
                    AVG(rev_1m) AS m_rev,
                    STDDEV_POP(rev_1m) AS s_rev,
                    AVG(amihud) AS m_ami,
                    STDDEV_POP(amihud) AS s_ami,
                    AVG(turnover) AS m_turn,
                    STDDEV_POP(turnover) AS s_turn
                FROM latest_panel
            )

            SELECT
                lp.date,
                lp.permno,
                lp.ME,
                lp.size_lnME,
                lp.ret_m,
                lp.mom_12_1,
                lp.rev_1m,
                lp.amihud,
                lp.turnover,
                -- z-scores with 3-sigma clipping done later in Python if needed
                (lp.size_lnME - s.m_size)/NULLIF(s.s_size,0) AS size_lnME_z,
                (lp.mom_12_1 - s.m_mom)/NULLIF(s.s_mom,0) AS mom_12_1_z,
                (lp.rev_1m - s.m_rev)/NULLIF(s.s_rev,0) AS rev_1m_z,
                (lp.amihud - s.m_ami)/NULLIF(s.s_ami,0) AS amihud_z,
                (lp.turnover - s.m_turn)/NULLIF(s.s_turn,0) AS turnover_z
            FROM latest_panel lp
            CROSS JOIN stats s
            ORDER BY lp.permno;
        """, con)

        # Write latest month factors with YYYYMM stamp
        if not factors_latest.empty:
            latest_dt = pd.to_datetime(factors_latest['date'].iloc[0])
            ym = latest_dt.strftime("%Y%m")
            _write_txt(factors_latest, f"factors_month_{ym}.txt")
        else:
            _write_text_block("No data found for factors_latest query.\n", "factors_month_latest.txt")

    finally:
        con.close()

if __name__ == "__main__":
    # Old behavior
    # _main_print_examples()

    # New behavior - create many separate files
    write_queries_to_files()
    print(f"Wrote outputs to: {OUTPUT_DIR.resolve()}")
