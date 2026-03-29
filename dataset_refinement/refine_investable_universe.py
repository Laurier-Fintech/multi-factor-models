import os
import duckdb

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path      = os.path.join(project_root, "new_factor_data.duckdb")


def refine_investable_universe(
    conn,
    min_price=16.48,
    min_market_cap=630_660_000,
    min_turnover=3.1702,
):
    """
    Refine crsp_clean to an investable universe by:
    1. Keeping only US domestic common stock CUSIPs
    2. Filtering out penny stocks, micro-caps, and illiquid names

    CUSIP filters applied:
    - First character must be a digit (0-9): removes foreign-domiciled issuers
        (G=Cayman, H=Ireland, Y=Bermuda, N=Netherlands, etc.)
    - Issue code (positions 7-8) must be '10' or '11': keeps only common stock,
        removes preferred (20), rights (30), warrants (40), mutual funds (50-88), ADRs

    Both the alpha model and risk model pipelines read from crsp_clean, so
    this should be run once before any factor or signal computation.

    Parameters
    ----------
    min_price      : minimum stock price — ABS(prc) >= min_price
    min_market_cap : minimum daily market cap in dollars
                    computed as ABS(prc) * shrout * 1000
    min_turnover   : minimum daily turnover in vol/shrout units
    """
    before_rows   = conn.execute("SELECT COUNT(*)             FROM crsp_clean").fetchone()[0]
    before_cusips = conn.execute("SELECT COUNT(DISTINCT cusip) FROM crsp_clean").fetchone()[0]

    print(f"\n  === PRE-FILTER ===")
    print(f"  Rows: {before_rows:,}   |   Unique cusips: {before_cusips:,}")

    # ── Stage 1: CUSIP-based equity filter ─────────────────────────────────────
    conn.execute("""
        CREATE OR REPLACE TABLE crsp_clean AS
        SELECT * FROM crsp_clean
        WHERE SUBSTRING(cusip, 1, 1) SIMILAR TO '[0-9]'
        AND SUBSTRING(cusip, 7, 2) IN ('10', '11')
    """)

    after_cusip_rows   = conn.execute("SELECT COUNT(*)             FROM crsp_clean").fetchone()[0]
    after_cusip_cusips = conn.execute("SELECT COUNT(DISTINCT cusip) FROM crsp_clean").fetchone()[0]

    print(f"\n  --- After CUSIP equity filter ---")
    print(f"  Rows:   {before_rows:,} → {after_cusip_rows:,}   ({before_rows - after_cusip_rows:,} removed)")
    print(f"  Cusips: {before_cusips:,} → {after_cusip_cusips:,}   ({before_cusips - after_cusip_cusips:,} removed)")
    print(f"  (Removed foreign domiciles [G/H/Y/N/...] and non-common-stock issue codes)")

    # ── Stage 2: Liquidity / size filters ──────────────────────────────────────
    dist = conn.execute("""
        SELECT
            PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY ABS(prc))                       AS p10_price,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ABS(prc))                       AS p25_price,
            PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY ABS(prc) * shrout * 1000)       AS p10_mktcap,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ABS(prc) * shrout * 1000)       AS p25_mktcap,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ABS(prc) * shrout * 1000)       AS p50_mktcap,
            PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY vol / NULLIF(shrout, 0))        AS p10_turnover,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY vol / NULLIF(shrout, 0))        AS p25_turnover
        FROM crsp_clean
        WHERE prc IS NOT NULL AND shrout IS NOT NULL AND vol IS NOT NULL
    """).fetch_df().iloc[0]

    print(f"\n  --- Post-CUSIP distribution (before size/liquidity cuts) ---")
    print(f"  Price    — p10: ${dist.p10_price:.2f}        p25: ${dist.p25_price:.2f}")
    print(f"  Mkt cap  — p10: ${dist.p10_mktcap:>15,.0f}   p25: ${dist.p25_mktcap:>15,.0f}   median: ${dist.p50_mktcap:>15,.0f}")
    print(f"  Turnover — p10: {dist.p10_turnover:.4f}      p25: {dist.p25_turnover:.4f}")

    conn.execute(f"""
        CREATE OR REPLACE TABLE crsp_clean AS
        SELECT
            cusip, date, permno, permco,
            prc, shrout, vol,
            ask, bid, askhi, bidlo,
            ret, retx
        FROM (
            SELECT
                *,
                ABS(prc) * shrout * 1000      AS _market_cap,
                vol / NULLIF(shrout, 0)        AS _turnover
            FROM crsp_clean
            WHERE prc    IS NOT NULL
            AND shrout IS NOT NULL
            AND vol    IS NOT NULL
        )
        WHERE ABS(prc)   >= {min_price}
        AND _market_cap >= {min_market_cap}
        AND _turnover   >= {min_turnover}
    """)

    after_rows   = conn.execute("SELECT COUNT(*)              FROM crsp_clean").fetchone()[0]
    after_cusips = conn.execute("SELECT COUNT(DISTINCT cusip) FROM crsp_clean").fetchone()[0]

    print(f"\n  === POST-FILTER (all stages) ===")
    print(f"  Rows:   {before_rows:,} → {after_rows:,}   ({before_rows - after_rows:,} removed, {(before_rows - after_rows)/before_rows:.1%})")
    print(f"  Cusips: {before_cusips:,} → {after_cusips:,}   ({before_cusips - after_cusips:,} removed)")


if __name__ == "__main__":
    conn = duckdb.connect(db_path)
    refine_investable_universe(conn)
    conn.close()