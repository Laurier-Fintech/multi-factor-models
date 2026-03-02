import os
import duckdb

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

new_db_path = os.path.join(project_root, "new_factor_data.duckdb")


def generate_factors(conn):
    conn.execute("""
    DROP TABLE IF EXISTS daily_factors_table;
    CREATE TABLE daily_factors_table AS
    SELECT
    c.cusip,
    c.date,
    c.permno,
    c.permco,
    c.prc,
    c.shrout,
    c.vol,
    c.ask,
    c.bid,
    c.askhi,
    c.bidlo,
    c.ret,
    c.retx,
    CASE WHEN c.prc IS NULL OR c.shrout IS NULL
        THEN NULL
        ELSE c.prc * c.shrout * 1000
    END AS market_cap,
        
    CASE WHEN c.vol IS NULL OR c.shrout IS NULL OR c.shrout = 0
        THEN NULL
        ELSE c.vol / c.shrout
        END AS turnover,

    CASE 
        WHEN c.ask IS NULL OR c.bid IS NULL OR (c.ask + c.bid ) = 0
        THEN NULL
        ELSE (c.ask - c.bid) / ((c.ask + c.bid) / 2.0)
    END AS bid_ask_spread

    FROM crsp_clean c
    ORDER BY c.cusip, c.date;
    """)

    df = conn.execute("""
    SELECT * FROM daily_factors_table
    ORDER BY date ASC, cusip
    LIMIT 5
    """).fetch_df()
    return df

def calculate_rolling_factors(conn):
    conn.execute("""
    DROP TABLE IF EXISTS daily_factors_table;
    CREATE TABLE daily_factors_table AS
    WITH base AS (
        SELECT
            c.cusip,
            c.date,
            c.permno,
            c.permco,
            c.prc,
            c.shrout,
            c.vol,
            c.ask,
            c.bid,
            c.askhi,
            c.bidlo,
            c.ret,
            c.retx,

            -- level features
            CASE WHEN c.prc IS NULL OR c.shrout IS NULL
                THEN NULL
                ELSE c.prc * c.shrout * 1000
            END AS market_cap,

            CASE WHEN c.vol IS NULL OR c.shrout IS NULL OR c.shrout = 0
                THEN NULL
                ELSE c.vol / c.shrout
            END AS turnover,

            CASE
                WHEN c.ask IS NULL OR c.bid IS NULL OR (c.ask + c.bid) = 0
                THEN NULL
                ELSE (c.ask - c.bid) / ((c.ask + c.bid) / 2.0)
            END AS bid_ask_spread,

            -- for momentum
            CASE
                WHEN c.ret IS NULL OR c.ret <= -1 THEN NULL
                ELSE ln(1.0 + c.ret)
            END AS logret
        FROM crsp_clean c
    ),
    roll AS (
        SELECT
            b.*,

            -- 12M-1M momentum window (daily): 252 trading days lookback, skip last 21 days
            sum(logret) OVER (
                PARTITION BY cusip
                ORDER BY date
                ROWS BETWEEN 252 PRECEDING AND 21 PRECEDING
            ) AS sum_logret_12m_1m,

            count(logret) OVER (
                PARTITION BY cusip
                ORDER BY date
                ROWS BETWEEN 252 PRECEDING AND 21 PRECEDING
            ) AS nobs_mom_12m_1m,

            avg(market_cap) OVER (
                PARTITION BY cusip
                ORDER BY date
                ROWS BETWEEN 252 PRECEDING AND 21 PRECEDING
            ) AS avg_market_cap_12m_1m,

            avg(turnover) OVER (
                PARTITION BY cusip
                ORDER BY date
                ROWS BETWEEN 252 PRECEDING AND 21 PRECEDING
            ) AS avg_turnover_12m_1m,

            avg(bid_ask_spread) OVER (
                PARTITION BY cusip
                ORDER BY date
                ROWS BETWEEN 252 PRECEDING AND 21 PRECEDING
            ) AS avg_bid_ask_spread_12m_1m,

            -- optional: require completeness for each rolling mean too
            count(market_cap) OVER (
                PARTITION BY cusip
                ORDER BY date
                ROWS BETWEEN 252 PRECEDING AND 21 PRECEDING
            ) AS nobs_mcap_12m_1m,

            count(turnover) OVER (
                PARTITION BY cusip
                ORDER BY date
                ROWS BETWEEN 252 PRECEDING AND 21 PRECEDING
            ) AS nobs_tov_12m_1m,

            count(bid_ask_spread) OVER (
                PARTITION BY cusip
                ORDER BY date
                ROWS BETWEEN 252 PRECEDING AND 21 PRECEDING
            ) AS nobs_bas_12m_1m

        FROM base b
    )
    SELECT
        cusip, date, permno, permco, prc, shrout, vol, ask, bid, askhi, bidlo, ret, retx,
        market_cap, turnover, bid_ask_spread,

        CASE
            -- window length is 252-21 = 232 observations when complete
            WHEN nobs_mom_12m_1m = 232
            THEN exp(sum_logret_12m_1m) - 1.0
            ELSE NULL
        END AS momentum_12m_1m,

        CASE WHEN nobs_mcap_12m_1m = 232 THEN avg_market_cap_12m_1m ELSE NULL END AS avg_market_cap_12m_1m,
        CASE WHEN nobs_tov_12m_1m  = 232 THEN avg_turnover_12m_1m  ELSE NULL END AS avg_turnover_12m_1m,
        CASE WHEN nobs_bas_12m_1m  = 232 THEN avg_bid_ask_spread_12m_1m ELSE NULL END AS avg_bid_ask_spread_12m_1m

    FROM roll
    ORDER BY cusip, date;
    """)

    conn.execute("""
        DROP TABLE IF EXISTS daily_factors_table_clean;

        CREATE TABLE daily_factors_table_clean AS
        SELECT *
        FROM daily_factors_table
        WHERE momentum_12m_1m IS NOT NULL
        AND avg_market_cap_12m_1m IS NOT NULL
        AND avg_turnover_12m_1m IS NOT NULL
        AND avg_bid_ask_spread_12m_1m IS NOT NULL;""")
    
    df = conn.execute("""
        SELECT *
        FROM daily_factors_table_clean
        ORDER BY date ASC, cusip
        LIMIT 5
    """).fetch_df()
    
    return df







if __name__ == "__main__":
    conn = duckdb.connect(new_db_path)
    df = generate_factors(conn)
    df = calculate_rolling_factors(conn)
    print(df)
    conn.close()