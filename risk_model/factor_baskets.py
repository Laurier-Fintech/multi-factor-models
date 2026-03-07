import os
import duckdb

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

new_db_path = os.path.join(project_root, "new_factor_data.duckdb")

def creating_factor_baskets(conn):
    conn.execute("""DROP TABLE IF EXISTS factor_basket_returns;

    CREATE TABLE factor_basket_returns AS
    WITH base AS (
        SELECT
            *,
            lead(retx)  OVER (PARTITION BY cusip ORDER BY date) AS retx_fwd1,
            lead(date) OVER (PARTITION BY cusip ORDER BY date) AS date_fwd1
        FROM daily_factors_table_clean
    ),
    ranked AS (
        SELECT
            date,
            date_fwd1,
            cusip,
            retx_fwd1,

            ntile(10) OVER (PARTITION BY date ORDER BY momentum_12m_1m)         AS dec_mom,
            ntile(10) OVER (PARTITION BY date ORDER BY avg_turnover_12m_1m)     AS dec_tov,
            ntile(10) OVER (PARTITION BY date ORDER BY avg_market_cap_12m_1m)   AS dec_mcap,
            ntile(10) OVER (PARTITION BY date ORDER BY avg_bid_ask_spread_12m_1m) AS dec_spread
        FROM base
        WHERE retx_fwd1 IS NOT NULL
        AND date_fwd1 IS NOT NULL
    )
    SELECT
        date_fwd1 AS date,

        avg(CASE WHEN dec_mom  = 10 THEN retx_fwd1 END) AS high_momentum,
        avg(CASE WHEN dec_mom  =  1 THEN retx_fwd1 END) AS low_momentum,

        avg(CASE WHEN dec_tov  = 10 THEN retx_fwd1 END) AS high_turnover,
        avg(CASE WHEN dec_tov  =  1 THEN retx_fwd1 END) AS low_turnover,

        avg(CASE WHEN dec_mcap = 10 THEN retx_fwd1 END) AS mega_cap,
        avg(CASE WHEN dec_mcap =  1 THEN retx_fwd1 END) AS micro_cap,

        -- tightest spreads = most liquid
        avg(CASE WHEN dec_spread =  1 THEN retx_fwd1 END) AS most_liquid,
        avg(CASE WHEN dec_spread = 10 THEN retx_fwd1 END) AS least_liquid

    FROM ranked
    GROUP BY date_fwd1
    ORDER BY date_fwd1;""")

    df = conn.execute("""SELECT * FROM factor_basket_returns
    ORDER BY date ASC;""").fetch_df()
    return df
    

def create_ls_factor_baskets(conn):
    conn.execute("""DROP TABLE IF EXISTS factor_ls_returns;

    CREATE TABLE factor_ls_returns AS
    SELECT
        date,
        (high_momentum - low_momentum)   AS momentum_basket,
        (high_turnover - low_turnover)   AS turnover_basket,
        (mega_cap - micro_cap)           AS size_basket,
        (most_liquid - least_liquid)     AS spread_basket
    FROM factor_basket_returns
    ORDER BY date;""")

    df = conn.execute("""SELECT * FROM factor_ls_returns
    ORDER BY date ASC;""").fetch_df()
    return df

if __name__ == "__main__":
    conn = duckdb.connect(new_db_path)
    df = creating_factor_baskets(conn)
    print(df)
    df = create_ls_factor_baskets(conn)
    print(df)

    conn.close()