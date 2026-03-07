import os
import duckdb
from pathlib import Path
import pandas as pd
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

new_db_path = os.path.join(project_root, "new_factor_data.duckdb")
CUSIP_CAPIQ_PATH = os.path.join(project_root, "risk_model", "cusip_CAPIQ_data.csv")


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

def generate_sector_factors(conn):
    # 1) Load CapIQ mapping into DuckDB
    ticker_convert_df = pd.read_csv(CUSIP_CAPIQ_PATH, encoding="latin1", low_memory=False)

    conn.execute("DROP TABLE IF EXISTS capiq_data;")
    conn.execute("""
        CREATE TABLE capiq_data AS
        SELECT
            TRIM(Cusip) AS Cusip,
            industry
        FROM ticker_convert_df
        WHERE Cusip IS NOT NULL
    """)

    # 2) Build a joined version (avoid self-referencing CREATE OR REPLACE)
    conn.execute("DROP TABLE IF EXISTS daily_factors_table_with_industry;")
    conn.execute("""
        CREATE TABLE daily_factors_table_with_industry AS
        SELECT
            d.*,
            m.industry
        FROM daily_factors_table d
        LEFT JOIN (
            -- If CapIQ has duplicates per Cusip, collapse to one row per Cusip.
            -- If CapIQ Cusip is 9 chars and CRSP cusip is 8, LEFT(..., 8) fixes it.
            SELECT
                LEFT(TRIM(Cusip), 8) AS cusip8,
                ANY_VALUE(industry) AS industry
            FROM capiq_data
            WHERE industry IS NOT NULL
            GROUP BY 1
        ) m
        ON d.cusip = m.cusip8
        ORDER BY d.cusip, d.date
    """)

    # 3) Overwrite daily_factors_table
    conn.execute("DROP TABLE daily_factors_table;")
    conn.execute("ALTER TABLE daily_factors_table_with_industry RENAME TO daily_factors_table;")
    
    # Convert the industries into GICS sectors
    conn.execute("""
    UPDATE daily_factors_table
    SET industry =
    CASE
    WHEN industry IS NULL OR TRIM(industry) = '' OR TRIM(industry) = '0' THEN NULL

    WHEN TRIM(industry) IN (
        'Energy Equipment and Services',
        'Oil, Gas and Consumable Fuels'
    ) THEN 'Energy'

    WHEN TRIM(industry) IN (
        'Chemicals',
        'Construction Materials',
        'Containers and Packaging',
        'Metals and Mining',
        'Paper and Forest Products'
    ) THEN 'Materials'

    WHEN TRIM(industry) IN (
        'Aerospace and Defense',
        'Air Freight and Logistics',
        'Building Products',
        'Commercial Services and Supplies',
        'Construction and Engineering',
        'Distributors',
        'Electrical Equipment',
        'Ground Transportation',
        'Industrial Conglomerates',
        'Machinery',
        'Marine Transportation',
        'Passenger Airlines',
        'Professional Services',
        'Trading Companies and Distributors',
        'Transportation Infrastructure'
    ) THEN 'Industrials'

    WHEN TRIM(industry) IN (
        'Automobile Components',
        'Automobiles',
        'Broadline Retail',
        'Diversified Consumer Services',
        'Hotels, Restaurants and Leisure',
        'Household Durables',
        'Leisure Products',
        'Specialty Retail',
        'Textiles, Apparel and Luxury Goods'
    ) THEN 'Consumer Discretionary'

    WHEN TRIM(industry) IN (
        'Beverages',
        'Consumer Staples Distribution and Retail',
        'Food Products',
        'Household Products',
        'Personal Care Products',
        'Tobacco'
    ) THEN 'Consumer Staples'

    WHEN TRIM(industry) IN (
        'Biotechnology',
        'Health Care Equipment and Supplies',
        'Health Care Providers and Services',
        'Health Care Technology',
        'Life Sciences Tools and Services',
        'Pharmaceuticals'
    ) THEN 'Health Care'

    WHEN TRIM(industry) IN (
        'Banks',
        'Capital Markets',
        'Consumer Finance',
        'Financial Services',
        'Insurance'
    ) THEN 'Financials'

    WHEN TRIM(industry) IN (
        'Communications Equipment',
        'Electronic Equipment, Instruments and Components',
        'IT Services',
        'Semiconductors and Semiconductor Equipment',
        'Software',
        'Technology Hardware, Storage and Peripherals'
    ) THEN 'Information Technology'

    WHEN TRIM(industry) IN (
        'Diversified Telecommunication Services',
        'Entertainment',
        'Interactive Media and Services',
        'Media',
        'Wireless Telecommunication Services'
    ) THEN 'Communication Services'

    WHEN TRIM(industry) IN (
        'Electric Utilities',
        'Gas Utilities',
        'Independent Power and Renewable Electricity Producers',
        'Multi-Utilities',
        'Water Utilities'
    ) THEN 'Utilities'

    WHEN TRIM(industry) IN (
        'Diversified REITs',
        'Health Care REITs',
        'Hotel and Resort REITs',
        'Industrial REITs',
        'Mortgage Real Estate Investment Trusts (REITs)',
        'Office REITs',
        'Real Estate Management and Development',
        'Residential REITs',
        'Retail REITs',
        'Specialized REITs'
    ) THEN 'Real Estate'

    ELSE NULL
    END;
    """)


    # quick sanity check
    return conn.execute("""
        SELECT cusip, date, market_cap, turnover, bid_ask_spread, industry
        FROM daily_factors_table
        ORDER BY date ASC, cusip
        LIMIT 5
    """).fetch_df()


    


def generate_cusip_sector_table(conn):
    conn.execute("""
        DROP TABLE IF EXISTS cusip_sector_table;
        CREATE TABLE cusip_sector_table AS
        SELECT DISTINCT cusip, industry
        FROM daily_factors_table
        WHERE industry IS NOT NULL;
    """)
    return conn.execute("SELECT * FROM cusip_sector_table").fetch_df()

def calculate_rolling_factors(conn):
    conn.execute("""
    DROP TABLE IF EXISTS daily_factors_table;
    CREATE TABLE daily_factors_table AS
    WITH industry_map AS (
        SELECT cusip, industry
        FROM cusip_sector_table
    ),
    base AS (
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

            im.industry,

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

            CASE
                WHEN c.ret IS NULL OR c.ret <= -1 THEN NULL
                ELSE ln(1.0 + c.ret)
            END AS logret
        FROM crsp_clean c
        LEFT JOIN industry_map im
          ON c.cusip = im.cusip
    ),
    roll AS (
        SELECT
            b.*,

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
        industry,

        CASE
            WHEN nobs_mom_12m_1m = 232 THEN exp(sum_logret_12m_1m) - 1.0
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
          AND avg_bid_ask_spread_12m_1m IS NOT NULL;
    """)

    return conn.execute("""
        SELECT *
        FROM daily_factors_table_clean
        ORDER BY date ASC, cusip
        LIMIT 5
    """).fetch_df()







if __name__ == "__main__":
    conn = duckdb.connect(new_db_path)
    df = generate_factors(conn)
    print(df)
    df = generate_sector_factors(conn)
    print(df)
    cusip_sector_df = generate_cusip_sector_table(conn)
    print(cusip_sector_df)
    df = calculate_rolling_factors(conn)
    print(df)
    conn.close()