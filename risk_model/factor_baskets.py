import os
import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

new_db_path = os.path.join(project_root, "new_factor_data.duckdb")

def creating_factor_baskets(conn):
    conn.execute("""DROP TABLE IF EXISTS factor_basket_returns;

    CREATE TABLE factor_basket_returns AS
    WITH base AS (
        SELECT
            *,
            lead(retx) OVER (PARTITION BY cusip ORDER BY date) AS retx_fwd1,
            lead(date)  OVER (PARTITION BY cusip ORDER BY date) AS date_fwd1
        FROM daily_factors_table_clean
    ),
    ranked AS (
        SELECT
            date,
            date_fwd1,
            cusip,
            retx_fwd1,

            ntile(10) OVER (PARTITION BY date ORDER BY momentum_12m_1m)           AS dec_mom,
            ntile(10) OVER (PARTITION BY date ORDER BY avg_turnover_12m_1m)       AS dec_tov,
            ntile(10) OVER (PARTITION BY date ORDER BY avg_market_cap_12m_1m)     AS dec_mcap,
            ntile(10) OVER (PARTITION BY date ORDER BY avg_bid_ask_spread_12m_1m) AS dec_spread,
            ntile(10) OVER (PARTITION BY date ORDER BY avg_volatility_12m_1m)     AS dec_vol,
            ntile(10) OVER (PARTITION BY date ORDER BY momentum_1m)               AS dec_rev
        FROM base
        WHERE retx_fwd1 IS NOT NULL
        AND date_fwd1 IS NOT NULL
    )
    SELECT
        date_fwd1 AS date,

        avg(CASE WHEN dec_mom    = 10 THEN retx_fwd1 END) AS high_momentum,
        avg(CASE WHEN dec_mom    =  1 THEN retx_fwd1 END) AS low_momentum,

        avg(CASE WHEN dec_tov    = 10 THEN retx_fwd1 END) AS high_turnover,
        avg(CASE WHEN dec_tov    =  1 THEN retx_fwd1 END) AS low_turnover,

        avg(CASE WHEN dec_mcap   = 10 THEN retx_fwd1 END) AS mega_cap,
        avg(CASE WHEN dec_mcap   =  1 THEN retx_fwd1 END) AS micro_cap,

        avg(CASE WHEN dec_spread =  1 THEN retx_fwd1 END) AS most_liquid,
        avg(CASE WHEN dec_spread = 10 THEN retx_fwd1 END) AS least_liquid,

        avg(CASE WHEN dec_vol    = 10 THEN retx_fwd1 END) AS high_vol,
        avg(CASE WHEN dec_vol    =  1 THEN retx_fwd1 END) AS low_vol,

        -- low reversal = recent losers (expected to rebound), dec 1 = short-term losers
        avg(CASE WHEN dec_rev    = 10 THEN retx_fwd1 END) AS high_reversal,
        avg(CASE WHEN dec_rev    =  1 THEN retx_fwd1 END) AS low_reversal

    FROM ranked
    GROUP BY date_fwd1
    ORDER BY date_fwd1;""")

    df = conn.execute("""SELECT * FROM factor_basket_returns ORDER BY date ASC;""").fetch_df()
    return df



def create_ls_factor_baskets(conn):
    conn.execute("""DROP TABLE IF EXISTS factor_ls_returns;

    CREATE TABLE factor_ls_returns AS
    SELECT
        date,
        (high_momentum  - low_momentum)  AS momentum_basket,
        (high_turnover  - low_turnover)  AS turnover_basket,
        (mega_cap       - micro_cap)     AS size_basket,
        (most_liquid    - least_liquid)  AS spread_basket,
        (high_vol       - low_vol)       AS vol_basket,
        -- reversal: short recent winners (dec 10), long recent losers (dec 1)
        (low_reversal   - high_reversal) AS reversal_basket
    FROM factor_basket_returns
    ORDER BY date;""")

    df = conn.execute("""SELECT * FROM factor_ls_returns ORDER BY date ASC;""").fetch_df()
    return df


def creating_sector_factor_baskets(conn):
      """Append sector basket return columns to factor_ls_returns."""
      conn.execute("""
      CREATE OR REPLACE TABLE factor_ls_returns AS
      WITH base AS (
          SELECT
              *,
              lead(retx) OVER (PARTITION BY cusip ORDER BY date) AS retx_fwd1,
              lead(date)  OVER (PARTITION BY cusip ORDER BY date) AS date_fwd1
          FROM daily_factors_table_clean
      ),
      sectors AS (
          SELECT
              date_fwd1 AS date,
              avg(CASE WHEN industry = 'Communication Services' THEN retx_fwd1 END) AS communication_services,
              avg(CASE WHEN industry = 'Consumer Discretionary' THEN retx_fwd1 END) AS consumer_discretionary,
              avg(CASE WHEN industry = 'Consumer Staples'       THEN retx_fwd1 END) AS consumer_staples,
              avg(CASE WHEN industry = 'Energy'                 THEN retx_fwd1 END) AS energy,
              avg(CASE WHEN industry = 'Financials'             THEN retx_fwd1 END) AS financials,
              avg(CASE WHEN industry = 'Health Care'            THEN retx_fwd1 END) AS health_care,
              avg(CASE WHEN industry = 'Industrials'            THEN retx_fwd1 END) AS industrials,
              avg(CASE WHEN industry = 'Information Technology' THEN retx_fwd1 END) AS information_technology,
              avg(CASE WHEN industry = 'Materials'              THEN retx_fwd1 END) AS materials,
              avg(CASE WHEN industry = 'Real Estate'            THEN retx_fwd1 END) AS real_estate,
              avg(CASE WHEN industry = 'Utilities'              THEN retx_fwd1 END) AS utilities
          FROM base
          WHERE retx_fwd1 IS NOT NULL
            AND date_fwd1 IS NOT NULL
          GROUP BY date_fwd1
      )
      SELECT
          f.*,
          s.communication_services,
          s.consumer_discretionary,
          s.consumer_staples,
          s.energy,
          s.financials,
          s.health_care,
          s.industrials,
          s.information_technology,
          s.materials,
          s.real_estate,
          s.utilities
      FROM factor_ls_returns f
      LEFT JOIN sectors s ON f.date = s.date
      ORDER BY f.date;""")

      df = conn.execute("SELECT * FROM factor_ls_returns ORDER BY date ASC;").fetch_df()
      return df

def plot_factor_returns(df):
    factor_cols = [
        'momentum_basket', 'turnover_basket', 'size_basket',
        'spread_basket', 'vol_basket', 'reversal_basket'
    ]
    sector_cols = [
        'communication_services', 'consumer_discretionary', 'consumer_staples',
        'energy', 'financials', 'health_care', 'industrials',
        'information_technology', 'materials', 'real_estate', 'utilities'
    ]

    cum = (1 + df.set_index('date')[factor_cols + sector_cols]).cumprod() - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for col in factor_cols:
        ax1.plot(cum.index, cum[col], label=col.replace('_', ' ').title())
    ax1.set_title('Factor Baskets — Cumulative Return')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')

    for col in sector_cols:
        ax2.plot(cum.index, cum[col], label=col.replace('_', ' ').title())
    ax2.set_title('Sector Baskets — Cumulative Return')
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    conn = duckdb.connect(new_db_path)
    df = creating_factor_baskets(conn)
    print(df)
    df = create_ls_factor_baskets(conn)
    print(df)
    df = creating_sector_factor_baskets(conn)
    print(df)
    plot_factor_returns(df)

    conn.close()