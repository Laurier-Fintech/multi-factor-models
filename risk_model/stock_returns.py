import os
import duckdb

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

new_db_path = os.path.join(project_root, "new_factor_data.duckdb")

def cusip_daily_returns(conn):
    conn.execute("""
    DROP TABLE IF EXISTS cusip_daily_retx;
    CREATE TABLE cusip_daily_retx AS
    SELECT
        date::DATE AS date,
        cusip,
        retx
    FROM crsp_clean
    WHERE date IS NOT NULL
    AND cusip IS NOT NULL
    AND retx IS NOT NULL
    QUALIFY row_number() OVER (
        PARTITION BY cusip, date
        ORDER BY permno, permco
    ) = 1
    ORDER BY date, cusip;
    """)

    df = conn.execute("""SELECT count(*) AS n_rows, count(DISTINCT cusip) AS n_cusips, count(DISTINCT date) AS n_dates
    FROM cusip_daily_retx;""").fetch_df()
    print(df)

    df = conn.execute("SELECT * FROM cusip_daily_retx ORDER BY date, cusip LIMIT 5").fetch_df()
    return df



if __name__ == "__main__":
    conn = duckdb.connect(new_db_path)
    df = cusip_daily_returns(conn)
    print(df)

    conn.close()