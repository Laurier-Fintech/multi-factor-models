import duckdb
import pandas as pd

db_path = r"A:\Users\Cassel Robson\fintech\multi-factor-cassel-space\multi-factor-models\database_creation\crsp_dataset.duckdb"
parquet_file = r"A:\Users\Cassel Robson\fintech\multi-factor-cassel-space\multi-factor-models\database_creation\CRSPdF20022025.parquet"

def main():
    conn = duckdb.connect(db_path)

    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS crsp_table AS
    SELECT * FROM read_parquet('{parquet_file}')
    """)

    df_preview = conn.execute("SELECT * FROM crsp_table LIMIT 5").fetchdf()
    print(df_preview)

    # Get all unique CUSIPs
    cusip_df = conn.execute("""
    SELECT DISTINCT cusip 
    FROM crsp_table
    ORDER BY cusip
    """).fetchdf()

    print(f"Number of unique CUSIPs: {len(cusip_df)}")
    print(cusip_df)
    # cusip_df.to_csv("cusips.csv", index=False)
    conn.close()


if __name__ == "__main__":
    main()