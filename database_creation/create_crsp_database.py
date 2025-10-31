import duckdb

# 1️⃣ Connect to (or create) a DuckDB database file
conn = duckdb.connect('crsp_dataset.duckdb')

# 2️⃣ Create a table from your Parquet file
# Replace the path with your actual Parquet file path
parquet_file = "crsp/CRSPdF20022025.parquet"

conn.execute(f"""
    CREATE TABLE IF NOT EXISTS crsp_table AS
    SELECT * FROM '{parquet_file}'
""")

# 3️⃣ Quick preview: First 10 rows
df_preview = conn.execute("SELECT * FROM crsp_table LIMIT 10").df()
print("Preview of crsp_table:")
print(df_preview)