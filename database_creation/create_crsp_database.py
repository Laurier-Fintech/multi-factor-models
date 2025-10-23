import duckdb

# 1️⃣ Connect to (or create) a DuckDB database file
conn = duckdb.connect('crsp_dataset.duckdb')

# 2️⃣ Create a table from your Parquet file
parquet_file = r"C:\Users\agand\Desktop\Personal_Coding\Finance_Projects\MultiFactorModel_LFQA\multi-factor-models\venv\crsp\CRSPdF20022025.parquet"

conn.execute(f"""
    CREATE TABLE IF NOT EXISTS crsp_table AS
    SELECT * FROM '{parquet_file}'
""")

# 3️⃣ Quick preview: First 10 rows
df_preview = conn.execute("SELECT * FROM crsp_table LIMIT 10").df()
print("Preview of crsp_table:")
print(df_preview)
