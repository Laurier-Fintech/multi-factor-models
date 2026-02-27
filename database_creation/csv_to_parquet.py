import pandas as pd

# Read the CSV in chunks to avoid memory issues
csv_file = r"A:\Users\Cassel Robson\fintech\multi-factor-cassel-space\multi-factor-models\CRSPdF20022025 (1).csv"
parquet_file = r"A:\Users\Cassel Robson\fintech\multi-factor-cassel-space\multi-factor-models\CRSPdF20022025.parquet"

# If the file fits in memory:
df = pd.read_csv(csv_file)
df.to_parquet(parquet_file, engine='pyarrow', index=False)
print(f"CSV converted to Parquet: {parquet_file}")