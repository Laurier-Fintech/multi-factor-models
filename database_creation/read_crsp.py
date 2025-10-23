import pandas as pd
from datetime import datetime
import os, psutil
import numpy as np

file_size = os.path.getsize(r"crsp\CRSPdF20022025.feather") / 1e9
mem_available = psutil.virtual_memory().available / 1e9

print(f"File size: {file_size:.2f} GB")
print(f"Available RAM: {mem_available:.2f} GB")
# df = pd.read_csv(r"crsp\CRSPdF20022025 (1).csv")
# df.to_feather(r"crsp\CRSPdF20022025.feather")
df = pd.read_feather(r"crsp\CRSPdF20022025.feather")

df['date'] = pd.to_datetime(df['date'])

df = df[df['date'] > datetime(2023, 1, 1)]
df = df.sort_values(['permno', 'date'])
df = df.astype({'retx': 'float32'})
print(df.tail())

def compute_return_factors(df):
    # Daily returns per stock
    df['retx'] = df['retx'].astype(float)
    
    # Group for rolling stats
    g = df.groupby('permno', group_keys=False)

    # --- Short-term reversal (previous day’s negative return)
    df['rev_1d'] = g['retx'].shift(1) * -1

    # --- 1M momentum (past 21 trading days cumulative return)
    df['mom_1m'] = (1 + g['retx'].apply(lambda x: x.rolling(21).apply(lambda r: np.prod(1 + r) - 1))).reset_index(level=0, drop=True)

    # --- 3M momentum (past 63 days)
    df['mom_3m'] = (1 + g['retx'].apply(lambda x: x.rolling(63).apply(lambda r: np.prod(1 + r) - 1))) - 1

    # --- 12M momentum (past 252 days, skip most recent 21)
    df['mom_12m'] = (
        (1 + g['retx'].apply(lambda x: (x.shift(21)).rolling(252).apply(lambda r: np.prod(1 + r) - 1))) - 1
    )

    # --- Volatility (rolling std of daily returns)
    df['vol_21d'] = g['retx'].transform(lambda x: x.rolling(21).std())
    df['vol_63d'] = g['retx'].transform(lambda x: x.rolling(63).std())
    df['vol_252d'] = g['retx'].transform(lambda x: x.rolling(252).std())

    # --- Downside Volatility (std of negative returns only)
    def downside_vol(x, window):
        return x.rolling(window).apply(lambda r: np.std(r[r < 0]) if np.any(r < 0) else np.nan)

    df['dvol_63d'] = g['retx'].apply(lambda x: downside_vol(x, 63)).reset_index(level=0, drop=True)

    # --- Autocorrelation (AR1)
    def rolling_autocorr(x, window):
        return x.rolling(window).apply(lambda r: np.corrcoef(r[:-1], r[1:])[0,1] if len(r.dropna()) > 2 else np.nan)

    df['ar1_63d'] = g['retx'].apply(lambda x: rolling_autocorr(x, 63)).reset_index(level=0, drop=True)

    # --- Return skewness and kurtosis (optional, heavy)
    df['skew_252d'] = g['retx'].transform(lambda x: x.rolling(252).skew())
    df['kurt_252d'] = g['retx'].transform(lambda x: x.rolling(252).kurt())

    return df


df = compute_return_factors(df)
print(df.head())