import duckdb
import pandas as pd
from datetime import datetime
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
import warnings
# warnings.simplefilter('ignore')
conn = duckdb.connect('crsp_dataset.duckdb')

def _pull_table():
    df = conn.execute("""
        SELECT *
        FROM crsp_table
        WHERE permno IN (
            SELECT DISTINCT permno
            FROM crsp_table
            ORDER BY permno
            LIMIT 100
        )
        ORDER BY permno, date
    """).df()
    return df

def _prep_table(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['permno', 'date'])
    return df

def compute_factors(df, momentum_window=21):
    """
    Compute basic daily factor exposures:
      - Momentum (past cumulative return)
      - Market capitalization
      - Bid-ask spread
    """
    df = df.copy()
    
    # --- Market Cap ---
    df['mktcap'] = df['prc'].abs() * df['shrout']
    df['ln_mktcap'] = np.log(df['mktcap'].replace(0, np.nan))

    # --- Bid-Ask Spread ---
    df['spread'] = (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2)

    # --- Momentum ---
    df['momentum'] = (
        df.groupby('permno', group_keys=False)['ret']
        .transform(lambda x: (1 + x.shift(1)).rolling(momentum_window).apply(np.prod, raw=True) - 1)
    )
    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['momentum', 'ln_mktcap', 'spread'])

    # Optional: return only factor columns
    factor_cols = ['permno', 'date', 'momentum', 'ln_mktcap', 'spread']
    df_factors = df[factor_cols].reset_index(drop=True)
    
    return df_factors

def build_factor_exposure_matrix(df_factors, standardize=True):
    """
    Build daily cross-sectional factor exposure matrix for regressions.
    
    Parameters
    ----------
    df_factors : pd.DataFrame
        Must include ['permno', 'date', factor columns...].
    standardize : bool
        Whether to z-score exposures cross-sectionally per date.
        
    Returns
    -------
    pd.DataFrame
        Long-form factor exposure matrix:
        date | permno | momentum | ln_mktcap | spread
    """

    df = df_factors.copy()
    factor_cols = [col for col in df.columns if col not in ['date', 'permno']]

    if standardize:
        # Cross-sectional z-score per day
        df[factor_cols] = df.groupby('date')[factor_cols].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )

    # Drop NaNs (early days with insufficient momentum data)
    df = df.dropna(subset=factor_cols)

    # Sort for clean time ordering
    df = df.sort_values(['date', 'permno']).reset_index(drop=True)
    
    return df

def estimate_daily_factor_returns(df_exposures, df_returns):
    """
    Run daily cross-sectional regressions (Fama-MacBeth step 1)
    to estimate factor returns (gamma_t's).
    
    Parameters
    ----------
    df_exposures : pd.DataFrame
        Must include ['date', 'permno', factor columns...].
    df_returns : pd.DataFrame
        Must include ['date', 'permno', 'ret'].
    
    Returns
    -------
    pd.DataFrame
        date | factor | gamma_t
    """
    df = pd.merge(
        df_exposures,
        df_returns[['date', 'permno', 'ret']],
        on=['date', 'permno'],
        how='inner'
    )

    factor_cols = [c for c in df.columns if c not in ['date', 'permno', 'ret']]

    results = []
    unique_dates = df['date'].drop_duplicates().sort_values()

    for date, group in df.groupby('date'):
        # Drop missing values
        sub = group.dropna(subset=factor_cols + ['ret'])
        if len(sub) < len(factor_cols) + 2:
            continue  # skip days with too few observations

        # Add constant for intercept
        X = sm.add_constant(sub[factor_cols])
        y = sub['ret']

        model = sm.OLS(y, X).fit()
        gamma_t = model.params.to_dict()

        # Store results
        for factor, val in gamma_t.items():
            results.append({'date': date, 'factor': factor, 'gamma_t': val})

    df_factor_returns = pd.DataFrame(results)

    # Clean up column names
    df_factor_returns = df_factor_returns.rename(columns={'gamma_t': 'factor_return'})
    df_factor_returns = df_factor_returns.sort_values(['date', 'factor']).reset_index(drop=True)

    return df_factor_returns

def compute_expected_returns(df_exposures, df_factor_returns, window=None):
    """
    Compute expected stock returns from factor exposures and factor returns.

    Parameters
    ----------
    df_exposures : pd.DataFrame
        ['date', 'permno', factor exposures...]
    df_factor_returns : pd.DataFrame
        ['date', 'factor', 'factor_return']
    window : int, optional
        Rolling window for averaging factor returns (e.g., 21 for 1 month)

    Returns
    -------
    pd.DataFrame
        ['date', 'permno', 'expected_return']
    """
    # Pivot factor returns: date as index, factors as columns
    df_wide = df_factor_returns.pivot(index='date', columns='factor', values='factor_return')
    df_wide = df_wide.sort_index()

    # Apply rolling window average if requested
    if window is not None:
        df_wide = df_wide.rolling(window).mean()

    # Merge factor returns into exposures
    df = pd.merge(df_exposures, df_wide, on='date', how='left', suffixes=('', '_gamma'))

    # Identify factor columns dynamically (present in both exposures and factor returns)
    factor_cols = [col for col in df_exposures.columns if col not in ['date', 'permno']]
    factor_gamma_cols = [c for c in df.columns if c in factor_cols]

    # Compute expected return as row-wise dot product
    df['expected_return'] = df[factor_cols].mul(df[factor_gamma_cols].values).sum(axis=1)
    df_expected = df[['date', 'permno', 'expected_return']].dropna().reset_index(drop=True)
    return df_expected




if __name__ == "__main__":
    df = _pull_table()
    df = _prep_table(df)
    # Step 1: Factor signals
    df_factors = compute_factors(df)

    # Step 2: Exposure matrix
    df_exposures = build_factor_exposure_matrix(df_factors)

    # Step 3: Daily regressions for factor returns
    df_factor_returns = estimate_daily_factor_returns(df_exposures, df)

    # Step 4: Save to Parquet
    df_factor_returns.to_parquet(r"factor returns\factor_returns.parquet", index=False)

    # Step 4: compute expected returns per stock
    df_expected = compute_expected_returns(df_exposures, df_factor_returns, window=21)
    df_expected.to_parquet(r"expected returns\expected_returns.parquet", index=False)

    print(df_expected.head())
