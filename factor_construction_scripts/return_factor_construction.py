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
    df = conn.execute("""SELECT * FROM crsp_table WHERE "cusip"='36720410' OR "cusip"='05978R10' LIMIT 9000""").df()
    return df

def _prep_table(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['permno', 'date'])
    return df

def garch_1_1(df):
    """
    The Generalized Autoregressive Conditional Heteroscedasticity (GARCH) Model

    This function implements a GARCH(1,1) model, which is used to model historical 
    and forecast future volatility levels of a marketable security.

    The input dataset must be a time series of raw price levels. The GARCH(1,1) model 
    first converts prices into relative returns, then performs an internal optimization 
    to fit the historical data to a mean-reverting volatility term structure. It assumes 
    that volatility is heteroscedastic — that is, it changes over time according to 
    certain econometric characteristics.

    Although it is a simplified case of the more general GARCH(p, q) model, GARCH(1,1) 
    is often sufficient for modeling financial time series volatility.
    """
    df = df.copy()
    df['adj_close'] = df['prc'] * df['cfacpr']
    df['relatives'] = df['adj_close'] / df['adj_close'].shift(1)
    df['daily_log_return'] = np.log(df['relatives'])
    df['daily_sqrd_return'] = df['daily_log_return'] ** 2

    returns = df['daily_log_return'].dropna().values

    def garch_likelihood(params, returns):
        omega, alpha, beta = params
        var = np.zeros_like(returns)
        var[0] = np.var(returns)
        for t in range(1, len(returns)):
            var[t] = omega + alpha * returns[t-1]**2 + beta * var[t-1]
        LL = -0.5 * np.sum(np.log(var) + (returns**2) / var)
        return -LL

    initial_guess = [1e-6, 0.1, 0.8]
    bounds = ((0, None), (0, 1), (0, 1))
    result = opt.minimize(garch_likelihood, initial_guess, args=(returns,), bounds=bounds, method='L-BFGS-B')
    omega, alpha, beta = result.x

    # Rebuild variance series
    var = np.zeros_like(returns)
    var[0] = np.var(returns)
    for t in range(1, len(returns)):
        var[t] = omega + alpha * returns[t-1]**2 + beta * var[t-1]

    # Align back with original dataframe
    df['garch_variance'] = np.nan
    df.loc[df['daily_log_return'].notna(), 'garch_variance'] = var
    df['garch_volatility'] = np.sqrt(df['garch_variance'])

    return df

def momentum_factor(df):
    """
    Calculate the momentum factor based on past 12-month cumulative returns,
    skipping the most recent month.

    Steps:
    1. Compute daily log returns if not provided.
    2. Aggregate to monthly log returns.
    3. Compute rolling 11-month cumulative return (t-12 to t-2).
    4. Merge back to daily data and cross-sectionally z-score per date.
    """
    df = df.copy()

    # 1) Calculate returns if not already present
    if 'daily_log_return' not in df.columns:
        df['adj_close'] = df['prc'] * df['cfacpr']
        df['relatives'] = df['adj_close'] / df['adj_close'].shift(1)
        df['daily_log_return'] = np.log(df['relatives'])

    # 2) Convert daily returns to monthly returns
    df['yearmonth'] = df['date'].dt.to_period('M')
    monthly_returns = (
        df.groupby(['permno', 'yearmonth'], as_index=False)['daily_log_return']
        .sum()
    )

    # 3) Calculate rolling 11-month cumulative returns (t-12 to t-2)
    monthly_returns['momentum_factor'] = (
        monthly_returns.groupby('permno')['daily_log_return']
        .transform(lambda x: x.rolling(window=11, min_periods=11).sum().shift(1))
    )

    # 4) Merge back to daily data
    df = df.merge(
        monthly_returns[['permno', 'yearmonth', 'momentum_factor']],
        on=['permno', 'yearmonth'],
        how='left'
    )


    return df

def market_cap_factor(df):
    """
    Calculate the market capitalization factor.

    Steps:
    1. Compute market capitalization as price * shares outstanding.
    2. Cross-sectionally z-score market capitalization per date.
    """
    df = df.copy()
    df['market_cap'] = df['prc'] * df['shrout'] * 1000  # Assuming 'shrout' is in thousands

    return df

def cross_sectional_standardize(df, factor_cols):
    for date, group in df.groupby('date'):
        for col in factor_cols:
            mean = group[col].mean()
            std = group[col].std()
            df.loc[group.index, f'{col}_z_cs'] = (group[col] - mean) / std
    return df


def compute_factor_returns(df, factor_cols, ret_col='retx'):
    """
    Cross-sectional regression to estimate daily factor returns.
    """
    factor_returns = []

    for date, group in df.groupby('date'):
        X = group[factor_cols]
        y = group[ret_col]
        
        # Drop rows where all factor values are NaN
        mask = X.notna().any(axis=1)
        X = X.loc[mask]
        y = y.loc[mask]

        # Skip if no valid rows
        if X.empty:
            continue

        X = sm.add_constant(X)  # add intercept
        model = sm.OLS(y, X, missing='drop')
        result = model.fit()

        row = {'date': date}
        row.update(result.params.to_dict())
        factor_returns.append(row)

    return pd.DataFrame(factor_returns)




if __name__ == "__main__":
    df = _pull_table()
    df = _prep_table(df)
    df = garch_1_1(df)
    df = momentum_factor(df)
    df = market_cap_factor(df)

    factor_cols = ['momentum_factor', 'garch_volatility', 'market_cap']
    df = cross_sectional_standardize(df, factor_cols)
    df = df.dropna(subset=['garch_volatility_z_cs', 'momentum_factor_z_cs', 'market_cap_z_cs', 'retx'])

    factor_cols = ['momentum_factor_z_cs', 'garch_volatility_z_cs', 'market_cap_z_cs']
    factor_premiums = compute_factor_returns(df, factor_cols)

    
    df = df.merge(factor_premiums, on='date', how='left', suffixes=('', '_premium'))
    df['expected_return'] = (
    df['const'] +  # regression intercept
    df['momentum_factor_z_cs'] * df['momentum_factor_z_cs_premium'] +
    df['garch_volatility_z_cs'] * df['garch_volatility_z_cs_premium'] +
    df['market_cap_z_cs'] * df['market_cap_z_cs_premium'])

    print(df)
