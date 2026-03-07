import os
import duckdb
import numpy as np
import pandas as pd


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

new_db_path = os.path.join(project_root, "new_factor_data.duckdb")

def B_matrix_construction(conn, start_date, end_date):
    # 1) Build sufficient statistics per cusip: X'X and X'y for OLS with intercept + 4 factors
    #    X = [1, mom, tov, size, spread], y = retx
    stats = conn.execute(f"""
        WITH j AS (
            SELECT
                r.cusip,
                r.date,
                r.retx AS y,
                f.momentum_basket AS mom,
                f.turnover_basket AS tov,
                f.size_basket AS siz,
                f.spread_basket AS spr
            FROM cusip_daily_retx r
            JOIN factor_ls_returns f
              ON r.date = f.date
            WHERE r.date >= DATE '{start_date}'
              AND r.date <  DATE '{end_date}'
              AND r.retx IS NOT NULL
              AND f.momentum_basket IS NOT NULL
              AND f.turnover_basket IS NOT NULL
              AND f.size_basket IS NOT NULL
              AND f.spread_basket IS NOT NULL
        )
        SELECT
            cusip,
            count(*) AS n,

            -- X'X terms
            sum(mom) AS s01,
            sum(tov) AS s02,
            sum(siz) AS s03,
            sum(spr) AS s04,

            sum(mom*mom) AS s11,
            sum(mom*tov) AS s12,
            sum(mom*siz) AS s13,
            sum(mom*spr) AS s14,

            sum(tov*tov) AS s22,
            sum(tov*siz) AS s23,
            sum(tov*spr) AS s24,

            sum(siz*siz) AS s33,
            sum(siz*spr) AS s34,

            sum(spr*spr) AS s44,

            -- X'y terms
            sum(y) AS t0,
            sum(mom*y) AS t1,
            sum(tov*y) AS t2,
            sum(siz*y) AS t3,
            sum(spr*y) AS t4,

            -- for R^2 (optional)
            sum(y*y) AS yy
        FROM j
        GROUP BY cusip
        HAVING count(*) >= 60
        ORDER BY cusip
    """).fetch_df()

    # 2) Solve betas per cusip: beta = (X'X)^(-1) X'y
    out_rows = []
    for row in stats.itertuples(index=False):
        n = float(row.n)
        # X'X (5x5) with intercept
        XTX = np.array([
            [n,      row.s01, row.s02, row.s03, row.s04],
            [row.s01, row.s11, row.s12, row.s13, row.s14],
            [row.s02, row.s12, row.s22, row.s23, row.s24],
            [row.s03, row.s13, row.s23, row.s33, row.s34],
            [row.s04, row.s14, row.s24, row.s34, row.s44],
        ], dtype=float)

        XTy = np.array([row.t0, row.t1, row.t2, row.t3, row.t4], dtype=float)

        # Solve; fall back to pseudo-inverse if singular
        try:
            coef = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(XTX) @ XTy

        alpha, b_mom, b_tov, b_siz, b_spr = coef.tolist()

        # Optional R^2
        # SSE = y'y - beta' X'y
        sse = float(row.yy) - float(coef @ XTy)
        # TSS = y'y - (sum y)^2 / n
        tss = float(row.yy) - float((row.t0 * row.t0) / n)
        r2 = None if tss <= 0 else max(0.0, 1.0 - sse / tss)

        out_rows.append({
            "cusip": row.cusip,
            "start_date": start_date,
            "end_date": end_date,
            "nobs": int(row.n),
            "alpha": alpha,
            "beta_momentum": b_mom,
            "beta_turnover": b_tov,
            "beta_size": b_siz,
            "beta_spread": b_spr,
            "r2": r2
        })

    betas_df = pd.DataFrame(out_rows)

    # 3) Persist B to DuckDB (this is your B matrix in table form)
    conn.execute("DROP TABLE IF EXISTS factor_loadings;")
    conn.register("betas_df", betas_df)
    conn.execute("CREATE TABLE factor_loadings AS SELECT * FROM betas_df;")

    # Quick peek
    print(conn.execute("SELECT * FROM factor_loadings ORDER BY cusip LIMIT 5").fetch_df())

def factor_covariance_matrix(conn, start_date, end_date, annualization_factor=252, table_name="factor_covariance_annual"):
    # 1) Pull factor return series for the window
    df = conn.execute(f"""
        SELECT
            momentum_basket,
            turnover_basket,
            size_basket,
            spread_basket
        FROM factor_ls_returns
        WHERE date >= DATE '{start_date}'
          AND date <  DATE '{end_date}'
          AND momentum_basket IS NOT NULL
          AND turnover_basket IS NOT NULL
          AND size_basket IS NOT NULL
          AND spread_basket IS NOT NULL
        ORDER BY date;
    """).fetch_df()

    # 2) Sample covariance (n-1) and annualize
    cov_daily = df.cov()
    cov_annual = cov_daily * float(annualization_factor)

    # 3) Store to DuckDB in long form (easy to query)
    cov_long = (
        cov_annual.stack()
        .reset_index()
        .rename(columns={"level_0": "factor_i", "level_1": "factor_j", 0: "cov_annual"})
    )
    cov_long["start_date"] = start_date
    cov_long["end_date"] = end_date
    cov_long["annualization_factor"] = float(annualization_factor)

    conn.execute(f"DROP TABLE IF EXISTS {table_name};")
    conn.register("cov_long", cov_long)
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM cov_long;")

    return cov_annual

def specific_risk_diagonal_matrix(conn, start_date, end_date, annualization_factor=252, min_obs=60, loadings_table='factor_loadings', returns_table='cusip_daily_retx', factors_table='factor_ls_returns', out_var_table='specific_variance', out_diag_table='sigma_e_diagonal_matrix'):
    # 1) Annualized specific variance per cusip
    conn.execute(f"""
        DROP TABLE IF EXISTS {out_var_table};

        CREATE TABLE {out_var_table} AS
        WITH j AS (
            SELECT
                r.cusip,
                r.date,
                r.retx AS y,
                f.momentum_basket AS mom,
                f.turnover_basket AS tov,
                f.size_basket AS siz,
                f.spread_basket AS spr,
                l.alpha,
                l.beta_momentum,
                l.beta_turnover,
                l.beta_size,
                l.beta_spread
            FROM {returns_table} r
            JOIN {factors_table} f
              ON r.date = f.date
            JOIN {loadings_table} l
              ON l.cusip = r.cusip
             AND l.start_date = '{start_date}'
             AND l.end_date   = '{end_date}'
            WHERE r.date >= DATE '{start_date}'
              AND r.date <  DATE '{end_date}'
              AND r.retx IS NOT NULL
              AND f.momentum_basket IS NOT NULL
              AND f.turnover_basket IS NOT NULL
              AND f.size_basket IS NOT NULL
              AND f.spread_basket IS NOT NULL
        ),
        eps AS (
            SELECT
                cusip,
                y - (
                    alpha
                    + beta_momentum * mom
                    + beta_turnover * tov
                    + beta_size     * siz
                    + beta_spread   * spr
                ) AS e
            FROM j
        )
        SELECT
            cusip,
            count(*) AS nobs,
            var_samp(e) * {float(annualization_factor)} AS specific_var_annual
        FROM eps
        GROUP BY cusip
        HAVING count(*) >= {int(min_obs)}
        ORDER BY cusip;
    """)

    # 2) Diagonal representation of Σ_e (store only diagonal entries)
    conn.execute(f"""
        DROP TABLE IF EXISTS {out_diag_table};

        CREATE TABLE {out_diag_table} AS
        SELECT
            cusip AS asset,
            specific_var_annual
        FROM {out_var_table}
        ORDER BY asset;
    """)

    # Return a quick preview + the diag vector
    preview = conn.execute(f"SELECT * FROM {out_var_table} ORDER BY cusip LIMIT 5").fetch_df()
    diag = conn.execute(f"SELECT * FROM {out_diag_table} ORDER BY asset LIMIT 5").fetch_df()
    return preview, diag

def implied_asset_covariance_factor_model(
    conn,
    start_date,
    end_date,
    loadings_table="factor_loadings",
    factor_cov_table="factor_covariance_matrix",
    specific_var_table="specific_variance",
    factors=("momentum", "turnover", "size", "spread"),
    cusips=None,                 # optional list[str] to limit universe
    store_table=None,            # e.g. "asset_cov_annual_long" to store long form
    max_assets_for_full_matrix=5000  # safety: NxN explodes
):
    # 1) Load B (betas) and specific variances; align on cusip
    cusip_filter = ""
    if cusips is not None and len(cusips) > 0:
        quoted = ",".join(f"'{c}'" for c in cusips)
        cusip_filter = f"AND l.cusip IN ({quoted})"

    betas = conn.execute(f"""
        SELECT
            l.cusip,
            l.beta_momentum,
            l.beta_turnover,
            l.beta_size,
            l.beta_spread
        FROM {loadings_table} l
        WHERE l.start_date = '{start_date}'
          AND l.end_date   = '{end_date}'
          {cusip_filter}
        ORDER BY l.cusip;
    """).fetch_df()

    spec = conn.execute(f"""
        SELECT
            cusip,
            specific_var_annual
        FROM {specific_var_table}
        ORDER BY cusip;
    """).fetch_df()

    df = betas.merge(spec, on="cusip", how="inner").dropna()
    if df.empty:
        raise ValueError("No overlapping assets found between factor_loadings and specific_variance_annual for this window.")

    n_assets = len(df)
    if store_table is None and n_assets > max_assets_for_full_matrix:
        raise ValueError(
            f"Universe has {n_assets} assets; building a full NxN matrix is too large. "
            f"Pass cusips=... to limit, or set store_table=... and store long form for a smaller subset."
        )

    # B: (N x K)
    B = df[["beta_momentum", "beta_turnover", "beta_size", "beta_spread"]].to_numpy(dtype=float)
    cusip_list = df["cusip"].tolist()

    # 2) Load Σ_f (annualized) from long table and pivot to 4x4
    cov_long = conn.execute(f"""
        SELECT factor_i, factor_j, cov_annual
        FROM {factor_cov_table}
        WHERE start_date = '{start_date}'
          AND end_date   = '{end_date}';
    """).fetch_df()

    if cov_long.empty:
        raise ValueError(f"No rows found in {factor_cov_table} for start_date/end_date.")

    Sigma_f = (
        cov_long.pivot(index="factor_i", columns="factor_j", values="cov_annual")
        .reindex(index=[f"{f}_basket" if f.endswith("_basket") else f"{f}_basket" for f in []])  # noop
    )

    # Your factor_cov table uses names like "momentum_basket" as factor_i/j in earlier code.
    # Build expected order explicitly:
    factor_cols = ["momentum_basket", "turnover_basket", "size_basket", "spread_basket"]
    Sigma_f = cov_long.pivot(index="factor_i", columns="factor_j", values="cov_annual").reindex(
        index=factor_cols, columns=factor_cols
    ).to_numpy(dtype=float)

    # 3) Systematic + specific
    systematic = B @ Sigma_f @ B.T                          # (N x N)
    specific = np.diag(df["specific_var_annual"].to_numpy(dtype=float))  # (N x N)
    Sigma = systematic + specific

    # 4) Optionally store as long form (i, j, cov)
    if store_table is not None:
        # long form with upper triangle (including diagonal) to cut storage ~half
        rows = []
        for i in range(n_assets):
            for j in range(i, n_assets):
                rows.append((cusip_list[i], cusip_list[j], float(Sigma[i, j])))

        out = pd.DataFrame(rows, columns=["cusip_i", "cusip_j", "cov_annual"])
        out["start_date"] = start_date
        out["end_date"] = end_date

        conn.execute(f"DROP TABLE IF EXISTS {store_table};")
        conn.register("out_cov", out)
        conn.execute(f"CREATE TABLE {store_table} AS SELECT * FROM out_cov;")

    return Sigma, cusip_list
    

if __name__ == "__main__":
    start_date = "2022-01-01"
    end_date = "2023-01-01"
    conn = duckdb.connect(new_db_path)
    B_matrix_construction(conn, start_date, end_date)
    df = factor_covariance_matrix(conn, start_date, end_date, table_name="factor_covariance_matrix")
    print(df)
    preview, diag = specific_risk_diagonal_matrix(conn, start_date, end_date, out_var_table="specific_variance", out_diag_table="sigma_e_diagonal_matrix")
    print("Preview of specific variance table:")
    print(preview)
    print("Preview of diagonal matrix table:")
    print(diag)

    Sigma, assets = implied_asset_covariance_factor_model(
    conn,
    start_date,
    end_date,
    cusips=["00036020","00036110","00037520"],   # example subset
    )
    print(Sigma.shape)
    print(assets)
    conn.close()