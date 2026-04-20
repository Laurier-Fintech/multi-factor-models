import os
import duckdb
import numpy as np
import pandas as pd


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

new_db_path = os.path.join(project_root, "new_factor_data.duckdb")

def B_matrix_construction(conn, start_date, end_date):
      # 1) Discover factors dynamically — everything in factor_ls_returns except 'date'
      factor_cols = [
          col for col in
          conn.execute("SELECT * FROM factor_ls_returns LIMIT 0").fetch_df().columns
          if col != 'date'
      ]
      K = len(factor_cols)

      # 2) Build SQL fragments programmatically
      aliases      = [f'f.{col} AS f{i}' for i, col in enumerate(factor_cols)]
      null_filters = [f'f.{col} IS NOT NULL' for col in factor_cols]

      # X'X: sum(fi) for intercept cross-terms, sum(fi*fj) for upper triangle
      sums_linear = [f'sum(f{i}) AS s0_{i}' for i in range(K)]
      sums_quad   = [f'sum(f{i}*f{j}) AS s_{i}_{j}' for i in range(K) for j in range(i, K)]

      # X'y: sum(fi*y) for each factor
      xy_terms = [f'sum(f{i}*y) AS t{i}' for i in range(K)]

      select_parts = (
          ['cusip', 'count(*) AS n']
          + sums_linear
          + sums_quad
          + ['sum(y) AS t_intercept']
          + xy_terms
          + ['sum(y*y) AS yy']
      )

      sql = f"""
          WITH j AS (
              SELECT
                  r.cusip,
                  r.retx AS y,
                  {', '.join(aliases)}
              FROM cusip_daily_retx r
              JOIN factor_ls_returns f ON r.date = f.date
              WHERE r.date >= DATE '{start_date}'
                AND r.date <  DATE '{end_date}'
                AND r.retx IS NOT NULL
                AND {' AND '.join(null_filters)}
          )
          SELECT {', '.join(select_parts)}
          FROM j
          GROUP BY cusip
          HAVING count(*) >= 60
          ORDER BY cusip
      """

      stats = conn.execute(sql).fetch_df()

      # 3) Solve betas per cusip
      out_rows = []
      for row in stats.itertuples(index=False):
          n = float(row.n)

          # Build XTX ((K+1) x (K+1)) with intercept in position 0
          XTX = np.zeros((K + 1, K + 1), dtype=float)
          XTX[0, 0] = n
          for i in range(K):
              v = getattr(row, f's0_{i}')
              XTX[0, i + 1] = v
              XTX[i + 1, 0] = v
          for i in range(K):
              for j in range(i, K):
                  v = getattr(row, f's_{i}_{j}')
                  XTX[i + 1, j + 1] = v
                  XTX[j + 1, i + 1] = v

          # Build XTy (K+1,)
          XTy = np.zeros(K + 1, dtype=float)
          XTy[0] = getattr(row, 't_intercept')
          for i in range(K):
              XTy[i + 1] = getattr(row, f't{i}')

          # Solve; fall back to pseudo-inverse if singular
          try:
              coef = np.linalg.solve(XTX, XTy)
          except np.linalg.LinAlgError:
              coef = np.linalg.pinv(XTX) @ XTy

          # R^2
          sse = float(row.yy) - float(coef @ XTy)
          tss = float(row.yy) - float(XTy[0] ** 2 / n)
          r2  = None if tss <= 0 else max(0.0, 1.0 - sse / tss)

          out = {
              'cusip':      row.cusip,
              'start_date': start_date,
              'end_date':   end_date,
              'nobs':       int(row.n),
              'alpha':      float(coef[0]),
          }
          for i, col in enumerate(factor_cols):
              out[f'beta_{col}'] = float(coef[i + 1])
          out['r2'] = r2

          out_rows.append(out)

      betas_df = pd.DataFrame(out_rows)

      # 4) Persist
      conn.execute("DROP TABLE IF EXISTS factor_loadings;")
      conn.register("betas_df", betas_df)
      conn.execute("CREATE TABLE factor_loadings AS SELECT * FROM betas_df;")

      #print(conn.execute("SELECT * FROM factor_loadings ORDER BY cusip LIMIT 5").fetch_df())

def factor_covariance_matrix(conn, start_date, end_date, annualization_factor=252, table_name="factor_covariance_annual"):
      # 1) Discover factors dynamically
      factor_cols = [
          col for col in
          conn.execute("SELECT * FROM factor_ls_returns LIMIT 0").fetch_df().columns
          if col != 'date'
      ]

      select_clause = ', '.join(factor_cols)
      null_filters  = ' AND '.join(f'{col} IS NOT NULL' for col in factor_cols)

      df = conn.execute(f"""
          SELECT {select_clause}
          FROM factor_ls_returns
          WHERE date >= DATE '{start_date}'
            AND date <  DATE '{end_date}'
            AND {null_filters}
          ORDER BY date;
      """).fetch_df()

      # 2) Sample covariance (n-1) and annualize
      cov_daily  = df.cov()
      cov_annual = cov_daily * float(annualization_factor)

      # 3) Store to DuckDB in long form
      cov_long = (
          cov_annual.stack()
          .reset_index()
          .rename(columns={"level_0": "factor_i", "level_1": "factor_j", 0: "cov_annual"})
      )
      cov_long["start_date"]          = start_date
      cov_long["end_date"]            = end_date
      cov_long["annualization_factor"] = float(annualization_factor)

      conn.execute(f"DROP TABLE IF EXISTS {table_name};")
      conn.register("cov_long", cov_long)
      conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM cov_long;")

      return cov_annual

def specific_risk_diagonal_matrix(conn, start_date, end_date, annualization_factor=252, min_obs=60, loadings_table='factor_loadings',
  returns_table='cusip_daily_retx', factors_table='factor_ls_returns', out_var_table='specific_variance', out_diag_table='sigma_e_diagonal_matrix'):
      # 1) Discover factors dynamically
      factor_cols = [
          col for col in
          conn.execute(f"SELECT * FROM {factors_table} LIMIT 0").fetch_df().columns
          if col != 'date'
      ]

      factor_aliases  = ', '.join(f'f.{col} AS f_{col}' for col in factor_cols)
      beta_selects    = ', '.join(f'l.beta_{col}' for col in factor_cols)
      null_filters    = ' AND '.join(f'f.{col} IS NOT NULL' for col in factor_cols)
      residual_terms  = ' + '.join(f'beta_{col} * f_{col}' for col in factor_cols)

      conn.execute(f"""
          DROP TABLE IF EXISTS {out_var_table};

          CREATE TABLE {out_var_table} AS
          WITH j AS (
              SELECT
                  r.cusip,
                  r.date,
                  r.retx AS y,
                  {factor_aliases},
                  l.alpha,
                  {beta_selects}
              FROM {returns_table} r
              JOIN {factors_table} f ON r.date = f.date
              JOIN {loadings_table} l
                ON l.cusip = r.cusip
               AND l.start_date = '{start_date}'
               AND l.end_date   = '{end_date}'
              WHERE r.date >= DATE '{start_date}'
                AND r.date <  DATE '{end_date}'
                AND r.retx IS NOT NULL
                AND {null_filters}
          ),
          eps AS (
              SELECT
                  cusip,
                  y - (alpha + {residual_terms}) AS e
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

      conn.execute(f"""
          DROP TABLE IF EXISTS {out_diag_table};

          CREATE TABLE {out_diag_table} AS
          SELECT cusip AS asset, specific_var_annual
          FROM {out_var_table}
          ORDER BY asset;
      """)

      preview = conn.execute(f"SELECT * FROM {out_var_table} ORDER BY cusip LIMIT 5").fetch_df()
      diag    = conn.execute(f"SELECT * FROM {out_diag_table} ORDER BY asset LIMIT 5").fetch_df()
      return preview, diag

def implied_asset_covariance_factor_model(
      conn,
      start_date,
      end_date,
      loadings_table="factor_loadings",
      factor_cov_table="factor_covariance_matrix",
      specific_var_table="specific_variance",
      cusips=None,
      store_table=None,
      max_assets_for_full_matrix=5000, 
      min_specific_var=0.04,
  ):
      # 1) Discover factors dynamically
      factor_cols = [
          col for col in
          conn.execute("SELECT * FROM factor_ls_returns LIMIT 0").fetch_df().columns
          if col != 'date'
      ]
      beta_cols = [f'beta_{col}' for col in factor_cols]

      # 2) Load B (betas) and specific variances; align on cusip
      cusip_filter = ""
      if cusips is not None and len(cusips) > 0:
          quoted = ",".join(f"'{c}'" for c in cusips)
          cusip_filter = f"AND l.cusip IN ({quoted})"

      beta_select = ', '.join(f'l.{bc}' for bc in beta_cols)

      betas = conn.execute(f"""
          SELECT l.cusip, {beta_select}
          FROM {loadings_table} l
          WHERE l.start_date = '{start_date}'
            AND l.end_date   = '{end_date}'
            {cusip_filter}
          ORDER BY l.cusip;
      """).fetch_df()

      spec = conn.execute(f"""
          SELECT cusip, specific_var_annual
          FROM {specific_var_table}
          ORDER BY cusip;
      """).fetch_df()

      df = betas.merge(spec, on="cusip", how="inner").dropna()
      if df.empty:
          raise ValueError("No overlapping assets found between factor_loadings and specific_variance for this window.")

      n_assets = len(df)
      if store_table is None and n_assets > max_assets_for_full_matrix:
          raise ValueError(
              f"Universe has {n_assets} assets; building a full NxN matrix is too large. "
              f"Pass cusips=... to limit, or set store_table=... to store in long form."
          )

      # B: (N x K)
      B          = df[beta_cols].to_numpy(dtype=float)
      cusip_list = df["cusip"].tolist()

      # 3) Load Sigma_f (annualized) from long table and pivot to (K x K)
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
          .reindex(index=factor_cols, columns=factor_cols)
          .to_numpy(dtype=float)
      )

      # 4) Systematic + specific
      systematic   = B @ Sigma_f @ B.T
      specific_var = np.maximum(df["specific_var_annual"].to_numpy(dtype=float), min_specific_var)
      specific     = np.diag(specific_var)
      Sigma        = systematic + specific

      # 5) Optionally store as long form (upper triangle)
      if store_table is not None:
          rows = []
          for i in range(n_assets):
              for j in range(i, n_assets):
                  rows.append((cusip_list[i], cusip_list[j], float(Sigma[i, j])))

          out = pd.DataFrame(rows, columns=["cusip_i", "cusip_j", "cov_annual"])
          out["start_date"] = start_date
          out["end_date"]   = end_date

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
    print(Sigma)
    print(assets)
    conn.close()