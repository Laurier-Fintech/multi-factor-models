"""
  create_forward_return.py

  Appends an H-day forward return column to the alpha signals table.
  """
import duckdb


def create_forward_return(
      conn: duckdb.DuckDBPyConnection,
      source_table: str = "alpha_signals",
      output_table: str = "alpha_signals_with_fwd",
      horizon: int = 5,
  ) -> None:
      """
      Append an H-day forward return column to the alpha signals table.

      Uses ABS(prc) to handle CRSP sign convention (negative price = bid-ask midpoint).
      Writes to output_table in DuckDB.
      """
      conn.execute(f"""
      CREATE OR REPLACE TABLE {output_table} AS
      SELECT
          *,
          ABS(LEAD(prc, {horizon}) OVER (PARTITION BY permno ORDER BY date))
              / NULLIF(ABS(prc), 0) - 1 AS fwd_ret_{horizon}d
      FROM {source_table}
      """)