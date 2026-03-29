def compute_rel_volume_expr(short_window: int = 5, long_window: int = 60) -> str:
      """
      Relative volume: ratio of short-term to medium-term average daily volume.

      Captures abnormal trading activity. A ratio > 1 means recent volume is
      elevated relative to its baseline — often precedes price continuation
      or reversal depending on direction of the accompanying return.
      """
      return f"""
      AVG(
          CASE WHEN vol IS NOT NULL AND vol > 0 THEN vol ELSE NULL END
      ) OVER (
          PARTITION BY permno
          ORDER BY date
          ROWS BETWEEN {short_window} PRECEDING AND 1 PRECEDING
      )
      / NULLIF(
          AVG(
              CASE WHEN vol IS NOT NULL AND vol > 0 THEN vol ELSE NULL END
          ) OVER (
              PARTITION BY permno
              ORDER BY date
              ROWS BETWEEN {long_window} PRECEDING AND 1 PRECEDING
          ),
      0)
      """