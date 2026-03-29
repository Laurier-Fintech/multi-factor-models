def compute_downside_vol_expr(window: int = 20) -> str:
      """
      Downside volatility: standard deviation computed only over negative return days.

      Captures asymmetric risk better than symmetric volatility. Stocks with
      high downside vol tend to underperform as they carry more crash risk.
      """
      return f"""
      STDDEV_SAMP(
          CASE
              WHEN ret IS NOT NULL AND ret < 0 THEN ret
              ELSE NULL
          END
      ) OVER (
          PARTITION BY permno
          ORDER BY date
          ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
      )
      """