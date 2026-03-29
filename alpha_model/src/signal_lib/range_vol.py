def compute_range_vol_expr(window: int = 20) -> str:
      """
      Intraday range volatility (Parkinson 1980 estimator proxy):
          avg( (askhi - bidlo) / ABS(prc) ) over trailing window

      Uses the daily high-low spread as a more efficient volatility
      estimator than close-to-close returns. Also captures intraday
      liquidity stress not visible in end-of-day prices.

      Requires askhi and bidlo columns from CRSP.
      """
      return f"""
      AVG(
          CASE
              WHEN askhi IS NOT NULL
               AND bidlo IS NOT NULL
               AND prc   IS NOT NULL
               AND ABS(prc) > 0
               AND askhi >= bidlo
              THEN (askhi - bidlo) / ABS(prc)
              ELSE NULL
          END
      ) OVER (
          PARTITION BY permno
          ORDER BY date
          ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
      )
      """