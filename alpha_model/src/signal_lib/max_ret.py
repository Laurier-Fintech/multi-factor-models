def compute_max_ret_expr(window: int = 20) -> str:
      """
      Maximum daily return over the trailing window (Bali et al. 2011).

      Stocks with a high recent maximum 1-day return are overpriced due to
      lottery demand and subsequently underperform. Negative predictor.
      Uses ret (total return including dividends).
      """
      return f"""
      MAX(
          CASE
              WHEN ret IS NOT NULL THEN ret
              ELSE NULL
          END
      ) OVER (
          PARTITION BY permno
          ORDER BY date
          ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
      )
      """