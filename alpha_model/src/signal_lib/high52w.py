def compute_high52w_expr(window: int = 252) -> str:
      """
      52-week high ratio (George & Hwang 2004):
          current price / rolling max price over past `window` trading days

      Stocks trading near their 52W high tend to continue outperforming.
      Uses ABS(prc) to handle CRSP negative-price convention.
      """
      return f"""
      ABS(prc) / NULLIF(
          MAX(ABS(prc)) OVER (
              PARTITION BY permno
              ORDER BY date
              ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
          ),
      0)
      """