

def compute_reversal_expr(window: int = 1) -> str:
    """
    Short-term reversal:
    negative trailing cumulative return over the last `window` periods.

    For window=1:
        reversal = -ret_{t-1}
    """
    return f"""
    -1 * (
        EXP(
            SUM(
                CASE
                    WHEN ret IS NOT NULL AND ret > -1 THEN LN(1 + ret)
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY permno
                ORDER BY date
                ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
            )
        ) - 1
    )
    """