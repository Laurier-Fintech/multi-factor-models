

def compute_momentum_expr(window: int = 20, skip_recent: int = 0) -> str:
    """
    Momentum signal:
    cumulative trailing return over a past window.

    If skip_recent > 0, excludes the most recent N rows to reduce short-term reversal contamination.

    Approximation:
        sum(log(1 + ret)) over trailing window, then exponentiate.

    Returns a SQL expression string.
    """
    start_offset = window + skip_recent
    end_offset = skip_recent + 1

    return f"""
    EXP(
        SUM(
            CASE
                WHEN ret IS NOT NULL AND ret > -1 THEN LN(1 + ret)
                ELSE NULL
            END
        ) OVER (
            PARTITION BY permno
            ORDER BY date
            ROWS BETWEEN {start_offset - 1} PRECEDING AND {end_offset - 1} PRECEDING
        )
    ) - 1
    """