

def compute_turnover_expr(window: int = 20) -> str:
    """
    Rolling average turnover:
        volume / shares_outstanding

    CRSP shrout is often in thousands, so use shrout * 1000.
    """
    return f"""
    AVG(
        CASE
            WHEN shrout IS NOT NULL AND shrout > 0 AND vol IS NOT NULL
            THEN vol / (shrout * 1000.0)
            ELSE NULL
        END
    ) OVER (
        PARTITION BY permno
        ORDER BY date
        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
    )
    """