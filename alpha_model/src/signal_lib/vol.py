

def compute_volatility_expr(window: int = 20) -> str:
    """
    Rolling trailing volatility of returns.
    """
    return f"""
    STDDEV_SAMP(ret) OVER (
        PARTITION BY permno
        ORDER BY date
        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
    )
    """