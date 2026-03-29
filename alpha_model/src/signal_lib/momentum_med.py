def compute_momentum_med_expr(window: int = 120, skip_recent: int = 5) -> str:
    """
    Medium-term momentum (Jegadeesh & Titman 1993):
        compounded return over a past window, skipping the most recent
        `skip_recent` days to avoid short-term reversal contamination.

    Default: 120-day window with 5-day skip captures the classic
    intermediate-horizon momentum effect, decorrelated from the
    short-term 20-day momentum signal.
    """
    start_offset = window + skip_recent
    end_offset   = skip_recent + 1

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