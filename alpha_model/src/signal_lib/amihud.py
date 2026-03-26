

def compute_amihud_expr(window: int = 20) -> str:
    """
    Rolling Amihud illiquidity:
        avg( |ret| / dollar_volume )

    Dollar volume approximated as:
        abs(prc) * vol

    Uses trailing window excluding current row.
    """
    return f"""
    AVG(
        CASE
            WHEN ret IS NOT NULL
             AND prc IS NOT NULL
             AND vol IS NOT NULL
             AND ABS(prc) > 0
             AND vol > 0
            THEN ABS(ret) / (ABS(prc) * vol)
            ELSE NULL
        END
    ) OVER (
        PARTITION BY permno
        ORDER BY date
        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
    )
    """