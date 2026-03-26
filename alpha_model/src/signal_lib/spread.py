

def compute_spread_expr() -> str:
    """
    Relative bid-ask spread using current day's bid/ask.

        spread = (ask - bid) / mid
        mid    = (ask + bid)/2

    If you want pure ex-ante prediction discipline, you may later shift this by 1 day.
    """
    return """
    CASE
        WHEN ask IS NOT NULL
         AND bid IS NOT NULL
         AND ask > 0
         AND bid > 0
         AND (ask + bid) <> 0
        THEN (ask - bid) / ((ask + bid) / 2.0)
        ELSE NULL
    END
    """