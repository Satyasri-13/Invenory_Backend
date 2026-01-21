import pandas as pd


def waste_trend_arrow(delta):
    if pd.isna(delta):
        return ""
    if delta > 0:
        return "⬆️ 🔴"
    elif delta < 0:
        return "⬇️ 🟢"
    else:
        return "➖"


def distributor_status(pct_from_limit: float, pct_change: float) -> str:
    """
    Classify distributor risk using limit first, then trend.
    """

    # 1️⃣ Primary: limit-based risk
    if not pd.isna(pct_from_limit):
        if pct_from_limit >= 120:
            return "High Risk"
        elif pct_from_limit >= 100:
            return "Risk"
        elif pct_from_limit < 80:
            return "Very Good"
        else:
            return "Good"

    # 2️⃣ Secondary: trend-based risk
    if not pd.isna(pct_change):
        if pct_change > 10:
            return "High Risk"
        elif pct_change > 0:
            return "Risk"
        elif pct_change < -10:
            return "Very Good"
        else:
            return "Good"

    # 3️⃣ Fallback
    return "Not Classified"
