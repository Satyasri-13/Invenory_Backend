import pandas as pd

def prepare_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - year_only
    - month_only
    - quarter

    Expects column:
    - Months (e.g., Feb-23)
    """

    df = df.copy()

    # Parse Month-Year safely
    df["Month_Parsed"] = pd.to_datetime(
        df["Months"], format="%b-%y", errors="coerce"
    )

    df["year_only"] = df["Month_Parsed"].dt.year
    df["month_only"] = df["Month_Parsed"].dt.month

    df["quarter"] = (
        "Q" + ((df["month_only"] - 1) // 3 + 1).astype("Int64").astype(str)
    )

    return df
