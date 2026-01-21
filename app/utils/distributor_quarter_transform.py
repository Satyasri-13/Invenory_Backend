import numpy as np
import pandas as pd
from app.utils.thresholds import distributor_status



def build_distributor_quarter_df(eda_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds distributor-quarter level dataset with waste metrics
    and risk classification.
    """

    df = eda_df.copy()

    required_cols = [
        "Distributor ID",
        "US States",
        "year_only",
        "quarter",
        "Deliveries_Quantity",
        "Returns_Quantity",
        "Waste_Allowance_Quantity",
        "Waste_Quantity_Sum",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    idp_level_B = (
        df.groupby(
            ["Distributor ID", "US States", "year_only", "quarter"],
            as_index=False
        )
        .agg(
            total_deliveries=("Deliveries_Quantity", "sum"),
            total_returns=("Returns_Quantity", "sum"),
            total_waste_allowance=("Waste_Allowance_Quantity", "sum"),
            total_waste=("Waste_Quantity_Sum", "sum"),
        )
    )

    idp_level_B[
        ["total_deliveries", "total_returns", "total_waste_allowance", "total_waste"]
    ] = idp_level_B[
        ["total_deliveries", "total_returns", "total_waste_allowance", "total_waste"]
    ].round(2)

    limit = idp_level_B["total_waste_allowance"]
    spoil = idp_level_B["total_waste"]

    idp_level_B["pct_from_limit"] = np.where(
        (limit == 0) | (spoil == 0),
        0,
        ((spoil - limit) / limit) * 100,
    ).round(2)

    idp_level_B = idp_level_B.sort_values(
        ["Distributor ID", "year_only", "quarter"]
    )

    idp_level_B["pct_change_Wastes_from_last_quarter"] = (
        idp_level_B
        .groupby("Distributor ID")["total_waste"]
        .pct_change() * 100
    ).round(2)

    idp_level_B["Status"] = idp_level_B.apply(
        lambda row: distributor_status(
            row["pct_from_limit"],
            row["pct_change_Wastes_from_last_quarter"],
        ),
        axis=1
    )

    return idp_level_B
