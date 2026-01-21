from fastapi import APIRouter, Request, HTTPException
import pandas as pd
import numpy as np

from app.utils.thresholds import distributor_status

router = APIRouter(
    prefix="/inventory",
    tags=["Inventory"]
)

# =================================================
# Helper: get dataset from app state
# =================================================
def get_df(request: Request) -> pd.DataFrame:
    if not hasattr(request.app.state, "df"):
        raise HTTPException(
            status_code=400,
            detail="Dataset not uploaded yet"
        )
    return request.app.state.df.copy()


# =================================================
# Status mapping (Business → UI friendly)
# =================================================
STATUS_MAP = {
    "High Risk": "Exceeded",
    "Risk": "At Risk",
    "Good": "OK",
    "Very Good": "OK",
    "Not Classified": "OK"
}


# =================================================
# 1️⃣ INVENTORY OVERVIEW (KPIs)
# =================================================
@router.get("/overview")
def inventory_overview(request: Request):
    df = get_df(request)

    # ---- Safe numeric conversion ----
    for col in ["Waste_Quantity_Sum", "Waste_Allowance_Quantity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    total_waste = df["Waste_Quantity_Sum"].sum()
    total_allowance = df["Waste_Allowance_Quantity"].sum()

    utilization_pct = (
        (total_waste / total_allowance) * 100
        if total_allowance > 0 else 0
    )

    # ---- High-risk states (>=80% utilization) ----
    state_risk = (
        df.groupby("US States", as_index=False)
        .agg(
            waste=("Waste_Quantity_Sum", "sum"),
            allowance=("Waste_Allowance_Quantity", "sum")
        )
    )

    state_risk["usage_pct"] = (
        state_risk["waste"] /
        state_risk["allowance"].replace(0, np.nan)
    ) * 100

    high_risk_states = state_risk[state_risk["usage_pct"] >= 80]

    return {
        "total_waste": {
            "value": round(total_waste, 2),
            "change_pct": 12.5   # placeholder
        },
        "total_allowance": {
            "value": round(total_allowance, 2),
        },
        "utilization_rate": {
            "value": round(utilization_pct, 1),
            "change_pct": -2.3   # placeholder
        },
        "high_risk_states": {
            "value": int(high_risk_states.shape[0]),
            "change": 2          # placeholder
        }
    }


# =================================================
# 2️⃣ INVENTORY CHARTS
# =================================================
@router.get("/charts")
def inventory_charts(request: Request):
    df = get_df(request)

    # ---- Build Month-Year safely ----
    df["Month_Year"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Months"].astype(str),
        errors="coerce"
    )

    monthly = (
        df.dropna(subset=["Month_Year"])
        .groupby("Month_Year", as_index=False)
        .agg(
            allowed=("Waste_Allowance_Quantity", "sum"),
            actual=("Waste_Quantity_Sum", "sum")
        )
        .sort_values("Month_Year")
        .tail(6)   # last 6 months
    )

    return {
        "allowed_vs_actual": [
            {
                "month": row["Month_Year"].strftime("%b"),
                "allowed": round(row["allowed"], 2),
                "actual": round(row["actual"], 2)
            }
            for _, row in monthly.iterrows()
        ],
        "loss_trend": [
            {
                "month": row["Month_Year"].strftime("%b"),
                "value": round(row["actual"], 2)
            }
            for _, row in monthly.iterrows()
        ]
    }


# =================================================
# 3️⃣ DISTRIBUTOR STALE ALLOWANCE STATUS
# =================================================
@router.get("/distributor-status")
def distributor_status_table(request: Request):
    df = get_df(request)

    # ---- Force numeric ----
    for col in ["Waste_Allowance_Quantity", "Waste_Quantity_Sum"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Compute medians ----
    median_allowance = df["Waste_Allowance_Quantity"].median()
    median_waste = df["Waste_Quantity_Sum"].median()

    # ---- Fill missing with MEDIAN ----
    df["Waste_Allowance_Quantity"] = df["Waste_Allowance_Quantity"].fillna(median_allowance)
    df["Waste_Quantity_Sum"] = df["Waste_Quantity_Sum"].fillna(median_waste)

    dist = (
        df.groupby("Distributor ID", as_index=False)
        .agg(
            allowance=("Waste_Allowance_Quantity", "sum"),
            actual_waste=("Waste_Quantity_Sum", "sum")
        )
    )

    # ---- % utilization ----
    dist["utilization_pct"] = (
        dist["actual_waste"] /
        dist["allowance"].replace(0, median_allowance)
    ) * 100

    # ---- % deviation from limit ----
    dist["pct_from_limit"] = (
        (dist["actual_waste"] - dist["allowance"]) /
        dist["allowance"].replace(0, median_allowance)
    ) * 100

    # ---- Risk classification ----
    dist["risk_status"] = dist["pct_from_limit"].apply(
        lambda x: distributor_status(x, np.nan)
    )

    # ---- UI status mapping ----
    dist["status"] = dist["risk_status"].apply(
        lambda x: STATUS_MAP.get(x, "OK")
    )

    # ---- JSON-safe response ----
    return [
        {
            "distributor_id": str(row["Distributor ID"]),
            "allowance": round(float(row["allowance"]), 2),
            "actual_waste": round(float(row["actual_waste"]), 2),
            "utilization_pct": round(float(row["utilization_pct"]), 1),
            "status": row["status"]
        }
        for _, row in
        dist.sort_values("utilization_pct", ascending=False).iterrows()
    ]
