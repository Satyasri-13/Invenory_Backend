from fastapi import APIRouter, Request, HTTPException
from typing import List, Optional
from fastapi import Query
import pandas as pd
import numpy as np

from app.utils.thresholds import waste_trend_arrow

router = APIRouter(
    prefix="/risk",
    tags=["Risk"]
)

# =================================================
# Helpers
# =================================================
def get_df(request: Request) -> pd.DataFrame:
    if not hasattr(request.app.state, "df"):
        raise HTTPException(400, "Dataset not uploaded yet")
    return request.app.state.df.copy()


def get_dq(request: Request) -> pd.DataFrame:
    if not hasattr(request.app.state, "data_dist_quarter"):
        raise HTTPException(400, "Distributor-quarter data not built")
    return request.app.state.data_dist_quarter.copy()

# =================================================
# 1️⃣ RISK OVERVIEW  (OVERVIEW TAB)
# =================================================
@router.get("/overview")
def risk_overview(
    request: Request,
    year: Optional[List[int]] = Query(None),
    month: Optional[List[str]] = Query(None),
    state: Optional[List[str]] = Query(None),           # accepted but ignored here
    distributor: Optional[List[str]] = Query(None),     # accepted but ignored here
):
    """
    OVERVIEW MODE (GLOBAL SUMMARY)

    ✔ Matches Lovable UI
    ✔ Supports multi-select filters
    ✔ Uses ONLY Year + Month
    ✔ Ignores State & Distributor for charts
    """

    # =================================================
    # BASE DATA
    # =================================================
    df = get_df(request)

    # --- Parse Month-Year (same as Streamlit) ---
    df["Month_Parsed"] = pd.to_datetime(
        df["Months"], format="%b-%y", errors="coerce"
    )
    df["Year_Parsed"] = df["Month_Parsed"].dt.year
    df["Month_Name"] = df["Month_Parsed"].dt.strftime("%b")

    # =================================================
    # APPLY TIME FILTERS ONLY (MULTI-SELECT SAFE)
    # =================================================
    if year and "All Years" not in year:
        df = df[df["Year_Parsed"].isin(year)]

    if month and "All Months" not in month:
        df = df[df["Month_Name"].isin(month)]

    # =================================================
    # STATE-WISE WASTE (TOP 10 STATES)
    # =================================================
    state_wise = (
        df.groupby("US States", as_index=False)["Waste_Quantity_Sum"]
        .sum()
        .sort_values("Waste_Quantity_Sum", ascending=False)
        .head(10)
    )

    state_wise_payload = [
        {
            "state": r["US States"],
            "value": round(r["Waste_Quantity_Sum"], 2)
        }
        for _, r in state_wise.iterrows()
    ]

    # =================================================
    # DISTRIBUTOR RISK (GLOBAL, YEAR FILTER ONLY)
    # =================================================
    dq = get_dq(request)

    if year and "All Years" not in year:
        dq = dq[dq["year_only"].isin(year)]

    dist_risk = (
        dq.groupby(["Distributor ID", "US States"], as_index=False)
        .agg(
            total_waste=("total_waste", "sum"),
            avg_pct_from_limit=("pct_from_limit", "mean")
        )
    )

    # --- Normalize risk to UI-friendly 0–100 ---
    dist_risk["risk_pct"] = (
        dist_risk["avg_pct_from_limit"]
        .clip(lower=0, upper=100)
        .round(1)
    )

    dist_risk["status"] = np.where(
        dist_risk["risk_pct"] >= 80, "High Risk",
        np.where(dist_risk["risk_pct"] >= 60, "Risk", "OK")
    )

    top_risky = (
        dist_risk
        .sort_values("risk_pct", ascending=False)
        .head(5)
    )

    high_risk_payload = [
        {
            "distributor_id": int(r["Distributor ID"]),
            "state": r["US States"],
            "risk_pct": r["risk_pct"],
            "status": r["status"]
        }
        for _, r in top_risky.iterrows()
    ]

    # =================================================
    # KEY INSIGHTS (GLOBAL STORY)
    # =================================================
    insights = []

    total_waste = df["Waste_Quantity_Sum"].sum()

    # ---- Insight 1: Top state contribution ----
    if not state_wise.empty and total_waste > 0:
        top_state = state_wise.iloc[0]
        pct = (top_state["Waste_Quantity_Sum"] / total_waste) * 100
        insights.append(
            f"{top_state['US States']} accounts for "
            f"{pct:.0f}% of total stale inventory losses."
        )

    # ---- Insight 2: Top-5 distributor concentration ----
    top5_ids = top_risky["Distributor ID"].tolist()

    top5_waste = (
        dq[dq["Distributor ID"].isin(top5_ids)]
        .groupby("Distributor ID")["total_waste"]
        .sum()
        .sum()
    )

    pct_top5 = (top5_waste / total_waste) * 100 if total_waste else 0

    insights.append(
        f"Top 5 distributors contribute to "
        f"{pct_top5:.0f}% of total waste."
    )

    # =================================================
    # FINAL RESPONSE (MATCHES UI EXACTLY)
    # =================================================
    return {
        "state_wise_waste": state_wise_payload,
        "high_risk_distributors": high_risk_payload,
        "key_insights": insights
    }

# =================================================
 #2... DISTRIBUTOR TREND (Single Distributor)
# =================================================
@router.get("/distributor-trend")
def distributor_trend(
    request: Request,
    distributor_id: str
):
    """
    Single Distributor Trend

    ✔ Requires Distributor ID only
    ✔ Quarter-wise waste trend
    ✔ Uses distributor-quarter dataset
    """

    dq = get_dq(request)

    # -------------------------------
    # FILTER: Distributor (REQUIRED)
    # -------------------------------
    df = dq[dq["Distributor ID"].astype(str) == distributor_id]

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No data found for selected distributor"
        )

    # -------------------------------
    # SORT BY TIME
    # -------------------------------
    df = df.sort_values(["year_only", "quarter"])

    # -------------------------------
    # RESPONSE FORMAT (UI READY)
    # -------------------------------
    return {
        "distributor_id": distributor_id,
        "trend": [
            {
                "quarter": f"{row['year_only']} {row['quarter']}",
                "waste": round(row["total_waste"], 2),
                "pct_change": (
                    None
                    if pd.isna(row["pct_change_Wastes_from_last_quarter"])
                    else round(row["pct_change_Wastes_from_last_quarter"], 2)
                ),
                "status": row["Status"]
            }
            for _, row in df.iterrows()
        ]
    }
# =================================================
# 3️⃣ QUARTER COMPARISON (UI ALIGNED)
# =================================================
@router.get("/quarter-comparison")
def quarter_comparison(
    request: Request,
    state: str,
    quarter_a: str,
    quarter_b: str,
    distributor_1: str,
    distributor_2: str | None = None,
):
    dq = get_dq(request)

    # -------------------------------
    # State filter (REQUIRED)
    # -------------------------------
    dq = dq[dq["US States"] == state]

    if dq.empty:
        raise HTTPException(404, "No data for selected state")

    # -------------------------------
    # Parse quarter labels safely
    # Format: "2022 2022Q2"
    # -------------------------------
    def parse_quarter(q_label: str):
        y, q = q_label.split()
        # "2022Q2" → "Q2"
        q = q.replace(str(y), "")
        return int(y), q

    y1, q1 = parse_quarter(quarter_a)
    y2, q2 = parse_quarter(quarter_b)

    df_q1 = dq[(dq["year_only"] == y1) & (dq["quarter"] == q1)]
    df_q2 = dq[(dq["year_only"] == y2) & (dq["quarter"] == q2)]

    # -------------------------------
    # Distributor selection
    # -------------------------------
    distributors = [distributor_1]
    if distributor_2:
        distributors.append(distributor_2)

    df_q1 = df_q1[df_q1["Distributor ID"].astype(str).isin(distributors)]
    df_q2 = df_q2[df_q2["Distributor ID"].astype(str).isin(distributors)]

    # =================================================
    # CASE 1: SAME QUARTER (A == B)
    # =================================================
    if quarter_a == quarter_b:
        if df_q1.empty:
            return {"comparison": []}

        base = df_q1.iloc[0]

        return {
            "state": state,
            "quarter_a": quarter_a,
            "quarter_b": quarter_b,
            "comparison": [
                {
                    "distributor_id": base["Distributor ID"],
                    "total_waste_q1": round(base["total_waste"], 2),
                    "total_waste_q2": round(base["total_waste"], 2),
                    "delta": 0,
                    "trend": "➖",
                    "status_change": f'{base["Status"]} → {base["Status"]}',
                }
            ],
        }

    # =================================================
    # CASE 2: DIFFERENT QUARTERS
    # =================================================
    merged = df_q1.merge(
        df_q2,
        on="Distributor ID",
        how="outer",
        suffixes=("_q1", "_q2"),
    )

    if merged.empty:
        return {"comparison": []}

    merged["delta"] = (
        merged["total_waste_q2"].fillna(0)
        - merged["total_waste_q1"].fillna(0)
    )

    merged["trend"] = merged["delta"].apply(waste_trend_arrow)

    merged["status_change"] = (
        merged["Status_q1"].fillna("Unknown")
        + " → "
        + merged["Status_q2"].fillna("Unknown")
    )

    return {
        "state": state,
        "quarter_a": quarter_a,
        "quarter_b": quarter_b,
        "comparison": [
            {
                "distributor_id": r["Distributor ID"],
                "total_waste_q1": round(r["total_waste_q1"], 2)
                if not pd.isna(r["total_waste_q1"]) else 0,
                "total_waste_q2": round(r["total_waste_q2"], 2)
                if not pd.isna(r["total_waste_q2"]) else 0,
                "delta": round(r["delta"], 2),
                "trend": r["trend"],
                "status_change": r["status_change"],
            }
            for _, r in merged.iterrows()
        ],
    }

# =================================================
# 4️⃣ TOP RISKY DISTRIBUTORS
# =================================================
@router.get("/top-risky")
def top_risky_distributors(request: Request):
    dq = get_dq(request)

    top = (
        dq.sort_values("pct_from_limit", ascending=False)
        .head(5)
    )

    return [
        {
            "distributor_id": r["Distributor ID"],
            "state": r["US States"],
            "risk_pct": round(r["pct_from_limit"], 1),
            "status": r["Status"]
        }
        for _, r in top.iterrows()
    ]
