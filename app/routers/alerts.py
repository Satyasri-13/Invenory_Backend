from fastapi import APIRouter, Request, HTTPException
import pandas as pd

router = APIRouter(
    prefix="/alerts",
    tags=["Alerts"]
)

# -----------------------------------------
# Helper
# -----------------------------------------
def get_df(request: Request) -> pd.DataFrame:
    if not hasattr(request.app.state, "df"):
        raise HTTPException(400, "Dataset not uploaded yet")
    return request.app.state.df.copy()


# =================================================
# ALERTS API
# =================================================
@router.get("/")
def get_alerts(
    request: Request,
    severity: str = "ALL",        # ALL | HIGH | MEDIUM | LOW
    distributor: str = "ALL",
    state: str = "ALL"
):
    severity = severity.upper()

    df = get_df(request)

    # -------------------------------
    # Cleanup
    # -------------------------------
    df.columns = df.columns.str.strip()

    numeric_cols = [
        "Waste_Quantity_Sum",
        "Waste_Allowance_Quantity",
        "Returns_Quantity",
        "Deliveries_Quantity"
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Distributor ID", "US States"])

    alerts = []

    # =================================================
    # HIGH — Waste exceeded allowance
    # =================================================
    over_allow = (
        df.groupby(["Distributor ID", "US States"], as_index=False)
        .agg(
            waste=("Waste_Quantity_Sum", "sum"),
            allowance=("Waste_Allowance_Quantity", "sum")
        )
    )

    over_allow = over_allow[over_allow["allowance"] > 0]
    over_allow["usage_pct"] = over_allow["waste"] / over_allow["allowance"]

    for _, r in over_allow[over_allow["usage_pct"] > 1.0].iterrows():
        alerts.append({
            "severity": "HIGH",
            "title": "Waste Threshold Exceeded",
            "description": f"Waste exceeded allowance by {(r['usage_pct']-1)*100:.1f}%",
            "distributor_id": int(r["Distributor ID"]),
            "state": r["US States"],
            "category": "Stale Inventory",
            "time_ref": "Recent"
        })

    # =================================================
    # MEDIUM — High returns
    # =================================================
    returns = (
        df.groupby(["Distributor ID", "US States"], as_index=False)
        .agg(
            returns=("Returns_Quantity", "sum"),
            deliveries=("Deliveries_Quantity", "sum")
        )
    )

    returns = returns[returns["deliveries"] > 0]
    returns["return_pct"] = returns["returns"] / returns["deliveries"]

    for _, r in returns[returns["return_pct"] > 0.08].iterrows():
        alerts.append({
            "severity": "MEDIUM",
            "title": "High Return Rate",
            "description": f"Returns at {r['return_pct']*100:.1f}% of deliveries",
            "distributor_id": int(r["Distributor ID"]),
            "state": r["US States"],
            "category": "Returns",
            "time_ref": "Recent"
        })

    # =================================================
    # LOW — Good performance
    # =================================================
    for _, r in over_allow[over_allow["usage_pct"] < 0.6].iterrows():
        alerts.append({
            "severity": "LOW",
            "title": "Good Inventory Control",
            "description": "Waste well within allowed limits",
            "distributor_id": int(r["Distributor ID"]),
            "state": r["US States"],
            "category": "Positive Signal",
            "time_ref": "Recent"
        })

    alerts_df = pd.DataFrame(alerts)

    # =================================================
    # SUMMARY (unique distributors)
    # =================================================
    summary = (
        alerts_df
        .groupby("severity")["distributor_id"]
        .nunique()
        .to_dict()
    )

    summary = {
        "high": summary.get("HIGH", 0),
        "medium": summary.get("MEDIUM", 0),
        "low": summary.get("LOW", 0),
    }

    # =================================================
    # ONE ALERT PER DISTRIBUTOR (PRIORITY)
    # =================================================
    priority = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    alerts_df["priority"] = alerts_df["severity"].map(priority)

    alerts_df = (
        alerts_df
        .sort_values("priority", ascending=False)
        .drop_duplicates(subset=["distributor_id"])
    )

    # =================================================
    # FILTERS
    # =================================================
    if severity != "ALL":
        alerts_df = alerts_df[alerts_df["severity"] == severity]

    if distributor != "ALL":
        alerts_df = alerts_df[alerts_df["distributor_id"] == int(distributor)]

    if state != "ALL":
        alerts_df = alerts_df[alerts_df["state"] == state]

    return {
        "summary": summary,
        "alerts": alerts_df.drop(columns=["priority"]).to_dict(orient="records")
    }
