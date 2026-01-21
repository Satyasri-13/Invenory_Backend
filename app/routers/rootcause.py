from fastapi import APIRouter, Request, HTTPException
import pandas as pd

router = APIRouter(prefix="/root-cause", tags=["Root Cause"])

@router.get("/")
def root_cause_analysis(request: Request):
    if not hasattr(request.app.state, "model_lab"):
        raise HTTPException(400, "Model not trained yet")

    model_lab = request.app.state.model_lab

    shap = model_lab.get("shap")
    if not shap:
        raise HTTPException(400, "SHAP data not available")

    df = pd.DataFrame(shap)

    # Normalize to %
    df["contribution_pct"] = df["importance"] / df["importance"].sum() * 100
    df = df.sort_values("contribution_pct", ascending=False)

    top_factors = (
        df.head(5)[["feature", "contribution_pct"]]
        .round(2)
        .to_dict(orient="records")
    )

    primary = top_factors[0]

    response = {
        "top_factors": top_factors,
        "primary_cause": {
            "feature": primary["feature"],
            "reason": "Primary driver based on highest model contribution"
        },
        "secondary_drivers": [
            {
                "feature": f["feature"],
                "reason": "Secondary contributor to inventory loss"
            }
            for f in top_factors[1:3]
        ],
        "recommended_actions": [
            "Improve return handling for top-risk distributors",
            "Optimize delivery quantities using demand signals",
            "Reduce storage duration for slow-moving inventory"
        ]
    }

    return response
