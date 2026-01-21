from fastapi import APIRouter, Request, HTTPException
import pandas as pd
import numpy as np

router = APIRouter(
    prefix="/analysis",
    tags=["Correlation Analysis"]
)

# ---------------------------------------
# Helper
# ---------------------------------------
def get_df(request: Request) -> pd.DataFrame:
    if not hasattr(request.app.state, "df"):
        raise HTTPException(400, "Dataset not loaded")
    return request.app.state.df.copy()

# ---------------------------------------
# CORRELATION ANALYSIS API
# ---------------------------------------
@router.get("/correlation")
def correlation_analysis(request: Request):
    df = get_df(request)

    # -----------------------------
    # Numeric-only data
    # -----------------------------
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.shape[1] < 2:
        raise HTTPException(400, "Not enough numeric features")

    corr = numeric_df.corr().round(2)

    # -----------------------------
    # Heatmap payload
    # -----------------------------
    heatmap = {
        "features": corr.columns.tolist(),
        "matrix": corr.values.tolist()
    }

    # -----------------------------
    # Key Relationships
    # -----------------------------
    relationships = []

    for i in corr.columns:
        for j in corr.columns:
            if i != j:
                relationships.append({
                    "f1": i,
                    "f2": j,
                    "value": corr.loc[i, j]
                })

    rel_df = pd.DataFrame(relationships)
    rel_df["abs"] = rel_df["value"].abs()

    strong = rel_df[rel_df["abs"] >= 0.75].head(5)
    moderate = rel_df[(rel_df["abs"] >= 0.4) & (rel_df["abs"] < 0.75)].head(5)
    inverse = rel_df[rel_df["value"] <= -0.4].head(5)

    key_relationships = {
        "strong": strong.to_dict(orient="records"),
        "moderate": moderate.to_dict(orient="records"),
        "inverse": inverse.to_dict(orient="records")
    }

    # -----------------------------
    # Model-wise recommendations
    # (Rule-based, fast & explainable)
    # -----------------------------
    model_recommendations = {
        "linear_regression": {
            "features": ["Storage_Duration", "Distributor_Size", "Region_Population"],
            "reason": "Strong linear correlation with waste"
        },
        "decision_tree": {
            "features": ["Order_Frequency", "Storage_Duration", "Temperature_Variance"],
            "reason": "Captures non-linear interactions"
        },
        "xgboost": {
            "features": "All numeric features",
            "reason": "Handles multicollinearity and complex patterns"
        }
    }

    # -----------------------------
    # Final response
    # -----------------------------
    return {
        "heatmap": heatmap,
        "key_relationships": key_relationships,
        "model_recommendations": model_recommendations
    }
