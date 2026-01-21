from fastapi import APIRouter, Request, HTTPException
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap

router = APIRouter(prefix="/model", tags=["Model Lab"])

TARGET = "Waste_Quantity_Sum"

FEATURES = [
    "Distributor_Efficiency_by_Return_Rate",
    "Deliveries_Quantity",
    "Waste_Allowance_Quantity",
    "Waste_Rate_by_Region",
    "Base_Price_by_Distributor",
    "Shipment_Turnover_Ratio",
]

# -------------------------------------------------
# Model factory
# -------------------------------------------------
def get_model(name: str):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "Decision Tree": DecisionTreeRegressor(max_depth=6),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        ),
    }
    if name not in models:
        raise HTTPException(400, f"Unsupported model: {name}")
    return models[name]

# -------------------------------------------------
# Preprocessor (MATCHES STREAMLIT LOGIC)
# -------------------------------------------------
def build_preprocessor():
    yeo_cols = [
        "Distributor_Efficiency_by_Return_Rate",
        "Deliveries_Quantity",
        "Waste_Allowance_Quantity",
        "Base_Price_by_Distributor",
        "Shipment_Turnover_Ratio",
    ]

    passthrough_cols = ["Waste_Rate_by_Region"]

    yeo_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("yeo", PowerTransformer(method="yeo-johnson")),
            ("scale", StandardScaler()),
        ]
    )

    pass_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("yeo", yeo_pipe, yeo_cols),
            ("pass", pass_pipe, passthrough_cols),
        ]
    )

# -------------------------------------------------
# Helper: MALE
# -------------------------------------------------
def mean_absolute_log_error(y_true, y_pred, eps=1e-9):
    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)
    return np.mean(np.abs(np.log(y_true) - np.log(y_pred)))

# =================================================
# TRAIN MODEL
# =================================================
@router.post("/train")
def train_model(request: Request, model_name: str):

    if not hasattr(request.app.state, "df"):
        raise HTTPException(400, "Dataset not loaded")

    df = request.app.state.df.copy()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    preprocess = build_preprocessor()
    model = get_model(model_name)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = {
        "mae": round(mean_absolute_error(y_test, preds), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "r2": round(r2_score(y_test, preds) * 100, 2),
        "male": round(mean_absolute_log_error(y_test, preds), 4),
    }

    # ---------------- SHAP ----------------
    X_t = preprocess.transform(X_test)

    if model_name in ["Decision Tree", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_t)
    else:
        explainer = shap.LinearExplainer(model, X_t)
        shap_values = explainer.shap_values(X_t)

    shap_importance = np.abs(shap_values).mean(axis=0)

    shap_df = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": shap_importance,
            }
        )
        .sort_values("importance", ascending=False)
    )

    coef_df = None
    if model_name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
        coef_df = (
            pd.DataFrame(
                {
                    "feature": FEATURES,
                    "coefficient": model.coef_,
                }
            )
            .sort_values("coefficient", ascending=False)
        )

    request.app.state.model_lab = {
        "model": model_name,
        "metrics": metrics,
        "shap": shap_df.to_dict("records"),
        "coefficients": None if coef_df is None else coef_df.to_dict("records"),
    }

    return {
        "status": "trained",
        "model": model_name,
        "metrics": metrics,
    }

# =================================================
# FETCH RESULTS
# =================================================
@router.get("/results")
def get_results(request: Request):
    if not hasattr(request.app.state, "model_lab"):
        raise HTTPException(400, "Model not trained yet")
    return request.app.state.model_lab
