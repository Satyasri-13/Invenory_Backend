from fastapi import APIRouter, UploadFile, File, Request, HTTPException
import pandas as pd

from app.utils.filters import prepare_time_columns
from app.utils.distributor_quarter_transform import build_distributor_quarter_df

router = APIRouter(
    prefix="/upload",
    tags=["Upload"]
)

@router.post("/")
def upload_dataset(
    request: Request,
    file: UploadFile = File(...)
):
    # -------------------------------
    # Read file
    # -------------------------------
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    else:
        df = pd.read_excel(file.file)

    # -------------------------------
    # 🔑 VERY IMPORTANT STEP
    # Build time columns FIRST
    # -------------------------------
    df = prepare_time_columns(df)
    # This creates: year_only, month_only, quarter

    # -------------------------------
    # Store base dataset
    # -------------------------------
    request.app.state.df = df

    # -------------------------------
    # Build distributor-quarter dataset
    # -------------------------------
    try:
        data_dist_quarter = build_distributor_quarter_df(df)
        request.app.state.data_dist_quarter = data_dist_quarter
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Distributor-quarter build failed: {str(e)}"
        )

    return {
        "message": "Dataset uploaded successfully",
        "rows": df.shape[0],
        "columns": df.columns.tolist()
    }
