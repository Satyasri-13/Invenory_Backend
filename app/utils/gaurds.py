from fastapi import HTTPException, Request

def get_df(request: Request):
    if not hasattr(request.app.state, "df"):
        raise HTTPException(
            status_code=400,
            detail="Dataset not uploaded. Please upload first."
        )
    return request.app.state.df