import pandas as pd

_DATASET: pd.DataFrame | None = None

def set_dataset(df: pd.DataFrame):
    global _DATASET
    _DATASET = df

def get_dataset() -> pd.DataFrame:
    if _DATASET is None:
        raise ValueError("Dataset not uploaded")
    return _DATASET
