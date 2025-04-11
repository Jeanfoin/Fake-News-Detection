import pandas as pd
from typing import Iterable


def extract_outliers_IQR(
    df: pd.DataFrame, subset: Iterable[str] = None, scal: float = 1.5
) -> pd.Index:
    """
    Extract all outliers from a given pandas DataFrame using
    InterQuartile Range (IQR) method
    """

    outliers_index = set()

    if subset is None:
        subset = df.select_dtypes("number").columns
    for col in subset:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        new_index = df[
            ((df[col] < Q1 - scal * IQR) | (df[col] > Q3 + scal * IQR))
        ].index
        outliers_index.update(new_index)

    return pd.Index(outliers_index)