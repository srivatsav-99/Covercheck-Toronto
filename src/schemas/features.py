import pandera.pandas as pa
from pandera.typing import Series
import pandas as pd


class FeatureSchema(pa.DataFrameModel):
    """Strict validation contract for the Feature Builder output."""
    date: Series[pd.Timestamp]
    nbhd_id: Series[int] = pa.Field(ge=1)

    class Config:
        strict = False  # Allows extra columns to pass through
        coerce = True  # Automatically fixes minor datatype issues