import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

surge_predictions_schema = DataFrameSchema(
    {
        "date": Column(pa.DateTime),
    },
    strict=False,
    coerce=True,
)

nbhd_predictions_schema = DataFrameSchema(
    {
        "date": Column(pa.DateTime),
        "nbhd_id": Column(int, Check.ge(1)),
    },
    strict=False,
    coerce=True,
)