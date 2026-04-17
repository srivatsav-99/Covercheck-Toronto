import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

gold_schema = DataFrameSchema(
    {
        "date": Column(pa.DateTime),
        "nbhd_id": Column(int, Check.ge(1)),
        "collisions": Column(float, Check.ge(0)),
    },
    strict=False,
    coerce=True,
)