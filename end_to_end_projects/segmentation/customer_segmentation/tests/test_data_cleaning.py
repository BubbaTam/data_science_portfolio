# to be tested
##


import pytest
import numpy as np
import pandas as pd
import os
import sys

from src.features import data_cleaning


@pytest.mark.parametrize(
    "column_name, post_column_name",
    [
        (["DOG","FISH"], ["dog","fish"]),
        (["EVER1","F1Sh"], ["ever1","f1sh"])
    ])

def test_Datacleaning(column_name,post_column_name):
    ""
    data = np.array([(1, 2), (3, 4)])
    df = pd.DataFrame(data=data,columns=column_name)
    class_instance = data_cleaning.DataCleaning()
    class_instance.clean_df_columns(df)
    assert all(df.columns == post_column_name)