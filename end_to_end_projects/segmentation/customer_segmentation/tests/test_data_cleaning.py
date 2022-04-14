# to be tested
## [check future] I think there is a better way to test


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
    """
    Args:
        column_name (list[str]): _description_
        post_column_name (list[str]): _description_
    """
    data = np.array([(1, 2), (3, 4)])
    df = pd.DataFrame(data=data,columns=column_name)
    class_instance = data_cleaning.DataCleaning()
    class_instance.clean_df_columns(df)
    assert all(df.columns == post_column_name)

@pytest.mark.parametrize(
    "features_orig,features_post,independent_feat_orig,independent_feat_post,\
        dependent_feat_orig,dependent_feat_post,continuous_feat_orig,\
            continuous_feat_post,nominal_feat_orig,nominal_feat_post,\
                ordinal_feat_orig,ordinal_feat_post",
    [
        (
            # data for all instance variables
            ["INDEPENDENT_1","CONTINUOUS_1","NOMINAL_1","ORDINAL_1","DEPENDENT_1"],
            ["independent_1","continuous_1","nominal_1","ordinal_1","dependent_1"],
            ["INDEPENDENT_1","CONTINUOUS_1","NOMINAL_1","ORDINAL_1"],
            ["independent_1","continuous_1","nominal_1","ordinal_1"],
            ["DEPENDENT_1"],
            ["dependent_1"],
            ["CONTINUOUS_1"],
            ["continuous_1"],
            ["NOMINAL_1"],
            ["nominal_1"],
            ["ORDINAL_1"],
            ["ordinal_1"]
        ),
        # test for empty variables
        (
            ["INDEPENDENT_1"],
            ["independent_1"],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            []
        ),
    ]
    )

def test_FeatureIdentification_clean_instance_variables(features_orig,
                                                        features_post,
                                                        independent_feat_orig,
                                                        independent_feat_post,
                                                        dependent_feat_orig,
                                                        dependent_feat_post,
                                                        continuous_feat_orig,
                                                        continuous_feat_post,
                                                        nominal_feat_orig,
                                                        nominal_feat_post,
                                                        ordinal_feat_orig,
                                                        ordinal_feat_post):
    """
    - Want to test when all initiation variables
    - When there are variables missing (None and empty list)
    """
    before = data_cleaning.FeatureIdentification(features=features_orig,
                                        independent_feat=independent_feat_orig,
                                        dependent_feat=dependent_feat_orig,
                                        continuous_feat=continuous_feat_orig,
                                        nominal_feat=nominal_feat_orig,
                                        ordinal_feat=ordinal_feat_orig)
    before.clean_instance_variables()
    after = data_cleaning.FeatureIdentification(features=features_post,
                                        independent_feat=independent_feat_post,
                                        dependent_feat=dependent_feat_post,
                                        continuous_feat=continuous_feat_post,
                                        nominal_feat=nominal_feat_post,
                                        ordinal_feat=ordinal_feat_post)
    assert before == after