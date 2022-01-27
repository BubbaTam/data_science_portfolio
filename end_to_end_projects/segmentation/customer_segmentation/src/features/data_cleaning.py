#what to do
# quality of life changes
# missing data
import os

from pandas.core.frame import DataFrame
from scipy.sparse import data 
from src.config import PARAMETERS_OUTPUT
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder ,MinMaxScaler, StandardScaler, Normalizer
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from joblib import dump
from dataclasses import dataclass


processed_features = ['gender','age','annual_income_(k$)','spending_score_(1-100)']
processed_dependent_var = None
processed_independent_var = processed_features.remove(processed_dependent_var)

processed_continuous_var = ['age','annual_income_(k$)','spending_score_(1-100)']
processed_nominal_var = ['gender']


@dataclass
class RawDataCleaning():
    column_names : list = None
    independent_var : list = None
    dependent_var : list = None
    continuous_var : list = None
    nominal_var : list = None
    ordinal_var : list = None

    def qol_column_names(dataframe):
        """[current iteration puts the column labels into lowercase and replaces a spaces with underscores]

        Args:
            dataframe ([pandas dataframe])
        """
        dataframe.columns = [column.lower().replace(' ','_').replace("-","-").replace("?","_").replace("/","_").replace("\\","_").replace("%","_") \
                                    .replace("(","").replace(")","").replace("$","") for column in dataframe.columns]











