#Possible to look at:
# quality of life changes
# missing data

#current implementations


#future improvements


import os
from src.config import PARAMETERS_OUTPUT
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder ,MinMaxScaler, StandardScaler, Normalizer
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from joblib import dump
from dataclasses import dataclass

class ScaleData(ABC):

    @abstractmethod
    def scale_numerical_data(self):
        """ scale the data for a feature"""

class OrdinalEncoding(ABC):
    @abstractmethod
    def map_ordinal(self):
        """ an interface for mapping ordinal data """

@dataclass
class FeatureIdentification():
    """ A collection of the feature types. The plan is to use this to make the process of data feature easier.
        I should be able to sub """
    features : list = None
    independent_feat : list = None
    dependent_feat : str = None
    continuous_feat : list = None
    nominal_feat : list = None
    ordinal_feat : list = None

    def clean_instance_variables(self):
        for key in self.__dict__:
            if self.__dict__[key] is not None:
                self.__dict__[key] = [_.lower().replace(' ','_').replace("-","_").replace("?","_").replace("/","_").replace("\\","_").replace("%","_") \
                                .replace("(","").replace(")","").replace("$","") for _ in self.__dict__[key]]
class RawDataCleaning():
    def clean_df_columns(dataframe):
        """[current iteration replaces spaces, dashes, question mark, slashes, percentage and brackets with space
        and dollar symbol with space]

        Args:
            dataframe ([pandas dataframe])
        """
        dataframe.columns = [column.lower().replace(' ','_').replace("-","_").replace("?","_").replace("/","_").replace("\\","_").replace("%","_") \
                                    .replace("(","").replace(")","").replace("$","") for column in dataframe.columns]