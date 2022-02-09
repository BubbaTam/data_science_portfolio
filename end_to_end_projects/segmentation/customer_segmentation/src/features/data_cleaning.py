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
    """ An interface for scaling numerical data """
    @abstractmethod
    def scale_numerical_data(self):
        """ scale the data for a feature"""

class OrdinalEncoding(ABC):
    """ An interface for transforming ordinal data """
    @abstractmethod
    def map_ordinal(self):
        """ transform ordinal data """

@dataclass
class MinMaxScalerParameters(ScaleData):

    data : list # to change this to a customisable typing
    range: tuple = (0,1)

    def scale_numerical_data(self,headers,save_scaler_parameter : bool = False):
        """
        [an implementation of ScaleData that uses MinMaxScaler to scale
        a pandas dataframe with relevant column headers by the instantiated range]

        Args:
            headers ([list]): [The features of the pandas dataframe to be scaled]
            save_scaler_parameter (bool, optional): [Send the parameters of the scaler to save_entities folder]. Defaults to False.
        """
        scaler = MinMaxScaler(feature_range=self.range)
        scaler_parameters = scaler.fit(self.data[headers])
        scaled_features = scaler_parameters.transform(self.data[headers])
        self.data[headers] = scaled_features
        if save_scaler_parameter is not False:
            dump(scaler_parameters,os.path.join(PARAMETERS_OUTPUT,"min_max_scale.bin"))

@dataclass
class Standardisation(ScaleData):

    data : list # to change this to a customisable typing

    def scale_numerical_data(self,headers,save_scaler_parameter : bool = False):
        """
        [an implementation of ScaleData that uses StandardScaler to scale
        a pandas dataframe with relevant column headers]

        Args:
            headers ([list]): [The features of the pandas dataframe to be scaled]
            save_scaler_parameter (bool, optional): [description]. Defaults to False.
        """
        scaler = StandardScaler()
        scaler_parameters = scaler.fit(self.data[headers])
        scaled_features = scaler_parameters.transform(self.data[headers])
        self.data[headers] = scaled_features
        if save_scaler_parameter is not False:
            dump(scaler_parameters,os.path.join(PARAMETERS_OUTPUT,"standardisation.bin"))

@dataclass
class Normalisation(ScaleData):

    data : list =  None # to change this to a customisable typing

    def scale_numerical_data(self,headers,save_scaler_parameter : bool = False):
        """
        [An implementation of ScaleData that uses Normalizer to scale
        a pandas dataframe with relevant column headers]

        Args:
            headers ([list]): [The features of the pandas dataframe to be scaled]
            save_scaler_parameter (bool, optional): [Send the parameters of the scaler to save_entities folder]. Defaults to False.
        """
        scaler = Normalizer()
        scaler_parameters = scaler.fit(self.data[headers])
        scaled_features = scaler_parameters.transform(self.data[headers])
        self.data[headers] = scaled_features
        if save_scaler_parameter is not False:
            dump(scaler_parameters,os.path.join(PARAMETERS_OUTPUT,"normalisation.bin"))
        return scaled_features


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