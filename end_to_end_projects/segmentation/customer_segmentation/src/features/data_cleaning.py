#Possible to look at:
# quality of life changes
# missing data
## what is the ideal way to indicate a pandas dataframe

#current implementations


#future improvements


import os
from src.config import PARAMETERS_OUTPUT
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder ,MinMaxScaler, StandardScaler, Normalizer
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
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

    data : pd.DataFrame # to change this to a customisable typing
    range: tuple = (0,1)

    def scale_numerical_data(self,headers,save_scaler_parameter : bool = False):
        """
        [an implementation of ScaleData that uses MinMaxScaler to scale
        a pandas dataframe with relevant column headers by the instantiated range]

        Args:
            headers ([list]): [The features of the pandas dataframe to be scaled]
            save_scaler_parameter (bool, optional): [Send the parameters of the scaler to save_entities folder]. Defaults to False.
        """
        if isinstance(self.data, pd.DataFrame): # not sure about this yet
            scaler = MinMaxScaler(feature_range=self.range)
            scaler_parameters = scaler.fit(self.data[headers])
            scaled_features = scaler_parameters.transform(self.data[headers])
            self.data[headers] = scaled_features
            if save_scaler_parameter is not False:
                dump(scaler_parameters,os.path.join(PARAMETERS_OUTPUT,"min_max_scale.bin"))
        else:
            raise("The input needs to be a pandas dataframe")

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
        if isinstance(self.data, pd.DataFrame): # not sure about this yet
            scaler = StandardScaler()
            scaler_parameters = scaler.fit(self.data[headers])
            scaled_features = scaler_parameters.transform(self.data[headers])
            self.data[headers] = scaled_features
            if save_scaler_parameter is not False:
                dump(scaler_parameters,os.path.join(PARAMETERS_OUTPUT,"standardisation.bin"))
        else:
            raise("The data needs to be a pandas dataframe")

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
        if isinstance(self.data, pd.DataFrame): # not sure about this yet
            scaler = Normalizer()
            scaler_parameters = scaler.fit(self.data[headers])
            scaled_features = scaler_parameters.transform(self.data[headers])
            self.data[headers] = scaled_features
            if save_scaler_parameter is not False:
                dump(scaler_parameters,os.path.join(PARAMETERS_OUTPUT,"normalisation.bin"))
        else:
            raise("The data needs to be a pandas dataframe")

@dataclass
class MapGender(OrdinalEncoding):
    """ To transform the gender feature in a pandas dataframe """

    data : pd.DataFrame =  None # to change this to a customisable typing

    def map_ordinal(self,header : str,save_scaler_parameter : bool = False):
        """
        [An implementation of OrdinalEncoding that transforms
        a pandas dataframe with relevant column headers]

        Args:
            header (list): [The features of the pandas dataframe to be transformed]
            save_scaler_parameter (bool, optional): [Send the parameters of the scaler to save_entities folder]. Defaults to False.
        """
        if isinstance(self.data, pd.DataFrame): # not sure about this yet
            transformer = LabelEncoder()
            transformer_parameters = transformer.fit(self.data[header])
            transformed_features = transformer_parameters.transform(self.data[header])
            self.data[header] = transformed_features
            if save_scaler_parameter is not False:
                dump(transformer_parameters,os.path.join(PARAMETERS_OUTPUT,"gender_transformer.bin"))
        else:
            raise("The data needs to be a pandas Dataframe")

@dataclass
class FeatureIdentification():
    """
    [A collection of the feature types. The plan is to use this to make the process of data features easier.
    I should be able to sub]
    """
    features : List[str] = None
    independent_feat : List[str] = None
    dependent_feat : str = None
    continuous_feat : List[str] = None
    nominal_feat : List[str] = None
    ordinal_feat : List[str] = None

    def clean_instance_variables(self):
        """
        [Goes through all attributes and cleans by replacing spaces, dashes, question mark, slashes, percentage and brackets with space
        and dollar symbol with space]
        """
        for key in self.__dict__:
            if self.__dict__[key] is not None:
                self.__dict__[key] = [_.lower().replace(' ','_').replace("-","_").replace("?","_").replace("/","_").replace("\\","_").replace("%","_") \
                                .replace("(","").replace(")","").replace("$","") for _ in self.__dict__[key]]

@dataclass
class DataCleaning():
    features : Optional[FeatureIdentification] = None
    scale_numerical_features : Optional[ScaleData] = None
    ordinal_encoding : Optional[OrdinalEncoding] = None
    dataframe = None

    # look into
    #def scale_numerical(self,dataframe):
    #    dataframe[self.features.nominal_feat] = self.scale_numerical_features.scale_numerical_data(dataframe[self.features.nominal_feat],dataframe)

    @staticmethod
    def clean_df_columns(dataframe):
        """
        [current iteration replaces spaces, dashes, question mark, slashes, percentage and brackets with space
        and dollar symbol with space]

        Args:
            dataframe ([pandas dataframe])
        """
        dataframe.columns = [column.lower().replace(' ','_').replace("-","_").replace("?","_").replace("/","_").replace("\\","_").replace("%","_") \
                                    .replace("(","").replace(")","").replace("$","") for column in dataframe.columns]