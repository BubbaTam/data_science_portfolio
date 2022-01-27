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












