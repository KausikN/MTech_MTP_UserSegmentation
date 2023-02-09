"""
Encode Utils
"""

# Imports
import os
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, LocallyLinearEmbedding

# Main Functions
## Encode Functions
def EncodeUtils_Encode_Date(date, split_key="/"):
    '''
    EncodeUtils - Encode Date
    '''
    # Init
    date = date.split(split_key)
    # Encode
    # date = int(date[0])*(10**0) + int(date[1])*(10**2) + int(date[2])*(10**4) # Int
    date = [float(d.strip()) for d in date] # List

    return date

def EncodeUtils_EncodeArray_Date(a, split_key="/"):
    '''
    EncodeUtils - Encode Array of Date
    '''
    # Init
    a = np.array(a)
    # Encode
    ea = np.array([EncodeUtils_Encode_Date(d, split_key) for d in a], dtype=float)

    return ea, {}

def EncodeUtils_EncodeArray_StrBool(a, true_token="T"):
    '''
    EncodeUtils - Encode Array of String as Boolean Float (0.0 or 1.0)
    '''
    # Init
    a = np.array(a)
    # Encode
    ea = np.array(a == true_token, dtype=float)

    return ea, {}
    
def EncodeUtils_EncodeArray_Categorical(a, unique_categories=None):
    '''
    EncodeUtils - Encode Array of Categorical Data
    '''
    # Init
    a = np.array(a)
    # Encode
    ea = np.array([unique_categories == d for d in a], dtype=float)

    return ea, {}

## Norm Functions
def EncodeUtils_NormData_MinMax(data, min_val=0.0, max_val=1.0):
    '''
    EncodeUtils - Norm Data - MinMax
    '''
    # Init
    data = np.array(data)
    if min_val == max_val: return data
    # Norm
    return (data - min_val) / (max_val - min_val)

# Main Vars
DR_METHODS = {
    None: None,
    "PCA": PCA,
    "SVD": TruncatedSVD,
    "LDA": LinearDiscriminantAnalysis,
    "ISOMAP": Isomap,
    "LLE": LocallyLinearEmbedding
}