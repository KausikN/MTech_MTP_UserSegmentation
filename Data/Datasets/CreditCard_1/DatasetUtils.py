"""
Dataset Utils for Credit Card 1 Dataset

Link: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

Expected Files in Dataset Folder:
    - CC_GENERAL.csv    :  Dataset CSV File
"""

# Imports
import os
import functools
import numpy as np
import pandas as pd
# Import from Parent Path
from Utils.EncodeUtils import *

# Main Vars
DATASET_PATH = "Data/Datasets/CreditCard_1/Data/"
DATASET_ITEMPATHS = {
    "test": "CC_GENERAL.csv"
}
DATASET_DATA = {
    "demographic": {
        "n_clusters": None, # None means no. of clusters is not fixed for this dataset (label not provided)
        "cols": {
            "all": {
                "demographic": [
                    "CUST_ID", "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", 
                    "CASH_ADVANCE", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY", 
                    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", 
                    "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"
                ]
            },
            "keep": {
                "demographic": [
                    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", 
                    "CASH_ADVANCE", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY", 
                    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", 
                    "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"
                ]
            },
            "keep_default": {
                "demographic": [
                    # "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY"
                    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", 
                    "CASH_ADVANCE", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY", 
                    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", 
                    "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"
                ]
            },
            "target": None # None means no target column for this dataset
        }
    }
}
DATASET_PARAMS = {
    "load": {
        "N_subset": 1.0
    },
    "encode": {
        "norm": True
    }
}
DATASET_SESSION_DATA = {}

# Main Functions
# Load Functions
def DatasetUtils_LoadCSV(path):
    '''
    DatasetUtils - Load CSV
    '''
    return pd.read_csv(path)

# Dataset Functions
def DatasetUtils_LoadDataset(
    path=DATASET_PATH, mode="test", 
    N=-1, 
    DATASET_ITEMPATHS=DATASET_ITEMPATHS, 

    keep_cols=None, 
    data_type="demographic",
    other_params=DATASET_PARAMS,

    **params
    ):
    '''
    DatasetUtils - Load Dataset
    '''
    # Init
    OTHER_PARAMS = other_params["load"]
    DatasetData = DATASET_DATA[data_type]
    if keep_cols is None: keep_cols = DatasetData["cols"]["keep"]
    # Get Dataset
    dataset = DatasetUtils_LoadCSV(os.path.join(path, DATASET_ITEMPATHS[mode]))
    # Take N range
    if OTHER_PARAMS["N_subset"] < 1.0: dataset = dataset.iloc[::int(1.0/OTHER_PARAMS["N_subset"])]
    if type(N) == int:
        if N > 0: dataset = dataset.head(N)
    elif type(N) == list:
        if len(N) == 2: dataset = dataset.iloc[N[0]:N[1]]
    # Reset Columns
    dataset.columns = DatasetData["cols"]["all"]["demographic"]
    # Remove NaN values
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    # Target Dataset
    dataset_target = None
    if DatasetData["cols"]["target"] is not None:
        dataset_target = dataset[DatasetData["cols"]["target"]].copy()

    # Demographic Dataset
    dataset_d = None
    if "demographic" in DatasetData["cols"]["keep"].keys():
        ## Keep only necessary columns
        dataset_d = dataset[keep_cols["demographic"]].copy()

    # Behavior Dataset
    dataset_b = None
    if "behavior" in DatasetData["cols"]["keep"].keys():
        ## Keep only necessary columns
        dataset_b = dataset[keep_cols["behavior"]].copy()

    # Return
    DATASET = {
        "data_type": data_type,
        "N": dataset.shape[0],
        "feature_names": {
            "demographic": list(dataset_d.columns) if dataset_d is not None else [],
            "behavior": list(dataset_b.columns) if dataset_b is not None else []
        },
        "target": dataset_target,
        "demographic": dataset_d,
        "behavior": dataset_b,

        "session_params": {
            "init_session": ((type(N) == int) and (N == -1)),
            "other_params": other_params
        }
    }
    return DATASET

# Encode Functions
def DatasetUtils_EncodeDataset(
    dataset, 
    n_clusters=DATASET_DATA["demographic"]["n_clusters"],
    **params
    ):
    '''
    DatasetUtils - Encode Dataset
    '''
    global DATASET_SESSION_DATA
    # Init
    ## Data Init
    Fs = {}
    Ls = None
    FEATURES_INFO = {}
    ## Params Init
    OTHER_PARAMS = dataset["session_params"]["other_params"]["encode"]
    INIT_SESSION = dataset["session_params"]["init_session"]
    if INIT_SESSION:
        DATASET_SESSION_DATA = {
            "norm_data": {}
        }

    # Target Dataset
    ## Dummy Labels - 0th cluster assigned to all data points
    Ls = np.zeros((dataset["N"], n_clusters if n_clusters is not None else 2))
    Ls[:, 0] = 1.0

    # Demographic Dataset
    dataset_d = dataset["demographic"]
    if dataset_d is not None:
        ## Init
        d_features_info = []
        ## Encode Dataset
        dataset_d = dataset_d.copy()
        ## Convert to numpy array
        dataset_d = dataset_d.to_numpy().astype(object)
        ## Norm
        if OTHER_PARAMS["norm"]:
            for i in range(len(dataset["feature_names"]["demographic"])):
                if INIT_SESSION:
                    DATASET_SESSION_DATA["norm_data"].update({
                        dataset["feature_names"]["demographic"][i]: {
                            "min": dataset_d[:, i].min(),
                            "max": dataset_d[:, i].max()
                        }
                    })
                dataset_d[:, i] = EncodeUtils_NormData_MinMax(
                    dataset_d[:, i], 
                    DATASET_SESSION_DATA["norm_data"][dataset["feature_names"]["demographic"][i]]["min"], 
                    DATASET_SESSION_DATA["norm_data"][dataset["feature_names"]["demographic"][i]]["max"]
                )
        ## Finalize
        dataset_d = dataset_d.astype(float)
        d_features_info = [{
            "name": dataset["feature_names"]["demographic"][i],
            "type": {"type": "number"}
        } for i in range(dataset_d.shape[1])]
        Fs["demographic"] = dataset_d
        FEATURES_INFO["demographic"] = d_features_info

    # Behavior Dataset
    dataset_b = dataset["behavior"]
    if dataset_b is not None:
        ## Init
        b_features_info = []
        ## Encode Dataset
        dataset_b = dataset_b.copy()
        ## Convert to numpy array
        dataset_b = dataset_b.to_numpy().astype(object)
        ## Finalize
        dataset_b = dataset_b.astype(float)
        b_features_info = [{
            "name": dataset["feature_names"]["behavior"][i],
            "type": {"type": "number"}
        } for i in range(dataset_b.shape[1])]
        Fs["behavior"] = dataset_b
        FEATURES_INFO["behavior"] = b_features_info

    # Return
    return Fs, Ls, FEATURES_INFO

# Display Functions
def DatasetUtils_DisplayDataset(
    dataset, 
    N=-1,
    **params
    ):
    '''
    DatasetUtils - Display Dataset
    '''
    # Init
    pass
    # Generate Display Data
    display_data = pd.DataFrame()
    for k in ["demographic", "behavior"]:
        if dataset[k] is not None:
            d = dataset[k]
            ## Take N range
            if type(N) == int:
                if N > 0: d = d.head(N)
            elif type(N) == list:
                if len(N) == 2: d = d.iloc[N[0]:N[1]]
            ## Concat
            display_data = pd.concat([display_data, d], axis=1)

    return display_data

# Main Vars
DATASET_FUNCS = {
    "full": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),
    "train": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),
    "val": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),
    "test": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),

    "encode": functools.partial(DatasetUtils_EncodeDataset),
    "display": functools.partial(DatasetUtils_DisplayDataset)
}