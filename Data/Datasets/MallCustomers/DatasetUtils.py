"""
Dataset Utils for Mall Customers Dataset

Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Expected Files in Dataset Folder:
    - Mall_Customers.csv    :  Dataset CSV File
"""

# Imports
import os
import functools
import numpy as np
import pandas as pd
# Import from Parent Path
from Utils.EncodeUtils import *

# Main Vars
DATASET_PATH = "Data/Datasets/MallCustomers/Data/"
DATASET_ITEMPATHS = {
    "test": "Mall_Customers.csv"
}
DATASET_DATA = {
    "demographic": {
        "n_clusters": 100, # None means no. of clusters is not fixed for this dataset (label not provided)
        "cols": {
            "all": {
                "demographic": [
                    "CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"
                ]
            },
            "keep": {
                "demographic": [
                    "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"
                ]
            },
            "keep_default": {
                "demographic": [
                    # "Annual Income (k$)", "Spending Score (1-100)"
                    "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"
                ]
            },
            "target": "Spending Score (1-100)" # None means no target column for this dataset
        }
    },
    "demographic-behavior": {
        "n_clusters": 100, # None means no. of clusters is not fixed for this dataset (label not provided)
        "cols": {
            "all": {
                "demographic": [
                    "CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"
                ],
                "behavior": [
                    "Spending Score (1-100)"
                ]
            },
            "keep": {
                "demographic": [
                    "Gender", "Age", "Annual Income (k$)"
                ],
                "behavior": [
                    "Spending Score (1-100)"
                ]
            },
            "keep_default": {
                "demographic": [
                    "Gender", "Age"
                ],
                "behavior": [
                    "Spending Score (1-100)"
                ]
            },
            "target": "Spending Score (1-100)" # None means no target column for this dataset
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
            "norm_cols": ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
            "norm_data": {}
        }

    # Target Dataset
    ## Encode Dataset
    dataset_target = dataset["target"].copy()
    ## Convert to numpy array
    dataset_target = dataset_target.to_numpy()
    ## Get Indices
    Ls = np.zeros((dataset_target.shape[0], n_clusters))
    Ls_indices = np.array(dataset_target, dtype=int) # -1 done to convert value to index ([1, 100] -> [0, 99])
    ### If true label count > n_clusters - Join extra labels (n_clusters - 1 true labels + 1 joint label for rest)
    if Ls_indices.max() > (n_clusters-1): Ls_indices[Ls_indices > n_clusters-1] = n_clusters-1
    ### One-hot encode labels
    Ls[np.arange(dataset_target.shape[0]), Ls_indices] = 1.0

    # Demographic Dataset
    dataset_d = dataset["demographic"]
    if dataset_d is not None:
        ## Init
        d_features_info = [{
            "name": dataset["feature_names"]["demographic"][i],
            "type": {"type": "number"}
        } for i in range(dataset_d.shape[1])]
        boolean_indices = {
            "Gender": list(dataset_d.columns).index("Gender") if "Gender" in dataset_d.columns else None
        }
        ## Encode Dataset
        dataset_d = dataset_d.copy()
        ## Convert to numpy array
        dataset_d = dataset_d.to_numpy().astype(object)
        ## Category Encoding
        for bk in boolean_indices.keys():
            if boolean_indices[bk] is not None:
                # dataset_d[:, boolean_indices[bk]] = dataset_d[:, boolean_indices[bk]] == "Male"
                dataset_d[:, boolean_indices[bk]], _ = EncodeUtils_EncodeArray_StrBool(dataset_d[:, boolean_indices[bk]], true_token="Male")
                d_features_info[boolean_indices[bk]]["type"] = {"type": "boolean", "categories": ["F", "M"]}
        ## Norm
        if OTHER_PARAMS["norm"]:
            for i in range(len(dataset["feature_names"]["demographic"])):
                fname = dataset["feature_names"]["demographic"][i]
                if fname in DATASET_SESSION_DATA["norm_cols"]:
                    if INIT_SESSION:
                        DATASET_SESSION_DATA["norm_data"].update({
                            fname: {
                                "min": dataset_d[:, i].min(),
                                "max": dataset_d[:, i].max()
                            }
                        })
                    dataset_d[:, i] = EncodeUtils_NormData_MinMax(
                        dataset_d[:, i], 
                        DATASET_SESSION_DATA["norm_data"][fname]["min"], 
                        DATASET_SESSION_DATA["norm_data"][fname]["max"]
                    )
        ## Finalize
        dataset_d = dataset_d.astype(float)
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
        ## Norm
        if OTHER_PARAMS["norm"]:
            for i in range(len(dataset["feature_names"]["behavior"])):
                fname = dataset["feature_names"]["behavior"][i]
                if fname in DATASET_SESSION_DATA["norm_cols"]:
                    if INIT_SESSION:
                        DATASET_SESSION_DATA["norm_data"].update({
                            fname: {
                                "min": dataset_b[:, i].min(),
                                "max": dataset_b[:, i].max()
                            }
                        })
                    dataset_b[:, i] = EncodeUtils_NormData_MinMax(
                        dataset_b[:, i], 
                        DATASET_SESSION_DATA["norm_data"][fname]["min"], 
                        DATASET_SESSION_DATA["norm_data"][fname]["max"]
                    )
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