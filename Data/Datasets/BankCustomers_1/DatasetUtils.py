"""
Dataset Utils for Bank Customers 1 Dataset

Link: https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation

Expected Files in Dataset Folder:
    - Mall_Customers.csv    :  Dataset CSV File
"""

# Imports
import os
import functools
import numpy as np
import pandas as pd
# Import from Parent Path
from Utils.KaggleUtils import *
from Utils.EncodeUtils import *

# Main Vars
DATASET_PATH = "Data/Datasets/BankCustomers_1/Data/"
DATASET_ITEMPATHS = {
    "kaggle": "shivamb/bank-customer-segmentation",
    "test": "bank_transactions.csv"
}
DATASET_DATA = {
    "demographic": {
        "n_clusters": None, # None means no. of clusters is not fixed for this dataset (label not provided)
        "cols": {
            "all": {
                "demographic": [
                    "TransactionID", "CustomerID", "CustomerDOB", "CustGender", "CustLocation", 
                    "CustAccountBalance", "TransactionDate", "TransactionTime", "TransactionAmount (INR)"
                ]
            },
            "keep": {
                "demographic": [
                    "CustomerDOB", "CustGender", 
                    # "CustLocation",
                    "CustAccountBalance", "TransactionDate", "TransactionTime", "TransactionAmount (INR)"
                ]
            },
            "keep_default": {
                "demographic": [
                    # "CustAccountBalance", "TransactionDate"
                    "CustomerDOB", "CustGender", 
                    # "CustLocation",
                    "CustAccountBalance", "TransactionDate", "TransactionTime", "TransactionAmount (INR)"
                ]
            },
            "target": None # None means no target column for this dataset
        }
    }
}
DATASET_PARAMS = {
    "load": {
        "N_subset": 0.01
    },
    "encode": {
        "norm": False
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
    csv_path = os.path.join(path, DATASET_ITEMPATHS[mode])
    # Download Dataset
    if not os.path.exists(csv_path):
        KaggleUtils_DownloadDataset(DATASET_ITEMPATHS["kaggle"], DATASET_PATH, quiet=False, unzip=True)
    # Get Dataset
    dataset = DatasetUtils_LoadCSV(csv_path)
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
            "unique_categories": {},
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
        boolean_indices = {
            "Gender": list(dataset_d.columns).index("CustGender") if "CustGender" in dataset_d.columns else None
        }
        date_indices = {
            "CustomerDOB": list(dataset_d.columns).index("CustomerDOB") if "CustomerDOB" in dataset_d.columns else None,
            "TransactionDate": list(dataset_d.columns).index("TransactionDate") if "TransactionDate" in dataset_d.columns else None
        }
        category_indices = {
            "CustLocation": list(dataset_d.columns).index("CustLocation") if "CustLocation" in dataset_d.columns else None
        }
        ## Encode Dataset
        dataset_d = dataset_d.copy()
        ## Convert to numpy array
        dataset_d = dataset_d.to_numpy().astype(object)
        dataset_d_features = np.empty((dataset_d.shape[0], 0), dtype=object)
        ## Encode
        for i in range(dataset_d.shape[1]):
            ## Boolean Encoding
            boolFound = False
            for bk in boolean_indices.keys():
                if boolean_indices[bk] is not None and i == boolean_indices[bk]:
                    # vals = np.array(dataset_d[:, boolean_indices[bk]] == "M", dtype=float)
                    vals, _ = EncodeUtils_EncodeArray_StrBool(dataset_d[:, boolean_indices[bk]], true_token="M")
                    dataset_d_features = np.concatenate((dataset_d_features, vals.reshape(-1, 1)), axis=1)
                    d_features_info.append({
                        "name": dataset["feature_names"]["demographic"][i],
                        "type": {"type": "boolean", "categories": ["F", "M"]}
                    })
                    boolFound = True
                    break
            if boolFound: continue
            ## Date Encoding
            dateFound = False
            for d in date_indices.keys():
                if date_indices[d] is not None and i == date_indices[d]:
                    ### Change 1/1/1800 to 1/1/1
                    dataset_d[dataset_d[:, date_indices[d]] == "1/1/1800", date_indices[d]] = "1/1/1"
                    ### Encode Date
                    # vals = np.array([EncodeUtils_Encode_Date(dataset_d[j, date_indices[d]]) for j in range(dataset_d.shape[0])], dtype=float)
                    vals, _ = EncodeUtils_EncodeArray_Date(dataset_d[:, date_indices[d]], split_key="/")
                    d_features_info.extend([{
                        "name": d+"_"+x,
                        "type": {"type": "number"}
                    } for x in ["D", "M", "Y"]])
                    ### Norm
                    if OTHER_PARAMS["norm"]:
                        if INIT_SESSION:
                            DATASET_SESSION_DATA["norm_data"].update({
                                # Day
                                d_features_info[-3]["name"]: {
                                    "min": 1.0,
                                    "max": 31.0
                                },
                                # Month
                                d_features_info[-2]["name"]: {
                                    "min": 1.0,
                                    "max": 12.0
                                },
                                # Year
                                d_features_info[-1]["name"]: {
                                    "min": vals[:, 2].min(),
                                    "max": vals[:, 2].max()
                                }
                            })
                        for j in range(3):
                            vals[:, j] = EncodeUtils_NormData_MinMax(
                                vals[:, j], 
                                DATASET_SESSION_DATA["norm_data"][d_features_info[-3+j]["name"]]["min"], 
                                DATASET_SESSION_DATA["norm_data"][d_features_info[-3+j]["name"]]["max"]
                            )
                    ### Concatenate
                    dataset_d_features = np.concatenate((dataset_d_features, vals), axis=1)
                    dateFound = True
                    break
            if dateFound: continue
            ## Category Encoding
            categoryFound = False
            for ck in category_indices.keys():
                if category_indices[ck] is not None and i == category_indices[ck]:
                    if INIT_SESSION:
                        unique_categories, _ = np.unique(dataset_d[:, category_indices[ck]], return_counts=True)
                        DATASET_SESSION_DATA["unique_categories"][ck] = unique_categories
                    else:
                        unique_categories = DATASET_SESSION_DATA["unique_categories"][ck]
                    # vals = np.array([unique_categories == dataset_d[j, category_indices[ck]] for j in range(dataset_d.shape[0])], dtype=float)
                    vals, _ = EncodeUtils_EncodeArray_Categorical(dataset_d[:, category_indices[ck]], unique_categories=unique_categories)
                    dataset_d_features = np.concatenate((dataset_d_features, np.empty((vals.shape[0], 1))), axis=1)
                    for j in range(vals.shape[0]): dataset_d_features[j, -1] = vals[j]
                    d_features_info.append({
                        "name": dataset["feature_names"]["demographic"][i],
                        "type": {"type": "category", "categories": list(unique_categories)}
                    })
                    categoryFound = True
                    break
            if categoryFound: continue
            ## Number Encoding
            vals = np.array(dataset_d[:, i].reshape(-1, 1), dtype=float)
            d_features_info.append({
                "name": dataset["feature_names"]["demographic"][i],
                "type": {"type": "number"}
            })
            ### Norm
            if OTHER_PARAMS["norm"]:
                if INIT_SESSION:
                    DATASET_SESSION_DATA["norm_data"].update({
                        d_features_info[-1]["name"]: {
                            "min": vals[:, 0].min(),
                            "max": vals[:, 0].max()
                        }
                    })
                vals = EncodeUtils_NormData_MinMax(
                    vals, 
                    DATASET_SESSION_DATA["norm_data"][d_features_info[-1]["name"]]["min"], 
                    DATASET_SESSION_DATA["norm_data"][d_features_info[-1]["name"]]["max"]
                )
            ### Concatenate
            dataset_d_features = np.concatenate((dataset_d_features, vals), axis=1)
        ## Finalize
        dataset_d = dataset_d_features
        Fs["demographic"] = dataset_d
        FEATURES_INFO["demographic"] = d_features_info

    # Behavior Dataset
    dataset_b = dataset["behavior"]
    b_features_info = []
    if dataset_b is not None:
        ## Init
        b_features_info = [{
            "name": dataset["feature_names"]["behavior"][i],
            "type": {"type": "number"}
        } for i in range(dataset_b.shape[1])]
        ## Encode Dataset
        dataset_b = dataset_b.copy()
        ## Convert to numpy array
        dataset_b = dataset_b.to_numpy()
        ## Finalize
        dataset_b = dataset_b.astype(float)
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