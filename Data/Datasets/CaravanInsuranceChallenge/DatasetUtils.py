"""
Dataset Utils for Caravan Insurance Challenge Dataset

Link: https://www.kaggle.com/datasets/uciml/caravan-insurance-challenge

Expected Files in Dataset Folder:
    - caravan-insurance-challenge.csv    :  Dataset CSV File
"""

# Imports
import os
import functools
import numpy as np
import pandas as pd
# Import from Parent Path
from Utils.EncodeUtils import *

# Main Vars
DATASET_PATH = "Data/Datasets/CaravanInsuranceChallenge/Data/"
DATASET_ITEMPATHS = {
    "test": "caravan-insurance-challenge.csv"
}
DATASET_DATA = {
    "demographic": {
        "n_clusters": 2, # None means no. of clusters is not fixed for this dataset (label not provided)
        "cols": {
            "all": {
                "demographic": [
                    "ORIGIN", "MOSTYPE", "MAANTHUI", "MGEMOMV", "MGEMLEEF", "MOSHOOFD", "MGODRK", "MGODPR", "MGODOV", 
                    "MGODGE", "MRELGE", "MRELSA", "MRELOV", "MFALLEEN", "MFGEKIND", "MFWEKIND", "MOPLHOOG", "MOPLMIDD", 
                    "MOPLLAAG", "MBERHOOG", "MBERZELF", "MBERBOER", "MBERMIDD", "MBERARBG", "MBERARBO", "MSKA", "MSKB1", 
                    "MSKB2", "MSKC", "MSKD", "MHHUUR", "MHKOOP", "MAUT1", "MAUT2", "MAUT0", "MZFONDS", "MZPART", "MINKM30", 
                    "MINK3045", "MINK4575", "MINK7512", "MINK123M", "MINKGEM", "MKOOPKLA", "PWAPART", "PWABEDR", "PWALAND", 
                    "PPERSAUT", "PBESAUT", "PMOTSCO", "PVRAAUT", "PAANHANG", "PTRACTOR", "PWERKT", "PBROM", "PLEVEN", "PPERSONG", 
                    "PGEZONG", "PWAOREG", "PBRAND", "PZEILPL", "PPLEZIER", "PFIETS", "PINBOED", "PBYSTAND", "AWAPART", "AWABEDR", 
                    "AWALAND", "APERSAUT", "ABESAUT", "AMOTSCO", "AVRAAUT", "AAANHANG", "ATRACTOR", "AWERKT", "ABROM", "ALEVEN", 
                    "APERSONG", "AGEZONG", "AWAOREG", "ABRAND", "AZEILPL", "APLEZIER", "AFIETS", "AINBOED", "ABYSTAND"
                    , "CARAVAN"
                ]
            },
            "keep": {
                "demographic": [
                    "MOSTYPE", "MAANTHUI", "MGEMOMV", "MGEMLEEF", "MOSHOOFD", "MGODRK", "MGODPR", "MGODOV", 
                    "MGODGE", "MRELGE", "MRELSA", "MRELOV", "MFALLEEN", "MFGEKIND", "MFWEKIND", "MOPLHOOG", "MOPLMIDD", 
                    "MOPLLAAG", "MBERHOOG", "MBERZELF", "MBERBOER", "MBERMIDD", "MBERARBG", "MBERARBO", "MSKA", "MSKB1", 
                    "MSKB2", "MSKC", "MSKD", "MHHUUR", "MHKOOP", "MAUT1", "MAUT2", "MAUT0", "MZFONDS", "MZPART", "MINKM30", 
                    "MINK3045", "MINK4575", "MINK7512", "MINK123M", "MINKGEM", "MKOOPKLA", "PWAPART", "PWABEDR", "PWALAND", 
                    "PPERSAUT", "PBESAUT", "PMOTSCO", "PVRAAUT", "PAANHANG", "PTRACTOR", "PWERKT", "PBROM", "PLEVEN", "PPERSONG", 
                    "PGEZONG", "PWAOREG", "PBRAND", "PZEILPL", "PPLEZIER", "PFIETS", "PINBOED", "PBYSTAND", "AWAPART", "AWABEDR", 
                    "AWALAND", "APERSAUT", "ABESAUT", "AMOTSCO", "AVRAAUT", "AAANHANG", "ATRACTOR", "AWERKT", "ABROM", "ALEVEN", 
                    "APERSONG", "AGEZONG", "AWAOREG", "ABRAND", "AZEILPL", "APLEZIER", "AFIETS", "AINBOED", "ABYSTAND"
                    # , "CARAVAN"
                ]
            },
            "keep_default": {
                "demographic": [
                    # "MOSTYPE", "MAANTHUI"
                    "MOSTYPE", "MAANTHUI", "MGEMOMV", "MGEMLEEF", "MOSHOOFD", "MGODRK", "MGODPR", "MGODOV", 
                    "MGODGE", "MRELGE", "MRELSA", "MRELOV", "MFALLEEN", "MFGEKIND", "MFWEKIND", "MOPLHOOG", "MOPLMIDD", 
                    "MOPLLAAG", "MBERHOOG", "MBERZELF", "MBERBOER", "MBERMIDD", "MBERARBG", "MBERARBO", "MSKA", "MSKB1", 
                    "MSKB2", "MSKC", "MSKD", "MHHUUR", "MHKOOP", "MAUT1", "MAUT2", "MAUT0", "MZFONDS", "MZPART", "MINKM30", 
                    "MINK3045", "MINK4575", "MINK7512", "MINK123M", "MINKGEM", "MKOOPKLA", "PWAPART", "PWABEDR", "PWALAND", 
                    "PPERSAUT", "PBESAUT", "PMOTSCO", "PVRAAUT", "PAANHANG", "PTRACTOR", "PWERKT", "PBROM", "PLEVEN", "PPERSONG", 
                    "PGEZONG", "PWAOREG", "PBRAND", "PZEILPL", "PPLEZIER", "PFIETS", "PINBOED", "PBYSTAND", "AWAPART", "AWABEDR", 
                    "AWALAND", "APERSAUT", "ABESAUT", "AMOTSCO", "AVRAAUT", "AAANHANG", "ATRACTOR", "AWERKT", "ABROM", "ALEVEN", 
                    "APERSONG", "AGEZONG", "AWAOREG", "ABRAND", "AZEILPL", "APLEZIER", "AFIETS", "AINBOED", "ABYSTAND"
                    # , "CARAVAN"
                ]
            },
            "target": "CARAVAN" # None means no target column for this dataset
        }
    }
}
DATASET_PARAMS = {
    "load": {
        "N_subset": 1.0
    },
    "encode": {
        "norm": False,
        "dr": {
            "method": None,
            "n_components": 2
        }
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
    if "demographic" in DatasetData["cols"]["keep"]:
        ## Keep only necessary columns
        dataset_d = dataset[keep_cols["demographic"]].copy()

    # Behavior Dataset
    dataset_b = None
    if "behavior" in DatasetData["cols"]["keep"]:
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
            "norm_data": {},
            "dimensionality_reducer": {
                "n_components": OTHER_PARAMS["dr"]["n_components"],
                # PCA, SVD, LDA, ISOMAP, LLE
                "method": DR_METHODS[OTHER_PARAMS["dr"]["method"]], 
                "obj": {
                    "demographic": None,
                    "behavior": None
                }
            }
        }
    DR_METHOD = DATASET_SESSION_DATA["dimensionality_reducer"]["method"]
    N_COMPONENTS = DATASET_SESSION_DATA["dimensionality_reducer"]["n_components"]

    # Target Dataset
    ## Encode Dataset
    dataset_target = dataset["target"].copy()
    ## Convert to numpy array
    dataset_target = dataset_target.to_numpy()
    ## Get Indices
    Ls = np.zeros((dataset_target.shape[0], n_clusters))
    Ls_indices = np.array(dataset_target, dtype=int)
    Ls_unique = np.unique(Ls_indices)
    ## If true label count > n_clusters - Join extra labels (n_clusters - 1 true labels + 1 joint label for rest)
    if Ls_unique.shape[0] > n_clusters: Ls_indices[Ls_indices >= Ls_unique[n_clusters-1]] = n_clusters-1
    ## One-hot encode labels
    Ls[np.arange(dataset_target.shape[0]), Ls_indices] = 1.0

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
        ## Dimensionality Reduction
        if DR_METHOD is not None:
            if INIT_SESSION:
                dr = DR_METHOD(n_components=N_COMPONENTS)
                dataset_d = dr.fit_transform(dataset_d, dataset["target"].to_numpy())
                dataset["feature_names"]["demographic"] = [f"PCA_{i}" for i in range(N_COMPONENTS)]
                DATASET_SESSION_DATA["dimensionality_reducer"]["obj"]["demographic"] = dr
            else:
                dr = DATASET_SESSION_DATA["dimensionality_reducer"]["obj"]["demographic"]
                dataset_d = dr.transform(dataset_d)
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
        ## Dimensionality Reduction
        if DR_METHOD is not None:
            if INIT_SESSION:
                dr = DR_METHOD(n_components=N_COMPONENTS)
                dataset_b = dr.fit_transform(dataset_b, dataset["target"].to_numpy())
                dataset["feature_names"]["behavior"] = [f"PCA_{i}" for i in range(N_COMPONENTS)]
                DATASET_SESSION_DATA["dimensionality_reducer"]["obj"]["behavior"] = dr
            else:
                dr = DATASET_SESSION_DATA["dimensionality_reducer"]["obj"]["behavior"]
                dataset_b = dr.transform(dataset_b)
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