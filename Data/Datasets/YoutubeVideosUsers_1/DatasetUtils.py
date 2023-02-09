"""
Dataset Utils for Youtube Videos-Users Dataset

Link: https://www.kaggle.com/datasets/kenjee/ken-jee-youtube-data

Expected Files in Dataset Folder:
    - Aggregated_Metrics_By_Country_And_Subscriber_Status.csv   :  Videos-Users Dataset CSV File
    - Aggregated_Metrics_By_Video.csv                           :  Videos Dataset CSV File
    - All_Comments_Final.csv                                    :  Comments Dataset CSV File
    - Video_Performance_Over_Time.csv                           :  Video Performances Dataset CSV File
"""

# Imports
from enum import unique
import os
import functools
import numpy as np
import pandas as pd
# Import from Parent Path
from Utils.KaggleUtils import *
from Utils.EncodeUtils import *

# Main Vars
DATASET_PATH = "Data/Datasets/YoutubeVideosUsers_1/Data/"
DATASET_ITEMPATHS = {
    "kaggle": "kenjee/ken-jee-youtube-data",
    "test": "Aggregated_Metrics_By_Country_And_Subscriber_Status.csv"
}
DATASET_DATA = {
    "demographic": {
        "n_clusters": None, # None means no. of clusters is not fixed for this dataset (label not provided)
        "cols": {
            "all": {
                "demographic": [
                    "Video Title", "External Video ID", "Video Length", "Thumbnail link", 
                    "Country Code", "Is Subscribed", "Views", "Video Likes Added", "Video Dislikes Added", 
                    "Video Likes Removed", "User Subscriptions Added", "User Subscriptions Removed", 
                    "Average View Percentage", "Average Watch Time", "User Comments Added"
                ]
            },
            "keep": {
                "demographic": [
                    # "External Video ID",  
                    "Video Length", 
                    "Country Code", 
                    "Is Subscribed", "Views", 
                    "Video Likes Added", "Video Dislikes Added", "Video Likes Removed", "User Subscriptions Added", 
                    "User Subscriptions Removed", "Average View Percentage", "Average Watch Time", 
                    "User Comments Added"
                ]
            },
            "keep_default": {
                "demographic": [
                    # "Video Length", "Views"
                    "Video Length", 
                    "Country Code", 
                    "Is Subscribed", "Views", 
                    "Video Likes Added", "Video Dislikes Added", "Video Likes Removed", "User Subscriptions Added", 
                    "User Subscriptions Removed", "Average View Percentage", "Average Watch Time", 
                    "User Comments Added"
                ]
            },
            "target": None # None means no target column for this dataset
        }
    },
    "demographic-behavior": {
        "n_clusters": None, # None means no. of clusters is not fixed for this dataset (label not provided)
        "cols": {
            "all": {
                "demographic": [
                    "Video Title", "External Video ID", "Video Length", "Thumbnail link", 
                    "Country Code", "Is Subscribed", "Views", "Video Likes Added", "Video Dislikes Added", 
                    "Video Likes Removed", "User Subscriptions Added", "User Subscriptions Removed", 
                    "Average View Percentage", "Average Watch Time", "User Comments Added"
                ],
                "behavior": [],
                "product": []
            },
            "keep": {
                "demographic": [
                    "Country Code", 
                    "Is Subscribed", 
                    "Average View Percentage", "Average Watch Time"
                ],
                "behavior": [
                    "Views", "Video Likes Added", "Video Dislikes Added", 
                    "Video Likes Removed", "User Subscriptions Added", "User Subscriptions Removed", 
                    "User Comments Added"
                ],
                "product": [
                    "Video Title", "External Video ID", "Video Length", "Thumbnail link"
                ]
            },
            "keep_default": {
                "demographic": [
                    # "Country Code", 
                    "Is Subscribed",
                    "Average View Percentage", "Average Watch Time"
                ],
                "behavior": [
                    "Views"
                    # "Views", "Video Likes Added", "Video Dislikes Added", 
                    # "Video Likes Removed", "User Subscriptions Added", "User Subscriptions Removed", 
                    # "User Comments Added"
                ],
                "product": [
                    "Video Length"
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

    # Datasets
    dataset_d = None
    dataset_b = None
    dataset_p = None

    # Demographic Dataset
    if data_type == "demographic":
        ## Keep only necessary columns
        dataset_d = dataset[keep_cols["demographic"]].copy()

    # Demographic-Behavior Dataset
    elif data_type == "demographic-behavior":
        ## Keep only necessary columns
        d = dataset[keep_cols["demographic"]].copy()
        b = dataset[keep_cols["behavior"]].copy()
        p = dataset[keep_cols["product"]].copy()
        ## Get Unique Users
        d_unique = d.drop_duplicates()
        d_unique.reset_index(drop=True, inplace=True)
        ### Get Unique Users Indices in Dataset
        UserIndices = np.zeros(d.shape[0], dtype=int)
        for i in range(d.shape[0]):
            d_obj = d.iloc[i]
            for j in range(d_unique.shape[0]):
                if np.array_equal(d_obj, d_unique.iloc[j]):
                    UserIndices[i] = j
                    break
        ### Add User Indices Column
        # d_unique.loc[:, "User ID"] = np.arange(d_unique.shape[0])
        b.loc[:, "User ID"] = UserIndices
        ## Get Unique Videos
        p_unique = p.drop_duplicates()
        p_unique.reset_index(drop=True, inplace=True)
        ### Get Unique Videos Indices in Dataset
        VideoIndices = np.zeros(p.shape[0], dtype=int)
        for i in range(p.shape[0]):
            p_obj = p.iloc[i]
            for j in range(p_unique.shape[0]):
                if np.array_equal(p_obj, p_unique.iloc[j]):
                    VideoIndices[i] = j
                    break
        ### Add Video Indices Column
        # p_unique.loc[:, "Video ID"] = np.arange(p_unique.shape[0])
        b.loc[:, "Video ID"] = VideoIndices
        ## Finalise
        ### dataset_d is only the unique users data
        ### dataset_p is only the unique videos data
        ### dataset_b is user-video combination data with extra columns = ["User ID", "Video ID"]
        dataset_d = d_unique
        dataset_b = b
        dataset_p = p_unique

    # Return
    DATASET = {
        "data_type": data_type,
        "N": dataset_d.shape[0],
        "feature_names": {
            "demographic": list(dataset_d.columns) if dataset_d is not None else [],
            "behavior": [f"Video_{i}" for i in range(dataset_p.shape[0])] if dataset_b is not None else [],
            "product": list(dataset_p.columns) if dataset_p is not None else []
        },
        "target": dataset_target,
        "demographic": dataset_d,
        "behavior": dataset_b,
        "product": dataset_p,

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
            "norm_cols": [
                "Video Length", "Views", "Video Likes Added", "Video Dislikes Added", 
                "Video Likes Removed", "User Subscriptions Added", "User Subscriptions Removed", 
                "Average View Percentage", "Average Watch Time", "User Comments Added"
            ],
            "norm_data": {
                "Video Length": {}, "Views": {}, "Video Likes Added": {}, "Video Dislikes Added": {}, 
                "Video Likes Removed": {}, "User Subscriptions Added": {}, "User Subscriptions Removed": {}, 
                "Average View Percentage": {}, "Average Watch Time": {}, "User Comments Added": {}
            }
        }

    # Target Dataset
    ## Dummy Labels - 0th cluster assigned to all data points
    Ls = np.zeros((dataset["N"], n_clusters if n_clusters is not None else 2))
    Ls[:, 0] = 1.0

    # Datasets
    # Demographic Dataset
    dataset_d = dataset["demographic"]
    if dataset_d is not None:
        ## Init
        dataset_d = dataset["demographic"]
        d_features_info = [{
            "name": dataset["feature_names"]["demographic"][i],
            "type": {"type": "number"}
        } for i in range(dataset_d.shape[1])]
        boolean_indices = {
            "Is Subscribed": list(dataset_d.columns).index("Is Subscribed") if "Is Subscribed" in dataset_d.columns else None
        }
        category_indices = {
            "External Video ID": list(dataset_d.columns).index("External Video ID") if "External Video ID" in dataset_d.columns else None,
            "Country Code": list(dataset_d.columns).index("Country Code") if "Country Code" in dataset_d.columns else None,
        }
        ## Encode Dataset
        dataset_d = dataset_d.copy()
        ## Convert to numpy array
        dataset_d = dataset_d.to_numpy().astype(object)
        ## Boolean Encoding
        for bk in boolean_indices.keys():
            if boolean_indices[bk] is not None:
                # for i in range(dataset_d.shape[0]): dataset_d[i, boolean_indices[bk]] = float(dataset_d[i, boolean_indices[bk]])
                dataset_d[:, boolean_indices[bk]] = dataset_d[:, boolean_indices[bk]].astype(float)
                d_features_info[boolean_indices[bk]]["type"] = {"type": "boolean", "categories": [False, True]}
        ## Category Encoding
        for ck in category_indices.keys():
            if category_indices[ck] is not None:
                if INIT_SESSION:
                    unique_categories, _ = np.unique(dataset_d[:, category_indices[ck]], return_counts=True)
                    DATASET_SESSION_DATA["unique_categories"][ck] = unique_categories
                else:
                    unique_categories = DATASET_SESSION_DATA["unique_categories"][ck]
                ids_onehot = np.zeros((dataset_d.shape[0], unique_categories.shape[0]))
                for i in range(dataset_d.shape[0]):
                    ids_onehot[i, np.where(unique_categories == dataset_d[i, category_indices[ck]])[0][0]] = 1.0
                    dataset_d[i, category_indices[ck]] = list(ids_onehot[i])
                d_features_info[category_indices[ck]]["type"] = {"type": "category", "categories": list(unique_categories)}
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
        dataset_d = dataset_d
        Fs["demographic"] = dataset_d
        FEATURES_INFO["demographic"] = d_features_info

    # Behavior Dataset
    dataset_b = dataset["behavior"]
    if dataset_b is not None:
        ## Init
        dataset_p = dataset["product"]
        p_features_info = [{
            "name": dataset["feature_names"]["product"][i],
            "type": {"type": "obj"}
        } for i in range(dataset_p.shape[1])]
        category_indices_p = {
            "External Video ID": list(dataset_p.columns).index("External Video ID") if "External Video ID" in dataset_p.columns else None,
        }
        ## Encode Dataset
        dataset_b = dataset_b.copy()
        dataset_p = dataset_p.copy()
        ## Convert to numpy array
        dataset_b = dataset_b.to_numpy().astype(object)
        dataset_p = dataset_p.to_numpy().astype(object)
        ## Category Encoding
        for ck in category_indices_p.keys():
            if category_indices_p[ck] is not None:
                if INIT_SESSION:
                    unique_categories = np.unique(dataset_p[:, category_indices_p[ck]])
                    DATASET_SESSION_DATA["unique_categories"][ck] = unique_categories
                else:
                    unique_categories = DATASET_SESSION_DATA["unique_categories"][ck]
                ids_onehot = np.zeros((dataset_p.shape[0], unique_categories.shape[0]))
                for i in range(dataset_p.shape[0]):
                    ids_onehot[i, np.where(unique_categories == dataset_p[i, category_indices_p[ck]])[0][0]] = 1.0
                    dataset_p[i, category_indices_p[ck]] = list(ids_onehot[i])
                p_features_info[category_indices_p[ck]]["type"] = {"type": "category", "categories": list(unique_categories)}
        ## Norm
        colI_X = list(dataset["behavior"].columns).index("User ID")
        colI_Y = list(dataset["behavior"].columns).index("Video ID")
        if OTHER_PARAMS["norm"]:
            ### Behavior
            for i in range(dataset_b.shape[1]):
                if not (i == colI_X or i == colI_Y):
                    fname = dataset["behavior"].columns[i]
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
            ### Product
            for i in range(len(dataset["feature_names"]["product"])):
                fname = dataset["feature_names"]["product"][i]
                if fname in DATASET_SESSION_DATA["norm_cols"]:
                    if INIT_SESSION:
                        DATASET_SESSION_DATA["norm_data"].update({
                            fname: {
                                "min": dataset_p[:, i].min(),
                                "max": dataset_p[:, i].max()
                            }
                        })
                    dataset_p[:, i] = EncodeUtils_NormData_MinMax(
                        dataset_p[:, i], 
                        DATASET_SESSION_DATA["norm_data"][fname]["min"], 
                        DATASET_SESSION_DATA["norm_data"][fname]["max"]
                    )
        ## Form Behavior Dataset
        b_cols_indices = [
            ci for ci in range(len(dataset["behavior"].columns)) 
                if ci not in [colI_X, colI_Y]
        ]
        b_cols = [dataset["behavior"].columns[i] for i in b_cols_indices]
        b = [[[0.0]*len(b_cols) for j in range(dataset_p.shape[0])] for i in range(dataset_d.shape[0])]
        for i in range(dataset_b.shape[0]):
            b_obj = dataset_b[i]
            b[b_obj[colI_X]][b_obj[colI_Y]] = list(b_obj[b_cols_indices])
        b = np.array(b)
        b_features_info = [{
            "name": dataset["feature_names"]["behavior"][i],
            "type": {"type": "category", "categories": b_cols}
        } for i in range(len(dataset["feature_names"]["behavior"]))]

        ## Finalize
        dataset_b = b
        dataset_p = dataset_p
        Fs["behavior"] = dataset_b
        Fs["product"] = dataset_p
        FEATURES_INFO["behavior"] = b_features_info
        FEATURES_INFO["product"] = p_features_info

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
    for k in ["demographic", "behavior", "product"]:
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