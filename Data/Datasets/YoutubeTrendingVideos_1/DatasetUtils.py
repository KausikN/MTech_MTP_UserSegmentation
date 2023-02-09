"""
Dataset Utils for Youtube Videos-Users Dataset

Link: https://www.kaggle.com/datasets/datasnaek/youtube-new

Expected Files in Dataset Folder:
    - {LANG}_category_id.json       :  Videos Categories JSON File
    - {LANG}videos.csv              :  Videos Data CSV File
    - Possible Languages: CA, DE, FR, GB, IN, JP, KR, MX, RU, US
"""

# Imports
import os
import json
import functools
import numpy as np
import pandas as pd
# Import from Parent Path
from Utils.KaggleUtils import *
from Utils.EncodeUtils import *

# Main Vars
DATASET_PATH = "Data/Datasets/YoutubeTrendingVideos_1/Data/"
DATASET_ITEMPATHS = {
    "kaggle": "datasnaek/youtube-new",
    "languages": ["CA", "DE", "FR", "GB", "IN", "JP", "KR", "MX", "RU", "US"],
    "test": {
        "category": "{}_category_id.json",
        "videos": "{}videos.csv"
    }
}
DATASET_DATA = {
    "demographic": {
        "n_clusters": None, # None means no. of clusters is not fixed for this dataset (label not provided)
        "cols": {
            "all": {
                "demographic": [
                    "language", "video_id", "trending_date", "title", "channel_title", "category_id", 
                    "publish_time", "tags", "views", "likes", "dislikes", "comment_count", "thumbnail_link", 
                    "comments_disabled", "ratings_disabled", "video_error_or_removed", "description"
                ]
            },
            "keep": {
                "demographic": [
                    "language", "video_id", "trending_date", "category_id", 
                    "views", "likes", "dislikes", "comment_count", 
                    "comments_disabled", "ratings_disabled", "video_error_or_removed"
                ]
            },
            "keep_default": {
                "demographic": [
                    "language", "trending_date", "category_id", "views", "likes", "dislikes", "comment_count", 
                    "comments_disabled", "ratings_disabled", "video_error_or_removed"
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
                    "language", "video_id", "trending_date", "title", "channel_title", "category_id", 
                    "publish_time", "tags", "views", "likes", "dislikes", "comment_count", "thumbnail_link", 
                    "comments_disabled", "ratings_disabled", "video_error_or_removed", "description"
                ],
                "behavior": [],
                "product": []
            },
            "keep": {
                "demographic": [
                    "language", "trending_date", "category_id",
                    
                ],
                "behavior": [
                    "views", "likes", "dislikes", "comment_count", 
                    "comments_disabled", "ratings_disabled", "video_error_or_removed"
                ],
                "product": [
                    "video_id", "title", "publish_time", "channel_title", "thumbnail_link", 
                    "tags", "description"
                ]
            },
            "keep_default": {
                "demographic": [
                    "language", "category_id"
                ],
                "behavior": [
                    "views"
                ],
                "product": [
                    "video_id"
                ]
            },
            "target": None # None means no target column for this dataset
        }
    }
}
DATASET_PARAMS = {
    "load": {
        "N_subset": 0.002
    },
    "encode": {
        "norm": False
    }
}
DATASET_SESSION_DATA = {}

# Main Functions
# Load Functions
def DatasetUtils_LoadCSV(path, **params):
    '''
    DatasetUtils - Load CSV
    '''
    return pd.read_csv(path, **params)

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
    # Download Dataset
    if not os.path.exists(os.path.join(path, DATASET_ITEMPATHS[mode]["videos"].format(DATASET_ITEMPATHS["languages"][0]))):
        os.makedirs(DATASET_PATH, exist_ok=True)
        KaggleUtils_DownloadDataset(DATASET_ITEMPATHS["kaggle"], DATASET_PATH, quiet=False, unzip=True)
    # Load Language-wise Datasets
    dataset_videos = {}
    dataset_category = {}
    for lang in DATASET_ITEMPATHS["languages"]:
        csv_path = os.path.join(path, DATASET_ITEMPATHS[mode]["videos"].format(lang))
        json_path = os.path.join(path, DATASET_ITEMPATHS[mode]["category"].format(lang))
        dataset_videos[lang] = DatasetUtils_LoadCSV(csv_path, encoding="latin-1")
        dataset_category[lang] = json.load(open(json_path, "rb"))
        dataset_category[lang]["item_ids"] = [x["id"] for x in dataset_category[lang]["items"]]
    # Merge Language-wise Datasets
    dataset = pd.DataFrame()
    for lang in DATASET_ITEMPATHS["languages"]:
        ## Add Language Column
        dataset_videos[lang].insert(0, "language", lang)
        ## Add Category Column

        dataset_videos[lang]["category_id"] = dataset_videos[lang]["category_id"].apply(
            lambda x: 
                dataset_category[lang]["items"][dataset_category[lang]["item_ids"].index(str(x))]["snippet"]["title"]
                if str(x) in dataset_category[lang]["item_ids"] else "Unknown"
        )
        ## Merge
        dataset = pd.concat([dataset, dataset_videos[lang]], axis=0)
    dataset.reset_index(drop=True, inplace=True)
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
        ## Get Unique VideoGroups
        d_unique = d.drop_duplicates()
        d_unique.reset_index(drop=True, inplace=True)
        ### Get Unique VideoGroup Indices in Dataset
        VGIndices = np.zeros(d.shape[0], dtype=int)
        for i in range(d.shape[0]):
            d_obj = d.iloc[i]
            for j in range(d_unique.shape[0]):
                if np.array_equal(d_obj, d_unique.iloc[j]):
                    VGIndices[i] = j
                    break
        ### Add VideoGroup Indices Column
        # d_unique.loc[:, "User ID"] = np.arange(d_unique.shape[0])
        b.loc[:, "VG ID"] = VGIndices
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
        ### dataset_d is only the unique videogroups data
        ### dataset_p is only the unique videos data
        ### dataset_b is videogroup-video combination data with extra columns = ["User ID", "Video ID"]
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
    return_labels=True,
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
            "norm_cols": ["views", "likes", "dislikes", "comment_count"],
            "norm_data": {"views": {}, "likes": {}, "dislikes": {}, "comment_count": {}}
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
        d_features_info = []
        boolean_indices = {
            "comments_disabled": list(dataset_d.columns).index("comments_disabled") if "comments_disabled" in dataset_d.columns else None,
            "ratings_disabled": list(dataset_d.columns).index("ratings_disabled") if "ratings_disabled" in dataset_d.columns else None,
            "video_error_or_removed": list(dataset_d.columns).index("video_error_or_removed") if "video_error_or_removed" in dataset_d.columns else None
        }
        date_indices = {
            "trending_date": list(dataset_d.columns).index("trending_date") if "trending_date" in dataset_d.columns else None
        }
        category_indices = {
            "language": list(dataset_d.columns).index("language") if "language" in dataset_d.columns else None,
            "video_id": list(dataset_d.columns).index("video_id") if "video_id" in dataset_d.columns else None,
            "category_id": list(dataset_d.columns).index("category_id") if "category_id" in dataset_d.columns else None
        }
        ## Encode Dataset
        dataset_d = dataset_d.copy()
        ## Convert to numpy array
        dataset_d = dataset_d.to_numpy().astype(object)
        dataset_d_features = np.empty((dataset_d.shape[0], 0), dtype=object)
        ## Encode
        for i in range(dataset_d.shape[1]):
            ### Boolean Encoding
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
            ### Date Encoding
            dateFound = False
            for d in date_indices.keys():
                if date_indices[d] is not None and i == date_indices[d]:
                    #### Encode Date
                    # vals = np.array([EncodeUtils_Encode_Date(dataset_d[j, date_indices[d]], split_key=".") for j in range(dataset_d.shape[0])], dtype=float)
                    vals, _ = EncodeUtils_EncodeArray_Date(dataset_d[:, date_indices[d]], split_key=".")
                    d_features_info.extend([{
                        "name": d+"_"+x,
                        "type": {"type": "number"}
                    } for x in ["D", "M", "Y"]])
                    #### Norm
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
                    #### Concatenate
                    dataset_d_features = np.concatenate((dataset_d_features, vals), axis=1)
                    dateFound = True
                    break
            if dateFound: continue
            ### Category Encoding
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
            ### Number Encoding
            vals = np.array(dataset_d[:, i].reshape(-1, 1), dtype=float)
            d_features_info.append({
                "name": dataset["feature_names"]["demographic"][i],
                "type": {"type": "number"}
            })
            #### Norm
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
            #### Concatenate
            dataset_d_features = np.concatenate((dataset_d_features, vals), axis=1)
        ## Finalize
        dataset_d = dataset_d_features
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
        boolean_indices_b = {
            "comments_disabled": list(dataset_b.columns).index("comments_disabled") if "comments_disabled" in dataset_b.columns else None,
            "ratings_disabled": list(dataset_b.columns).index("ratings_disabled") if "ratings_disabled" in dataset_b.columns else None,
            "video_error_or_removed": list(dataset_b.columns).index("video_error_or_removed") if "video_error_or_removed" in dataset_b.columns else None
        }
        category_indices_p = {
            "video_id": list(dataset_p.columns).index("video_id") if "video_id" in dataset_p.columns else None,
        }
        ## Encode Dataset
        dataset_b = dataset_b.copy()
        dataset_p = dataset_p.copy()
        ## Convert to numpy array
        dataset_b = dataset_b.to_numpy().astype(object)
        dataset_p = dataset_p.to_numpy().astype(object)
        ## Encode
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
        ## Number Encoding
        ### Behavior
        colI_X = list(dataset["behavior"].columns).index("VG ID")
        colI_Y = list(dataset["behavior"].columns).index("Video ID")
        for i in range(dataset_b.shape[1]):
            #### Boolean Encoding
            boolFound = False
            for bk in boolean_indices_b.keys():
                if boolean_indices_b[bk] is not None and i == boolean_indices_b[bk]:
                    # for j in range(dataset_b.shape[0]): dataset_b[j, i] = float(dataset_b[j, boolean_indices_b[bk]])
                    dataset_b[:, boolean_indices_b[bk]] = dataset_b[:, boolean_indices_b[bk]].astype(float)
                    boolFound = True
                    break
            if boolFound: continue
            #### Number Encoding
            if not (i == colI_X or i == colI_Y):
                ##### Norm
                if OTHER_PARAMS["norm"]:
                    if INIT_SESSION:
                        DATASET_SESSION_DATA["norm_data"].update({
                            dataset["behavior"].columns[i]: {
                                "min": dataset_b[:, i].min(),
                                "max": dataset_b[:, i].max()
                            }
                        })
                    dataset_b[:, i] = EncodeUtils_NormData_MinMax(
                        dataset_b[:, i], 
                        DATASET_SESSION_DATA["norm_data"][dataset["behavior"].columns[i]]["min"], 
                        DATASET_SESSION_DATA["norm_data"][dataset["behavior"].columns[i]]["max"]
                    )
        ## Product
        ### Norm
        if OTHER_PARAMS["norm"]:
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
        ### Form Behavior Dataset
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