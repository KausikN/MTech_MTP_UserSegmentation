"""
Evaluator - Visualisation Script
"""

# Imports
import sys
import json
import math
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from UserSegmentation import *
from Utils.EvalUtils import *

# CONFIG
CONFIG = json.load(open("config.json", "r"))

# Main Vars
FIGSIZE = tuple(CONFIG["eval_vis_params"]["figsize"])
EVALMETRIC_KEYS = CONFIG["eval_vis_params"]["eval_metrics"]
EVALMETRIC_INVERSE = CONFIG["eval_vis_params"]["eval_metrics_inverse"]
ALGORITHM_KEYS = CONFIG["eval_vis_params"]["algorithms"]
ALGORITHM_TYPES = {
    "Density-Based": ["DBSCAN", "OPTICS", "MeanShift", "AffinityPropagation"], 
    "Centroid-Based": ["KMeans++", "MiniBatchKMeans++", "KPrototypes"],
    "Hierarchy-Based": ["Agglomerative Clustering", "Birch"],
    "Graph-Based": ["Spectral"],
    "Fuzzy-Based": ["Fuzzy CMeans", "GMM"],
    "Combined": ["SOM"]
}
EVAL_DATA_FORMAT = {
    "params": {
        "n_clusters": None,
    },
    "data": {
        sk: {
            ak: [] for ak in ALGORITHM_KEYS
        } for sk in EVALMETRIC_KEYS
    },
    "other_data": {
        "Time": {
            ak: [] for ak in ALGORITHM_KEYS
        }
    }
}

# Main Functions
def LoadEvalData_FromCSV(dir_path, n_clusters):
    '''
    Load Evaluation Data from CSV
    '''
    # Init
    EVAL_DATA = EVAL_DATA_FORMAT.copy()
    # Load
    for nc in n_clusters:
        path = os.path.join(dir_path, f"eval_data_{nc}.csv")
        data = pd.read_csv(path, index_col=0)
        # Load Eval Metrics Data
        for sk in EVALMETRIC_KEYS:
            for ak in ALGORITHM_KEYS:
                EVAL_DATA["data"][sk][ak].append(data[sk][ak])
        # Load Execution Time Data
        for ak in ALGORITHM_KEYS:
            EVAL_DATA["other_data"]["Time"][ak].append(0.0)
    EVAL_DATA["params"]["n_clusters"] = n_clusters

    return EVAL_DATA

def LoadEvalData_JSONDIR(eval_dir_path, n_clusters, key_eval_metric="Silhouette Score"):
    '''
    Load Evaluation Data from Directory of JSON Files
    '''
    # Init
    EVAL_DATA = EVAL_DATA_FORMAT.copy()
    ALGORITHMS = list(ALGORITHM_KEYS)
    # Load
    for algo in ALGORITHMS:
        algo_path = None
        for k in ALGORITHM_TYPES.keys():
            if algo in ALGORITHM_TYPES[k]:
                algo_path = os.path.join(eval_dir_path, name_to_path(k), name_to_path(algo))
                break
        if algo_path is None: continue
        for nc in n_clusters:
            file_path = os.path.join(algo_path, str(nc), "best_params.json")
            if os.path.exists(file_path):
                data = json.load(open(file_path, "r"))
                eval_vals = data["best_params"][key_eval_metric]["eval_data"]["Cluster Evaluations"]
                # Load Eval Metrics Data
                for ek in eval_vals.keys():
                    if ek in EVALMETRIC_KEYS:
                        EVAL_DATA["data"][ek][algo].append(eval_vals[ek])
                # Load Execution Time Data
                EVAL_DATA["other_data"]["Time"][algo].append(
                    data["best_params"][key_eval_metric]["eval_data"]["Time"]["train"]["overall"]["time"]
                )
            else:
                # Load Eval Metrics Data
                for ek in EVALMETRIC_KEYS:
                    EVAL_DATA["data"][ek][algo].append(None)
                # Load Execution Time Data
                EVAL_DATA["other_data"]["Time"][algo].append(None)
    EVAL_DATA["params"]["n_clusters"] = n_clusters

    return EVAL_DATA

def EvalPlot_Comparison(EVAL_DATA):
    '''
    Eval Plot - Comparison Plot
    '''
    # Init
    eval_params = EVAL_DATA["params"]
    eval_data = EVAL_DATA["data"]
    eval_data.update(EVAL_DATA["other_data"])
    N_COLS = 2
    FIGS = {
        "ComparisonPlot": {},
        "RadarPlot": {},
        "RadarRankPlot": {}
    }
    METRICS = [k for k in EVALMETRIC_KEYS if len(eval_data[k][ALGORITHM_KEYS[0]]) > 0]
    METRICS.append("Time")
    # Score Plot - Separate
    for fig_key in METRICS:
        n_clusters = eval_params["n_clusters"]
        data = eval_data[fig_key]
        FIGS["ComparisonPlot"][fig_key] = plt.figure(figsize=FIGSIZE)
        plt.xlabel("Number of Clusters")
        plt.ylabel(fig_key)
        plt.title(f"{fig_key} Comparison")
        for k in data.keys():
            n_clusters_avail = [n_clusters[i] for i in range(len(n_clusters)) if data[k][i] is not None]
            data_available = [data[k][i] for i in range(len(n_clusters)) if data[k][i] is not None]
            if len(n_clusters_avail) == 0: continue
            plt.plot(
                n_clusters_avail, data_available, 
                label=k, alpha=0.5
            )
            plt.scatter(
                n_clusters_avail, data_available, 
                label="_"+k, alpha=0.5
            )
        plt.legend()
        plt.close(FIGS["ComparisonPlot"][fig_key])
    # Score Plot - Combined
    fig_combined = plt.figure(figsize=FIGSIZE)
    fig_keys = METRICS
    N_ROWS = math.ceil(len(METRICS) / N_COLS)
    for i in range(len(fig_keys)):
        fig_key = fig_keys[i]
        n_clusters = eval_params["n_clusters"]
        data = eval_data[fig_key]
        plt.subplot(N_ROWS, N_COLS, i+1)
        plt.xlabel("Number of Clusters")
        plt.ylabel(fig_key)
        plt.title(f"{fig_key} Comparison")
        for k in data.keys():
            n_clusters_avail = [n_clusters[ii] for ii in range(len(n_clusters)) if data[k][ii] is not None]
            data_available = [data[k][ii] for ii in range(len(n_clusters)) if data[k][ii] is not None]
            if len(n_clusters_avail) == 0: continue
            plt.plot(
                n_clusters_avail, data_available, 
                label=k, alpha=0.5
            )
            plt.scatter(
                n_clusters_avail, data_available, 
                label="_"+k, alpha=0.5
            )
        plt.legend()
    plt.close(fig_combined)
    FIGS["ComparisonPlot"]["Combined"] = fig_combined
    # Radar Plot
    n_clusters = eval_params["n_clusters"]
    metric_names = METRICS
    radar_categories = [*metric_names, metric_names[0]]
    radar_angles = np.linspace(start=0, stop=2*np.pi, num=len(radar_categories))
    for i, nc in enumerate(n_clusters):
        ## Init
        eval_data_nc_norm = {k: {} for k in metric_names}
        eval_data_nc_order = {k: None for k in metric_names}
        available_indices = [
            algo_i for algo_i in range(len(ALGORITHM_KEYS)) 
            if eval_data[metric_names[0]][ALGORITHM_KEYS[algo_i]][i] is not None and
            np.isfinite([eval_data[mname][ALGORITHM_KEYS[algo_i]][i] for mname in metric_names]).all() 
        ]
        for ek in metric_names:
            eval_data_nc_norm[ek] = np.array([eval_data[ek][ALGORITHM_KEYS[algo_i]][i] for algo_i in available_indices])
            eval_data_nc_order[ek] = stats.rankdata(eval_data_nc_norm[ek])
            val_min, val_max = np.min(eval_data_nc_norm[ek]), np.max(eval_data_nc_norm[ek])
            if val_min == val_max:
                eval_data_nc_norm[ek] = np.zeros_like(eval_data_nc_norm[ek])
            else:
                eval_data_nc_norm[ek] = (eval_data_nc_norm[ek] - val_min) / (val_max - val_min)
            if ek in EVALMETRIC_INVERSE:
                eval_data_nc_norm[ek] = 1.0 - eval_data_nc_norm[ek]
                eval_data_nc_order[ek] = np.max(eval_data_nc_order[ek]) - eval_data_nc_order[ek]
        ## Radar Plot - Normalised
        FIGS["RadarPlot"][nc] = plt.figure(figsize=FIGSIZE)
        for algo_ii, algo_i in enumerate(available_indices):
            algo = ALGORITHM_KEYS[algo_i]
            data = [eval_data_nc_norm[ek][algo_ii] for ek in metric_names]
            if None in data: continue
            data.append(data[0])
            data = np.array(data)
            plt.subplot(polar=True)
            plt.plot(
                radar_angles, data, 
                label=algo, alpha=0.5
            )
            plt.scatter(
                radar_angles, data, 
                label="_"+algo, alpha=0.5
            )
        _, _ = plt.thetagrids(np.degrees(radar_angles), labels=radar_categories)
        plt.legend()
        plt.title(f"Radar Plot - {nc} Clusters")
        plt.close(FIGS["RadarPlot"][nc])
        ## Radar Rank Plot
        FIGS["RadarRankPlot"][nc] = plt.figure(figsize=FIGSIZE)
        for algo_ii, algo_i in enumerate(available_indices):
            algo = ALGORITHM_KEYS[algo_i]
            data = [eval_data_nc_order[ek][algo_ii] for ek in metric_names]
            if None in data: continue
            data.append(data[0])
            data = np.array(data)
            plt.subplot(polar=True)
            plt.plot(
                radar_angles, data, 
                label=algo, alpha=0.5
            )
            plt.scatter(
                radar_angles, data, 
                label="_"+algo, alpha=0.5
            )
        _, _ = plt.thetagrids(np.degrees(radar_angles), labels=radar_categories)
        plt.legend()
        plt.title(f"Radar Rank Plot - {nc} Clusters")
        plt.close(FIGS["RadarRankPlot"][nc])

    return FIGS

def EvalDataFrame_Comparison(EVAL_DATA):
    '''
    Eval DataFrame - Comparison DataFrame
    '''
    # Init
    eval_params = EVAL_DATA["params"]
    eval_data = EVAL_DATA["data"]
    eval_data.update(EVAL_DATA["other_data"])
    DFS = {
        k: None for k in EVALMETRIC_KEYS if len(eval_data[k][ALGORITHM_KEYS[0]]) > 0
    }
    DFS.update({
        "Time": None
    })
    # Score DataFrame
    for df_key in DFS.keys():
        n_clusters = eval_params["n_clusters"]
        data = eval_data[df_key]
        df = pd.DataFrame(data, index=n_clusters)
        DFS[df_key] = df

    return DFS

# RunCode
# Params
EVAL_DATA_DIR = CONFIG["eval_dir"] # "../Evaluations/_evaluations/"
EVAL_FIG_DIR = os.path.join(EVAL_DATA_DIR, "_figures/")
EVAL_DF_DIR = os.path.join(EVAL_DATA_DIR, "_comparisons/")
# Params
# Algorithms Type
# demographic
# {
#     "Density-Based": ["DBSCAN", "OPTICS", "MeanShift", "AffinityPropagation"], 
#     "Centroid-Based": ["KMeans++"],
#     "Hierarchy-Based": ["Agglomerative Clustering", "Birch"],
#     "Graph-Based": ["Spectral"],
#     "Fuzzy-Based": ["Fuzzy CMeans", "GMM"],
#     "Combined": ["SOM"]
# }
# demographic-behavior
# {
#   "Cluster-Classifier": ["NMF + Decision Tree"]
# }
SEGMENTATION_TYPE_KEY = "demographic"
# Algorithms Type
# Data
DATASET_KEYS = [
    "Bank Customers 1", "Caravan Insurance Challenge", "Credit Card 1", "Mall Customers", 
    "Youtube Videos-Users 1", "Youtube Trending-Videos 1"
]
INPUT_DATASET_KEY = "bank" if len(sys.argv) < 2 else str(sys.argv[1])
DATASET_KEY = SelectKey_StartsWith(DATASET_KEYS, INPUT_DATASET_KEY)
N_CLUSTERS = [2] if len(sys.argv) < 3 else [int(x) for x in sys.argv[2].split(",")]
DATASET_SUFFIX = "" if len(sys.argv) < 4 else str(sys.argv[3])
# Data

# Load
DATASET_PATH_STR = name_to_path(DATASET_KEY) + DATASET_SUFFIX
EVAL_DATASET_DIR = os.path.join(
    EVAL_DATA_DIR, DATASET_PATH_STR, 
    name_to_path(SEGMENTATION_TYPE_KEY)
)
EVAL_DATA = LoadEvalData_JSONDIR(EVAL_DATASET_DIR, N_CLUSTERS)
# print(json.dumps(EVAL_DATA, indent=4))
# Comparison CSV
if True:
    DFS = EvalDataFrame_Comparison(EVAL_DATA)
    DIR_PATH = os.path.join(EVAL_DF_DIR, DATASET_PATH_STR)
    if not os.path.exists(DIR_PATH): os.makedirs(DIR_PATH)
    for df_key in DFS.keys():
        DFS[df_key].to_csv(
            os.path.join(DIR_PATH, f"{DATASET_KEY}_{df_key}.csv"), 
            index=True
        )
# # Plot
if True:
    FIGS = EvalPlot_Comparison(EVAL_DATA)
    DIR_PATH = os.path.join(EVAL_FIG_DIR, DATASET_PATH_STR)
    if not os.path.exists(DIR_PATH): os.makedirs(DIR_PATH)
    for fig_type_key in FIGS.keys():
        for fig_key in FIGS[fig_type_key].keys():
            FIGS[fig_type_key][fig_key].savefig(os.path.join(
                DIR_PATH, 
                f"{DATASET_KEY}_{fig_type_key}_{fig_key}.png"
            ))

# Commands
# cd .\_Academics\IITM_Files\_Projects\MTech_MTP_UserSegmentation\
# python evaluator_vis.py bank 2,3,4,5,6,7,8,9,10 "_withoutcat_0.1k"