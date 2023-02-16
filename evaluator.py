"""
Evaluator Script
"""

# Imports
import sys
import copy
from UserSegmentation import *
from Utils.EvalUtils import *

# CONFIG
CONFIG = json.load(open("config.json", "r"))

# Utils Functions
def SearchAlgo_FromParamsGridDict(params_grid_dict, algo_key):
    '''
    Search Algo - Search for algo_key in params_grid_dict (lower cased search)
    '''
    # Init
    algo_key = algo_key.lower()
    # Check
    for SEGMENTATION_TYPE_KEY in params_grid_dict.keys():
        for SEGMENTATION_METHOD_KEY in params_grid_dict[SEGMENTATION_TYPE_KEY].keys():
            for SEGMENTATION_ALGO_KEY in params_grid_dict[SEGMENTATION_TYPE_KEY][SEGMENTATION_METHOD_KEY].keys():
                k = SEGMENTATION_ALGO_KEY.lower()
                if k.startswith(algo_key):
                    return (SEGMENTATION_TYPE_KEY, SEGMENTATION_METHOD_KEY, SEGMENTATION_ALGO_KEY)

# Main Functions
def GridParams_ConstructGrid(params_list):
    '''
    Grid Params - Construct all possible Grid Parameters from List
    '''
    # Base Case
    if len(list(params_list.keys())) == 0: return [{}]
    # Get All Possiblities
    PARAM_KEYS = list(params_list.keys())
    PARAMS_ITERLIST = list(itertools.product(*params_list.values()))
    # Construct Grid
    GRID_PARAMS = [{PARAM_KEYS[j]: PARAMS_ITERLIST[i][j] for j in range(len(PARAM_KEYS))} for i in range(len(PARAMS_ITERLIST))]

    return GRID_PARAMS

# RunCode
# Params
EVAL_DIR = CONFIG["eval_dir"]
PARAMS_BEST_MODEL = CONFIG["best_model_params"]
DATASET_PARAMS_DICT = CONFIG["eval_params"]["dataset_params"]
PARAMS_GRID_DICT = CONFIG["eval_params"]["algorithm_grid_params"]
# Dataset Params
DATASET_KEYS = list(DATASETS.keys())
INPUT_DATASET_KEY = "bank" if len(sys.argv) < 2 else str(sys.argv[1])
DATASET_KEY = SelectKey_StartsWith(DATASET_KEYS, INPUT_DATASET_KEY)
INPUT_ALGO_KEY = "dbscan" if len(sys.argv) < 3 else str(sys.argv[2])
ALGO_KEYS = SearchAlgo_FromParamsGridDict(PARAMS_GRID_DICT, INPUT_ALGO_KEY)
SEGMENTATION_TYPE_KEY = ALGO_KEYS[0]
SEGMENTATION_METHOD_KEY = ALGO_KEYS[1]
SEGMENTATION_ALGO_KEY = ALGO_KEYS[2]
N_CLUSTERS_LIST = [2] if len(sys.argv) < 4 else [int(x) for x in sys.argv[3].split(",")]
DATASET_SUFFIX = "" if len(sys.argv) < 5 else str(sys.argv[4])
# Grid Params
PARAMS_GRID_LIST = {
    # "norm": [False, True],
    "norm": [False],
    **PARAMS_GRID_DICT[SEGMENTATION_TYPE_KEY][SEGMENTATION_METHOD_KEY][SEGMENTATION_ALGO_KEY]
}
# Params

# RunCode
for N_CLUSTERS in N_CLUSTERS_LIST:
    # Path Init
    SAVE_DIR = os.path.join(
        EVAL_DIR, name_to_path(DATASET_KEY) + DATASET_SUFFIX, 
        name_to_path(SEGMENTATION_TYPE_KEY), name_to_path(SEGMENTATION_METHOD_KEY), name_to_path(SEGMENTATION_ALGO_KEY),
        str(N_CLUSTERS)
    )
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    # Load Dataset
    DATASET_MODULE = DATASETS[DATASET_KEY]
    DATASET = DATASET_MODULE.DATASET_FUNCS["test"](
        keep_cols=None, 
        data_type=SEGMENTATION_TYPE_KEY,
        other_params=DATASET_PARAMS_DICT[DATASET_KEY]
    )
    Fs, Ls, FEATURES_INFO = DATASET_MODULE.DATASET_FUNCS["encode"](DATASET, n_clusters=N_CLUSTERS)
    # Load Algorithm
    SEGMENTATION_ALGO = SEGMENTATION_MODULES[SEGMENTATION_TYPE_KEY][SEGMENTATION_METHOD_KEY][SEGMENTATION_ALGO_KEY]

    # Grid Search
    GRID_EVAL_DATA = []
    EVAL_BEST_MODEL = {
        "best_eval_metric": PARAMS_BEST_MODEL["best_eval_metric"],
        "params": {},
        "eval_data": {},
        "model": None
    }
    GRID_PARAMS = GridParams_ConstructGrid(PARAMS_GRID_LIST)
    print("N CLUSTERS:", N_CLUSTERS)
    print("ALGO:", SEGMENTATION_ALGO_KEY)
    print("DATASET:", DATASET_KEY, Fs["demographic"].shape)
    print("DATASET PARAMS:", DATASET_PARAMS_DICT[DATASET_KEY])
    print("FEATURES:", FEATURES_INFO)
    # print(GRID_PARAMS)
    for PARAMS in tqdm(GRID_PARAMS):
        try:
            # Update Params
            SEGMENTATION_ALGO["params"].update(PARAMS)
            # Train Model
            MODEL = SEGMENTATION_ALGO["class"](**SEGMENTATION_ALGO["params"])
            MODEL_PARAMS = {
                "features_info": FEATURES_INFO
            }
            # Train Model
            MODEL.train(Fs, Ls, **MODEL_PARAMS)
            # Evaluate Model
            VisData = MODEL.visualise(disable_plots=True)
            if SEGMENTATION_TYPE_KEY == "demographic" and True: del VisData["data"]["Cluster Centers"]
            else:
                cluster_center_featurenames = list(VisData["data"]["Cluster Centers"].columns)
                cluster_centers = []
                for i in range(VisData["data"]["Cluster Centers"].shape[0]):
                    cluster_centers.append({
                        k: VisData["data"]["Cluster Centers"][k][i] for k in cluster_center_featurenames
                    })
                VisData["data"]["Cluster Centers"] = cluster_centers
            EVAL_DATA = VisData["data"]
            # Record Data
            GRID_EVAL_DATA.append({
                "params": deepcopy(SEGMENTATION_ALGO["params"]),
                "eval_data": deepcopy(EVAL_DATA)
            })
            # Record Best Model
            if EVAL_BEST_MODEL["model"] is None or \
            EVAL_DATA["Cluster Evaluations"][PARAMS_BEST_MODEL["best_eval_metric"]] > \
            EVAL_BEST_MODEL["eval_data"]["Cluster Evaluations"][PARAMS_BEST_MODEL["best_eval_metric"]]:
                EVAL_BEST_MODEL["params"] = deepcopy(SEGMENTATION_ALGO["params"])
                EVAL_BEST_MODEL["eval_data"] = deepcopy(EVAL_DATA)
                EVAL_BEST_MODEL["model"] = MODEL

        except Exception as e:
            print()
            print("PARAMS:")
            print(PARAMS)
            print("ERROR:")
            print(e)
            print()
            continue
    OUT_DATA = {
        "dataset": DATASET_KEY,
        "segmentation_type": SEGMENTATION_TYPE_KEY,
        "segmentation_method": SEGMENTATION_METHOD_KEY,
        "segmentation_algorithm": SEGMENTATION_ALGO_KEY,
        "n_clusters": N_CLUSTERS,
        "evaluation": GRID_EVAL_DATA
    }
    # print(OUT_DATA)

    # Save Data
    json.dump(OUT_DATA, open(os.path.join(SAVE_DIR, "eval.json"), "w"), indent=4)

    # Compute and Save Best Params
    BEST_PARAMS = EvalUtils_GetBestParams(OUT_DATA)
    json.dump(BEST_PARAMS, open(os.path.join(SAVE_DIR, "best_params.json"), "w"), indent=4)

    # Save Best Model
    if PARAMS_BEST_MODEL["save"] and EVAL_BEST_MODEL["model"] is not None:
        BEST_MODEL_PARAMS = {
            "best_eval_metric": PARAMS_BEST_MODEL["best_eval_metric"],
            "best_params": EVAL_BEST_MODEL["params"],
            "best_eval_data": EVAL_BEST_MODEL["eval_data"]
        }
        json.dump(BEST_MODEL_PARAMS, open(os.path.join(SAVE_DIR, "best_model_params.json"), "w"), indent=4)
        MODEL = EVAL_BEST_MODEL["model"]
        MODEL.save(SAVE_DIR)

# Commands
## Params
# Algorithm Params
# [
#     "DBSCAN", "OPTICS", "MeanShift", "AffinityPropagation", 
#     "KMeans++", "MiniBatchKMeans++", "KPrototypes",
#     "Agglomerative Clustering", "Birch", 
#     "Spectral", 
#     "Fuzzy CMeans", "GMM", 
#     "SOM"
# ]
# ["NMF + Decision Tree"]

# dataset_key: bank, caravan, credit, mall, youtube, "youtube tr"
# algo_key: dbscan, optics, meanshift, aff, kmeans, minibatchkmeans, kprototypes, agg, birch, spec, fuzzy, gmm, som, nmf
# n_clusters: 2,3,4,5,6,7,8,9,10

# Path Commands
# cd .\_Academics\IITM_Files\_Projects\MTech_MTP_UserSegmentation\