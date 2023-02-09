"""
Eval Utils
"""

# Imports
import os
import json
import itertools
import numpy as np
from copy import deepcopy

# TQDM
CONFIG = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json"), "r"))
if CONFIG["tqdm_notebook"]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# Util Functions
def name_to_path(name):
    # Convert to Lowercase
    name = name.lower()
    # Remove Special Chars
    for c in [" ", "-", ".", "<", ">", "/", "\\", ",", ";", ":", "'", '"', "|", "*", "?"]:
        name = name.replace(c, "_")

    return name

def SelectKey_StartsWith(KEYS, s):
    '''
    Select Key - Check for key that starts with s (lower cased search)
    '''
    # Init
    s = s.lower()
    # Check
    for KEY in KEYS:
        k = KEY.lower()
        if k.startswith(s):
            return KEY
    return None

# Main Functions
def EvalUtils_GetBestParams(eval_data):
    '''
    Eval Utils - Get Best Params for maximising scores
    '''
    # Init
    best_params = {
        "Silhouette Score": None,
        "Calinski Harabasz Score": None,
        "Davies Bouldin Score": None,
        "Rand Score": None,
        "Adjusted Rand Score": None,
        "V Measure Score": None
    }
    # Iterate
    for ei in range(len(eval_data["evaluation"])):
        e = eval_data["evaluation"][ei]
        e_score = e["eval_data"]["Cluster Evaluations"]
        # Silhouette Score
        if best_params["Silhouette Score"] is None or e_score["Silhouette Score"] > best_params["Silhouette Score"][1]:
            best_params["Silhouette Score"] = (ei, e_score["Silhouette Score"])
        # Calinski Harabasz Score
        if best_params["Calinski Harabasz Score"] is None or e_score["Calinski Harabasz Score"] > best_params["Calinski Harabasz Score"][1]:
            best_params["Calinski Harabasz Score"] = (ei, e_score["Calinski Harabasz Score"])
        # Davies Bouldin Score
        if best_params["Davies Bouldin Score"] is None or e_score["Davies Bouldin Score"] < best_params["Davies Bouldin Score"][1]:
            best_params["Davies Bouldin Score"] = (ei, e_score["Davies Bouldin Score"])
        # Rand Score
        if "Rand Score" in e_score.keys():
            if best_params["Rand Score"] is None or e_score["Rand Score"] > best_params["Rand Score"][1]:
                best_params["Rand Score"] = (ei, e_score["Rand Score"])
        # Adjusted Rand Score
        if "Adjusted Rand Score" in e_score.keys():
            if best_params["Adjusted Rand Score"] is None or e_score["Adjusted Rand Score"] > best_params["Adjusted Rand Score"][1]:
                best_params["Adjusted Rand Score"] = (ei, e_score["Adjusted Rand Score"])
        # V Measure Score
        if "V Measure Score" in e_score.keys():
            if best_params["V Measure Score"] is None or e_score["V Measure Score"] > best_params["V Measure Score"][1]:
                best_params["V Measure Score"] = (ei, e_score["V Measure Score"])
    
    # Output
    best_params = {k: eval_data["evaluation"][best_params[k][0]] for k in best_params.keys()}
    out_data = {k: eval_data[k] for k in eval_data.keys() if k != "evaluation"}
    out_data["best_params"] = best_params
    return out_data