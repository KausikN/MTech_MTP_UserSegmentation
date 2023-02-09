"""
Evaluator Command Generator
"""

# Imports
import os
import sys
import json

# Main Functions

# RunCode
## Params
# dataset_key: bank, caravan, credit, mall, youtube, "youtube tr"
DATASET = "bank" if len(sys.argv) < 2 else sys.argv[1]
DEFAULT_CLUSTER = "2" if len(sys.argv) < 3 else sys.argv[2]
DEFAULT_CLUSTER_RANGE = "2,3,4,5,6,7,8,9,10" if len(sys.argv) < 4 else sys.argv[3]
DATASET_SUFFIX = "_9k" if len(sys.argv) < 5 else sys.argv[4]

SAVE_DIR = "_evaluations/"
## Params

# RunCode
## Algorithms
ALGORITHMS_SINGLECC = {
    "Density-Based": ["dbscan", "optics", "meanshift", "aff"]
}
ALGORITHMS_MULTICC = {
    # "Centroid-Based": ["kmeans", "minibatchkmeans"],
    "Centroid-Based": ["kmeans", "minibatchkmeans", "kprototypes"],
    "Hierarchy-Based": ["agg", "birch"],
    "Graph-Based": ["spec"],
    "Fuzzy-Based": ["fuzzy", "gmm"],
    "Combined": ["som"]
}
## Form Individual Commands
EVAL_CMDS_INDIV = {}
EVAL_CMDS_INDIV.update({
    k: [
        f'python evaluator.py "{DATASET}" "{algo}" "{DEFAULT_CLUSTER}" "{DATASET_SUFFIX}"' for algo in ALGORITHMS_SINGLECC[k]
    ] for k in ALGORITHMS_SINGLECC.keys()
})
EVAL_CMDS_INDIV.update({
    k: [
        f'python evaluator.py "{DATASET}" "{algo}" "{DEFAULT_CLUSTER_RANGE}" "{DATASET_SUFFIX}"' for algo in ALGORITHMS_MULTICC[k]
    ] for k in ALGORITHMS_MULTICC.keys()
})
## Form Group Commands
EVAL_CMDS_GROUP = {
    k: ";".join(EVAL_CMDS_INDIV[k]) for k in EVAL_CMDS_INDIV.keys()
}
## Form All Commands
EVAL_CMDS_ALL = ";".join([EVAL_CMDS_GROUP[k] for k in EVAL_CMDS_GROUP.keys()])
## Form Eval Commands
EVAL_CMDS = {
    "Individual": EVAL_CMDS_INDIV,
    "Group": EVAL_CMDS_GROUP,
    "All": EVAL_CMDS_ALL
}
CMD_STR = "# Individual Commands\n"
for algo_type in EVAL_CMDS_INDIV.keys():
    CMD_STR += f"## {algo_type}\n"
    for cmd in EVAL_CMDS_INDIV[algo_type]:
        CMD_STR += f"{cmd}\n"
CMD_STR += "# Group Commands\n"
for algo_type in EVAL_CMDS_GROUP.keys():
    CMD_STR += f"## {algo_type}\n"
    CMD_STR += f"{EVAL_CMDS_GROUP[algo_type]}\n"
CMD_STR += "# All Commands\n"
CMD_STR += f"{EVAL_CMDS_ALL}\n\n"
CMD_STR += "# Eval Visualizer Command\n"
CMD_STR += f'python evaluator_vis.py "{DATASET}" "{DEFAULT_CLUSTER_RANGE}" "{DATASET_SUFFIX}"\n'

# Display and Save
# print(CMD_STR)
open(os.path.join(SAVE_DIR, f"eval_cmds.sh"), "w").write(CMD_STR)
# json.dump(EVAL_CMDS, open(os.path.join(SAVE_DIR, f"eval_cmds.json"), "w"), indent=4)

# Path Commands
# cd .\_Academics\IITM_Files\_Projects\MTech_MTP_UserSegmentation\

# Run Commands
# python eval_cmd_gen.py "bank" "2" "2,3,4,5,6,7,8,9,10" "_9k"