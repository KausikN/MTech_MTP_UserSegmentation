"""
Utils
"""

# Imports
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, jaccard_score, hamming_loss, zero_one_loss

# Main Classes
class UserSegmentation_Interpret_Base:
    def __init__(self,
    clustering_algorithm=None,
    **params
    ):
        '''
        User Segmentation - Interpret - Algorithm

        Params:
         - clustering_algorithm : Clustering Algorithm Object used to get Cluster Labels
        '''
        self.clustering_algorithm = clustering_algorithm
        self.__dict__.update(params)

    def train(self,
        **params
        ):
        '''
        Train

        Train interpretation algorithm on given features and labels from clustering algorithm.

        Inputs: None
            
        Outputs:
            - interpretation_model : Model used to interpret clusters
        '''
        return None
        
    def visualise(self):
        '''
        Visualise
        '''
        return {}
    
    def save(self, path):
        '''
        Save
        '''
        # Init
        path_data = os.path.join(path, "data.p")
        data = self.__dict__
        # Save
        with open(path_data, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        '''
        Load
        '''
        # Init
        path_data = os.path.join(path, "data.p")
        # Load
        with open(path_data, "rb") as f:
            data = pickle.load(f)
        # Update
        self.__dict__.update(data)

# Main Functions
# Array Functions
def Array_OneHot2Index(a, feature_types, dtype={"numerical": float, "categorical": int}):
    '''
    Array - One Hot 2 Index
    '''
    # Check which columns are lists
    category_check = []
    category_cols = []
    for j in range(a.shape[1]):
        if feature_types[j]["type"] == "category":
            category_check.append(True)
            category_cols.append(j)
        else:
            category_check.append(False)
    # Convert
    a_converted = []
    for i in range(a.shape[0]):
        row = []
        for j in range(a.shape[1]):
            if category_check[j]:
                row.append(dtype["categorical"](np.argmax(a[i, j])))
            else:
                row.append(dtype["numerical"](a[i, j]))
        a_converted.append(row)
    a_converted = np.array(a_converted)

    return a_converted, category_cols

def Array_Flatten(a, feature_types, dtype=float):
    '''
    Array - Flatten
    '''
    # Find Pos of each element
    list_check = []
    pos = []
    curPos = 0
    for j in range(len(feature_types)):
        if feature_types[j]["type"] == "category":
            list_check.append(True)
            pos.append([curPos, curPos + len(feature_types[j]["categories"])])
        else:
            list_check.append(False)
            pos.append([curPos, curPos + 1])
        curPos = pos[-1][1]
    # Flatten
    a_flat = []
    for i in range(a.shape[0]):
        row = []
        for j in range(a.shape[1]):
            if list_check[j]:
                row.extend(a[i, j])
            else:
                row.append(a[i, j])
        a_flat.append(row)
    a_flat = np.array(a_flat, dtype=dtype)

    return a_flat, pos

def Array_FlatFeaturesNameMap(feature_names, feature_types, pos):
    '''
    Array - Flat Features Name Map
    '''
    # Map
    feature_names_flat = []
    for i in range(len(feature_types)):
        if pos[i][0] == pos[i][1] - 1:
            feature_names_flat.append(feature_names[i])
        else:
            if feature_types[i]["type"] == "category":
                for j in range(len(feature_types[i]["categories"])):
                    feature_names_flat.append(feature_names[i] + "_" + str(feature_types[i]["categories"][j]))
    feature_names_flat = np.array(feature_names_flat)

    return feature_names_flat

def Array_Normalize(a, norm_params=None):
    '''
    Array - Normalize (Only works on array of flat arrays)
    '''
    # Normalize
    if norm_params is None:
        a_min, a_max = np.min(a, axis=0), np.max(a, axis=0)
        a_minmaxequal = a_min == a_max
        a_min[a_minmaxequal] = 0.0
        a_max[a_minmaxequal] = 1.0
        a_norm = (a - a_min) / (a_max - a_min)
        norm_params = {
            "mins": a_min,
            "maxs": a_max
        }
    else:
        a_norm = (a - norm_params["mins"]) / (norm_params["maxs"] - norm_params["mins"])

    return a_norm, norm_params

# Time Functions
def Time_Record(name, time_data=None, finish=False):
    '''
    Time - Record
    '''
    # Init
    if time_data is None:
        curtime = time.time()
        time_data = {
            "overall": {
                "title": name
            },
            "current": {
                "prev": curtime,
                "cur": curtime
            },
            "record": []
        }
        return time_data
    # Finish
    if finish:
        time_data["current"]["cur"] = time.time()
        time_data["overall"].update({
            "time": sum([i["time"] for i in time_data["record"]]),
        })
        del time_data["current"]
        return time_data
    # Record
    time_data["current"]["cur"] = time.time()
    time_data["record"].append({
        "name": name,
        "time": time_data["current"]["cur"] - time_data["current"]["prev"]
    })
    time_data["current"]["prev"] = time_data["current"]["cur"]
    return time_data

# Plot Functions
def ClusterVis_ClusterMapPlot(
    feature_points, cluster_labels, 
    cluster_centers_unique, unique_labels,
    feature_names,
    cluster_labels_soft=None
    ):
    '''
    Cluster Vis - Cluster Map Plot
    '''
    # Init
    COLORS = {
        "AXIS": "green",
        "BACKGROUND": "white"
    }
    if cluster_labels_soft is None: alphas = [0.5 for i in unique_labels]
    else:
        alphas = [cluster_labels_soft[cluster_labels==unique_labels[ii]][:, ii] for ii in range(len(unique_labels))]
        alphas = [(a - a.min()) / (a.max() - a.min()) for a in alphas]
        alphas = [(a * 0.75) + 0.25 for a in alphas]
    # Cluster Map Plot
    fig_map = plt.figure()
    ## Convert Features and Cluster Centers to points
    c_pts, pts = None, None
    # ND
    if feature_points.shape[1] >= 3:
        ## Init Points
        c_pts = np.array(cluster_centers_unique)
        pts = np.array(feature_points)
        ## Init Plot
        for fi in range(feature_points.shape[1]):
            for fj in range(0, fi+1):
                ax_nd_cur = fig_map.add_subplot(
                    feature_points.shape[1], feature_points.shape[1], 
                    fi * feature_points.shape[1] + fj + 1
                )
                ## Plot Points
                for ii, i in enumerate(unique_labels):
                    ax_nd_cur.scatter(
                        pts[cluster_labels == i, fj], pts[cluster_labels == i, fi], 
                        label=i, alpha=alphas[ii]
                    )
                ## Plot Centers
                ax_nd_cur.scatter(
                    c_pts[:, fj], c_pts[:, fi], 
                    marker="x", color="black", label="Center"
                )
                ## Settings
                if fi == 0 and fj == 0: plt.title("Cluster Map", color=COLORS["AXIS"])
                if fj == 0:
                    ax_nd_cur.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
                    ax_nd_cur.set_ylabel(feature_names[fi], color=COLORS["AXIS"], fontsize=5)
                if fi == feature_points.shape[1] - 1:
                    ax_nd_cur.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
                    ax_nd_cur.set_xlabel(feature_names[fj], color=COLORS["AXIS"], fontsize=5)
                ## Other Settings
                if fj != 0 and fi != feature_points.shape[1] - 1:
                    ax_nd_cur.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        fig_map.subplots_adjust(wspace=0, hspace=0)
    # 3D
    elif feature_points.shape[1] == 3:
        ## Init Points
        c_pts = np.zeros((cluster_centers_unique.shape[0], 3))
        pts = np.zeros((feature_points.shape[0], 3))
        c_pts[:, :feature_points.shape[1]] = cluster_centers_unique
        pts[:, :feature_points.shape[1]] = feature_points
        ## Init Plot
        ax_3d = plt.axes(projection ="3d")
        ax_3d.set_xlabel(feature_names[0])
        ax_3d.set_ylabel(feature_names[1])
        ax_3d.set_zlabel(feature_names[2])
        ax_3d.xaxis.label.set_color(COLORS["AXIS"])
        ax_3d.yaxis.label.set_color(COLORS["AXIS"])
        ax_3d.zaxis.label.set_color(COLORS["AXIS"])
        ## Plot Points
        for ii, i in enumerate(unique_labels):
            ax_3d.scatter3D(
                pts[cluster_labels == i, 0], pts[cluster_labels == i, 1], pts[cluster_labels == i, 2], 
                label=i, alpha=alphas[ii]
            )
        ## Plot Centers
        ax_3d.scatter3D(
            c_pts[:, 0], c_pts[:, 1], c_pts[:, 2],
            marker="x", color="black", label="Center"
        )
    # 2D
    else:
        ## Init Plot
        ax_2d = plt.axes()
        ax_2d.xaxis.label.set_color(COLORS["AXIS"])
        ax_2d.yaxis.label.set_color(COLORS["AXIS"])
        ax_2d.set_facecolor(COLORS["BACKGROUND"])
        ## Init Points
        if feature_points.shape[1] <= 2:
            c_pts = np.zeros((cluster_centers_unique.shape[0], 2))
            pts = np.zeros((feature_points.shape[0], 2))
            c_pts[:, :feature_points.shape[1]] = cluster_centers_unique
            pts[:, :feature_points.shape[1]] = feature_points
            ax_2d.set_xlabel(feature_names[0])
            if len(feature_names) > 1: ax_2d.set_ylabel(feature_names[1])
            else: ax_2d.set_ylabel("-")
        else:
            c_pts = np.zeros((cluster_centers_unique.shape[0], 2))
            pts = np.zeros((feature_points.shape[0], 2))
            c_pts[:, 0] = np.mean(cluster_centers_unique, axis=-1)
            c_pts[:, 1] = np.std(cluster_centers_unique, axis=-1)
            pts[:, 0] = np.mean(feature_points, axis=-1)
            pts[:, 1] = np.std(feature_points, axis=-1)
            ax_2d.set_xlabel("Mean")
            ax_2d.set_ylabel("Std")
        ## Plot Points
        for ii, i in enumerate(unique_labels):
            ax_2d.scatter(pts[cluster_labels == i, 0], pts[cluster_labels == i, 1], label=i, alpha=alphas[ii])
        ## Plot Centers
        ax_2d.scatter(c_pts[:, 0], c_pts[:, 1], marker="x", color="black", label="Center")
    ## Other Plot Settings
    if feature_points.shape[1] <= 3:
        plt.title("Cluster Map", color=COLORS["AXIS"])
        if len(unique_labels) <= 15: plt.legend()

    OutData = {
        "fig": fig_map
    }
    return OutData

# Evaluation Functions
def ClassifierEval_Basic(y_true, y_pred):
    '''
    Cluster Eval - Basic
    '''
    # Init
    if y_true.ndim == 1 or y_pred.ndim == 1:
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=-1)
        elif y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=-1)
    y_true_indices = np.argmax(y_true, axis=-1) if y_true.ndim > 1 else y_true
    y_pred_indices = np.argmax(y_pred, axis=-1) if y_pred.ndim > 1 else y_pred
    # Eval
    cluster_evals = {
        "confusion_matrix": confusion_matrix(y_true_indices, y_pred_indices).tolist(),
        "accuracy": accuracy_score(y_true_indices, y_pred_indices),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "cohen_kappa": cohen_kappa_score(y_true_indices, y_pred_indices),
        "matthews_corrcoef": matthews_corrcoef(y_true_indices, y_pred_indices),
        "jaccard": jaccard_score(y_true, y_pred, average="weighted"),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "zero_one_loss": zero_one_loss(y_true, y_pred)
    }

    return cluster_evals