"""
Utils
"""

# Imports
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import fowlkes_mallows_score

# Main Classes
class UserSegmentation_Cluster_Base:
    def __init__(self,
    model=None,
    **params
    ):
        '''
        User Segmentation - Demographic and Behavior Combined - Algorithm

        Params:
         - model : Pretained model
        '''
        self.model = model
        self.__dict__.update(params)

    def train(self,
        Fs, Ls, 
        feature_names={"demographic": []},
        **params
        ):
        '''
        Train

        Train clustering algorithm on given features and labels.

        Inputs:
            - Fs : Features of Users
                - demographic : (N_Samples, N_Features, Feature_Dim)
                - behavior : (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        return None
        
    def visualise(self):
        '''
        Visualise
        '''
        return {}

    def predict(self,
        Fs, 

        **params
        ):
        '''
        Predict

        Segment users based on clustering algorithm.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)

        Outputs:
            - Ls : Label Distributions of Users (N_Samples, Label_Dim)
        '''
        return None
    
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
def Array_Flatten_Naive(a, dtype=float):
    '''
    Array - Flatten
    '''
    # Find Pos of each element (Naive - Without type info)
    list_check = []
    pos = []
    curPos = 0
    for j in range(a.shape[1]):
        if isinstance(a[0, j], list):
            list_check.append(True)
            pos.append([curPos, curPos + len(a[0, j])])
            curPos += len(a[0, j])
        else:
            list_check.append(False)
            pos.append([curPos, curPos + 1])
            curPos += 1
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

def Array_FlatFeaturesNameMap_Naive(feature_names, pos):
    '''
    Array - Flat Features Name Map (Naive - Without type info)
    '''
    # Map
    feature_names_flat = []
    for i in range(len(pos)):
        if pos[i][0] == pos[i][1] - 1:
            feature_names_flat.append(feature_names[i])
        else:
            for j in range(pos[i][0], pos[i][1]):
                feature_names_flat.append(feature_names[i] + "_" + str(j))
    feature_names_flat = np.array(feature_names_flat)

    return feature_names_flat

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
    feature_names
    ):
    '''
    Cluster Vis - Cluster Map Plot
    '''
    # Init
    COLORS = {
        "AXIS": "green",
        "BACKGROUND": "white"
    }
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
                for i in unique_labels:
                    ax_nd_cur.scatter(
                        pts[cluster_labels == i, fj], pts[cluster_labels == i, fi], 
                        label=i, alpha=0.5
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
        for i in unique_labels:
            ax_3d.scatter3D(
                pts[cluster_labels == i, 0], pts[cluster_labels == i, 1], pts[cluster_labels == i, 2], 
                label=i, alpha=0.5
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
        for i in unique_labels:
            ax_2d.scatter(pts[cluster_labels == i, 0], pts[cluster_labels == i, 1], label=i, alpha=0.5)
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
def ClusterEval_Basic(feature_points, cluster_labels, unique_labels):
    '''
    Cluster Eval - Basic
    '''
    cluster_evals = {
        "Silhouette Score": silhouette_score(feature_points, cluster_labels),
        "Calinski Harabasz Score": calinski_harabasz_score(feature_points, cluster_labels),
        "Davies Bouldin Score": davies_bouldin_score(feature_points, cluster_labels)
    } if unique_labels.shape[0] > 1 else {
        "Silhouette Score": -1.0,
        "Calinski Harabasz Score": -1.0,
        "Davies Bouldin Score": np.inf
    }

    return cluster_evals

def ClusterEval_External(cluster_labels_true, cluster_labels_pred, unique_labels):
    '''
    Cluster Eval - Basic
    '''
    cluster_evals = {
        "Rand Score": rand_score(cluster_labels_true, cluster_labels_pred),
        "Adjusted Rand Score": adjusted_rand_score(cluster_labels_true, cluster_labels_pred),
        "Normalised Mutual Information Score": normalized_mutual_info_score(cluster_labels_true, cluster_labels_pred),
        "Adjusted Mutual Information Score": adjusted_mutual_info_score(cluster_labels_true, cluster_labels_pred),
        "Homogeneity Score": homogeneity_score(cluster_labels_true, cluster_labels_pred),
        "Completeness Score": completeness_score(cluster_labels_true, cluster_labels_pred),
        "V Measure Score": v_measure_score(cluster_labels_true, cluster_labels_pred),
        "Fowlkes Mallows Score": fowlkes_mallows_score(cluster_labels_true, cluster_labels_pred)
    } if (unique_labels.shape[0] > 1) and (np.unique(cluster_labels_true).shape[0] > 1) else {
        "Rand Score": -1.0,
        "Adjusted Rand Score": -1.0,
        "Normalised Mutual Information Score": -1.0,
        "Adjusted Mutual Information Score": -1.0,
        "Homogeneity Score": -1.0,
        "Completeness Score": -1.0,
        "V Measure Score": -1.0,
        "Fowlkes Mallows Score": -1.0
    }

    return cluster_evals