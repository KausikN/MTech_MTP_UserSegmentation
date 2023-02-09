"""
User Segmentation - Demographic - Cluster - Graph-Based Algorithms

Pipeline Steps:
 - Cluster input features using clustering algorithm and store cluster centers (number of clusters is given as input)
 - Predict label of new features using stored cluster centers
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

from .Utils import *

# Main Classes
# Spectral Clustering
class UserSegmentation_Cluster_Spectral(UserSegmentation_Cluster_Base):
    def __init__(self,
    model=None,

    norm=True,
    random_state=0,

    **params
    ):
        '''
        User Segmentation - Clustering - Spectral Clustering

        Params:
         - model : Pretained model
         - norm : Normalise features
         - random_state : Random State

        '''
        self.model = model
        self.norm = norm
        self.random_state = random_state
        self.__dict__.update(params)
        # Norm Params
        self.norm_params = {}
        # Time Params
        self.time_data = {
            "train": {},
            "predict": {}
        }

    def train(self,
        Fs, Ls, 
        features_info={"demographic": []},
        **params
        ):
        '''
        Train

        Train Spectral clustering on given features and labels.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("Spectral - Train")
        ## Features Info Init
        for k in features_info.keys():
            if len(features_info[k]) == 0: features_info[k] = [{"name": f"Feature_{i}", "type": {"type": "number"}} for i in range(Fs[k].shape[1])]
        self.feature_types = [fi["type"] for fi in features_info["demographic"]]
        self.feature_names = [fi["name"] for fi in features_info["demographic"]]
        ## Data Init
        Fs = np.array(Fs["demographic"])
        Ls = np.array(Ls)
        N_CLASSES = Ls.shape[-1]
        # Get Clusters
        Fs_flat, self.features_flat_pos = Array_Flatten(Fs, self.feature_types)
        self.feature_names_flat = Array_FlatFeaturesNameMap(self.feature_names, self.feature_types, self.features_flat_pos)
        if self.norm: Fs_flat, self.norm_params = Array_Normalize(Fs_flat)
        self.time_data["train"] = Time_Record("Data Preprocess", self.time_data["train"])
        clustering_data = SpectralClustering(
            n_clusters=N_CLASSES, 
            random_state=self.random_state
        ).fit(Fs_flat)
        self.time_data["train"] = Time_Record("Model Training", self.time_data["train"])
        # Record
        self.model = {
            "model": clustering_data,
            "n_classes": N_CLASSES,
            "features": Fs,
            "true_labels": Ls.argmax(axis=-1)
        }
        self.time_data["train"] = Time_Record("", self.time_data["train"], finish=True)

        return self.model
        
    def visualise(self, disable_plots=False):
        '''
        Visualise
        '''
        # Init
        VisData = {
            "figs": {
                "pyplot": {},
                "plotly_chart": {}
            },
            "data": {}
        }
        Plots = {
            "Cluster Map": []
        }
        Data = {}
        # Get Data
        feature_names = self.feature_names_flat
        clustering_data = self.model["model"]
        feature_points, _ = Array_Flatten(self.model["features"], self.feature_types)
        if self.norm: feature_points, _ = Array_Normalize(feature_points, self.norm_params)
        cluster_labels = np.array(clustering_data.labels_)
        unique_labels = np.unique(cluster_labels)
        cluster_centers_unique = np.array([np.mean(feature_points[cluster_labels == i], axis=0) for i in unique_labels])
        # Cluster Map Plot
        if disable_plots:
            fig_map = plt.figure()
        else:
            fig_map = ClusterVis_ClusterMapPlot(feature_points, cluster_labels, cluster_centers_unique, unique_labels, feature_names)["fig"]
        ## Record
        Plots["Cluster Map"].append(fig_map)
        Data["Cluster Evaluations"] = {
            **ClusterEval_Basic(feature_points, cluster_labels, unique_labels),
            **ClusterEval_External(self.model["true_labels"], cluster_labels, unique_labels),
        }
        Data["Cluster Count"] = {
            "N Samples": feature_points.shape[0],
            "N Clusters": cluster_centers_unique.shape[0]
        }
        Data["Cluster Centers"] = pd.DataFrame([
            {feature_names[j]: cluster_centers_unique[i, j] for j in range(cluster_centers_unique.shape[-1])}
            for i in range(cluster_centers_unique.shape[0])
        ])
        Data["Time"] = self.time_data
        ## CleanUp
        plt.close(fig_map)
        # Record
        VisData["figs"]["plotly_chart"]["Cluster Map"] = Plots["Cluster Map"]
        VisData["data"] = Data

        return VisData

    def predict(self,
        Fs, 

        **params
        ):
        '''
        Predict

        Segment users based on Spectral clustered features.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)

        Outputs:
            - Ls : Label Distributions of Users (N_Samples, Label_Dim)
        '''
        # Init
        N_CLASSES = self.model["n_classes"]
        Fs = np.array(Fs["demographic"])
        Ls = np.zeros((Fs.shape[0], N_CLASSES))
        # Get Closest Clusters
        Fs_flat, _ = Array_Flatten(Fs, self.feature_types)
        if self.norm: Fs_flat, _ = Array_Normalize(Fs_flat, self.norm_params)
        AllFs_flat = self.model["features"].reshape((self.model["features"].shape[0], -1))
        if self.norm: AllFs_flat, _ = Array_Normalize(AllFs_flat, self.norm_params)
        cluster_labels = np.array(self.model["model"].labels_)
        unique_labels = np.unique(cluster_labels)
        cluster_centers_unique = np.array([np.mean(AllFs_flat[cluster_labels == i], axis=0) for i in unique_labels])
        Ls_indices = []
        for i in range(Fs_flat.shape[0]):
            Ls_indices.append(unique_labels[np.argmin(np.linalg.norm(Fs_flat[i] - cluster_centers_unique, axis=-1))])
        Ls[np.arange(Ls.shape[0]), Ls_indices] = 1.0

        return Ls

# Main Vars
SEG_FUNCS = {
    "Spectral": {
        "class": UserSegmentation_Cluster_Spectral,
        "params": {
            "model": None,
            "norm": False,
            "random_state": 0
        }
    }
}