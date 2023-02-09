"""
User Segmentation - Demographic - Clustering - Density-Based Algorithms

Pipeline Steps:
 - Cluster input features using clustering algorithm and store cluster centers (number of clusters is not given)
 - Predict label of new features using stored cluster centers
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS, MeanShift, AffinityPropagation

from .Utils import *

# Main Functions
# DBSCAN
class UserSegmentation_Cluster_DBSCAN(UserSegmentation_Cluster_Base):
    def __init__(self,
    model=None,

    norm=True,
    eps=0.5,
    min_samples=5,

    **params
    ):
        '''
        User Segmentation - Clustering - DBSCAN

        Params:
         - model : Pretained model
         - norm : Normalise features
         - eps : The maximum distance between two samples for one to be considered as in the neighborhood of the other
         - min_samples : Number of samples in a neighborhood for a point to be considered as a core point (including itself)
        '''
        self.model = model
        self.norm = norm
        self.eps = eps
        self.min_samples = min_samples
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

        Train DBSCAN clustering on given features and labels.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("DBSCAN - Train")
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
        clustering_data = DBSCAN(
            eps=self.eps, min_samples=self.min_samples
        ).fit(Fs_flat)
        N_CLASSES = np.unique(clustering_data.labels_).shape[0]
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
            "Channel": [],
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
        # Cluster Evaluation
        cluster_mask = cluster_labels != -1
        n_cluster_pts = np.sum(cluster_mask)
        cluster_eval_data = {
            **ClusterEval_Basic(feature_points[cluster_mask], cluster_labels[cluster_mask], unique_labels),
            # **ClusterEval_Basic(feature_points, cluster_labels, unique_labels),
            **ClusterEval_External(self.model["true_labels"], cluster_labels, unique_labels),
        }
        ## Record
        Plots["Cluster Map"].append(fig_map)
        Data["Cluster Evaluations"] = cluster_eval_data
        Data["Cluster Count"] = {
            "N Samples": feature_points.shape[0],
            "N Clusters": cluster_centers_unique.shape[0],
            "N Cluster Points": int(n_cluster_pts),
            "N Outliers": int(cluster_mask.shape[0] - n_cluster_pts)
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

        Segment users based on DBSCAN clustered features.

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

# OPTICS
class UserSegmentation_Cluster_OPTICS(UserSegmentation_Cluster_Base):
    def __init__(self,
    model=None,

    norm=True,
    metric="euclidean",
    min_samples=5,

    **params
    ):
        '''
        User Segmentation - Clustering - OPTICS

        Params:
         - model : Pretained model
         - norm : Normalise features
         - metric : Distance metric ["minkowski", "euclidean", "manhattan", "chebyshev", "canberra", "braycurtis", "mahalanobis", "wminkowski", "seuclidean", "cosine", "correlation", "hamming", "jaccard", "dice", "kulsinski", "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath", "yule"]
         - min_samples : Number of samples in a neighborhood for a point to be considered as a core point (including itself)
        '''
        self.model = model
        self.norm = norm
        self.metric = metric
        self.min_samples = min_samples
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

        Train OPTICS clustering on given features and labels.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("OPTICS - Train")
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
        clustering_data = OPTICS(
            min_samples=self.min_samples, metric=self.metric
        ).fit(Fs_flat)
        N_CLASSES = np.unique(clustering_data.labels_).shape[0]
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
            "Channel": [],
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
        # Cluster Evaluation
        cluster_mask = cluster_labels != -1
        n_cluster_pts = np.sum(cluster_mask)
        cluster_eval_data = {
            **ClusterEval_Basic(feature_points[cluster_mask], cluster_labels[cluster_mask], unique_labels),
            # **ClusterEval_Basic(feature_points, cluster_labels, unique_labels),
            **ClusterEval_External(self.model["true_labels"], cluster_labels, unique_labels),
        }
        ## Record
        Plots["Cluster Map"].append(fig_map)
        Data["Cluster Evaluations"] = cluster_eval_data
        Data["Cluster Count"] = {
            "N Samples": feature_points.shape[0],
            "N Clusters": cluster_centers_unique.shape[0],
            "N Cluster Points": int(n_cluster_pts),
            "N Outliers": int(cluster_mask.shape[0] - n_cluster_pts)
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

        Segment users based on OPTICS clustered features.

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

# MeanShift
class UserSegmentation_Cluster_MeanShift(UserSegmentation_Cluster_Base):
    def __init__(self,
    model=None,

    norm=True,
    max_iter=300,

    **params
    ):
        '''
        User Segmentation - Clustering - MeanShift

        Params:
         - model : Pretained model
         - norm : Normalise features
         - max_iter : Maximum number of iterations
        '''
        self.model = model
        self.norm = norm
        self.max_iter = max_iter
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

        Train MeanShift clustering on given features and labels.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("MeanShift - Train")
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
        clustering_data = MeanShift(
            bandwidth=None, max_iter=self.max_iter
        ).fit(Fs_flat)
        N_CLASSES = np.unique(clustering_data.labels_).shape[0]
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
            "Channel": [],
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
        cluster_centers_unique = np.array(clustering_data.cluster_centers_[unique_labels])
        # Cluster Map Plot
        if disable_plots:
            fig_map = plt.figure()
        else:
            fig_map = ClusterVis_ClusterMapPlot(feature_points, cluster_labels, cluster_centers_unique, unique_labels, feature_names)["fig"]
        # Cluster Evaluation
        cluster_mask = cluster_labels != -1
        n_cluster_pts = np.sum(cluster_mask)
        cluster_eval_data = {
            **ClusterEval_Basic(feature_points[cluster_mask], cluster_labels[cluster_mask], unique_labels),
            # **ClusterEval_Basic(feature_points, cluster_labels, unique_labels),
            **ClusterEval_External(self.model["true_labels"], cluster_labels, unique_labels),
        }
        ## Record
        Plots["Cluster Map"].append(fig_map)
        Data["Cluster Evaluations"] = cluster_eval_data
        Data["Cluster Count"] = {
            "N Samples": feature_points.shape[0],
            "N Clusters": cluster_centers_unique.shape[0],
            "N Cluster Points": int(n_cluster_pts),
            "N Outliers": int(cluster_mask.shape[0] - n_cluster_pts)
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

        Segment users based on MeanShift clustered features.

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
        Ls_indices = np.array(self.model["model"].predict(Fs_flat))
        Ls[np.arange(Ls.shape[0]), Ls_indices] = 1.0

        return Ls

# Affinity Propagation
class UserSegmentation_Cluster_AffinityPropagation(UserSegmentation_Cluster_Base):
    def __init__(self,
    model=None,

    norm=True,
    damping=0.5,
    max_iter=200,
    random_state=0,

    **params
    ):
        '''
        User Segmentation - Clustering - AffinityPropagation

        Params:
         - model : Pretained model
         - norm : Normalise features
         - damping : Damping factor between 0.5 and 1
         - max_iter : Maximum number of iterations
         - random_state : Random state
        '''
        self.model = model
        self.norm = norm
        self.damping = damping
        self.max_iter = max_iter
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

        Train AffinityPropagation clustering on given features and labels.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("AffinityPropagation - Train")
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
        clustering_data = AffinityPropagation(
            damping=self.damping, max_iter=self.max_iter, random_state=self.random_state
        ).fit(Fs_flat)
        N_CLASSES = np.unique(clustering_data.labels_).shape[0]
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
            "Channel": [],
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
        cluster_centers_unique = np.array(clustering_data.cluster_centers_[unique_labels])
        # Cluster Map Plot
        if disable_plots:
            fig_map = plt.figure()
        else:
            fig_map = ClusterVis_ClusterMapPlot(feature_points, cluster_labels, cluster_centers_unique, unique_labels, feature_names)["fig"]
        # Cluster Evaluation
        cluster_mask = cluster_labels != -1
        n_cluster_pts = np.sum(cluster_mask)
        cluster_eval_data = {
            **ClusterEval_Basic(feature_points[cluster_mask], cluster_labels[cluster_mask], unique_labels),
            # **ClusterEval_Basic(feature_points, cluster_labels, unique_labels),
            **ClusterEval_External(self.model["true_labels"], cluster_labels, unique_labels),
        }
        ## Record
        Plots["Cluster Map"].append(fig_map)
        Data["Cluster Evaluations"] = cluster_eval_data
        Data["Cluster Count"] = {
            "N Samples": feature_points.shape[0],
            "N Clusters": cluster_centers_unique.shape[0],
            "N Cluster Points": int(n_cluster_pts),
            "N Outliers": int(cluster_mask.shape[0] - n_cluster_pts)
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

        Segment users based on AffinityPropagation clustered features.

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
        Ls_indices = np.array(self.model["model"].predict(Fs_flat))
        Ls[np.arange(Ls.shape[0]), Ls_indices] = 1.0

        return Ls

# Main Vars
SEG_FUNCS = {
    "DBSCAN": {
        "class": UserSegmentation_Cluster_DBSCAN,
        "params": {
            "model": None,
            "norm": False,
            "eps": 0.5,
            "min_samples": 5
        }
    },
    "OPTICS": {
        "class": UserSegmentation_Cluster_OPTICS,
        "params": {
            "model": None,
            "norm": False,
            "min_samples": 5,
            "metric": "euclidean"
        }
    },
    "MeanShift": {
        "class": UserSegmentation_Cluster_MeanShift,
        "params": {
            "model": None,
            "norm": False,
            "max_iter": 300
        }
    },
    "AffinityPropagation": {
        "class": UserSegmentation_Cluster_AffinityPropagation,
        "params": {
            "model": None,
            "norm": False,
            "damping": 0.5,
            "max_iter": 200,
            "random_state": 0
        }
    }
}