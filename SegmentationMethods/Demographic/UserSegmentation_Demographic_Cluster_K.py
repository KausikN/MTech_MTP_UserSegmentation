"""
User Segmentation - Demographic - Cluster - K Algorithms

Pipeline Steps:
 - Cluster input features using clustering algorithm and store cluster centers (number of clusters is given as input)
 - Predict label of new features using stored cluster centers
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import kmeans_plusplus

from .Utils import *

# Main Classes
# KMeans++
class UserSegmentation_Cluster_KMeans(UserSegmentation_Cluster_Base):
    def __init__(self,
    model=None,

    norm=True,
    random_state=0,

    **params
    ):
        '''
        User Segmentation - Clustering - KMeans

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

        Train KMeans clustering on given features and labels.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("K-Means - Train")
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
        clustering_data = KMeans(
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
        cluster_centers_unique = np.array(clustering_data.cluster_centers_[unique_labels])
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

        Segment users based on KMeans clustered features.

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

# Minibatch KMeans++
class UserSegmentation_Cluster_MiniBatchKMeans(UserSegmentation_Cluster_Base):
    def __init__(self,
    model=None,

    norm=True,
    batch_size=1024,
    random_state=0,

    **params
    ):
        '''
        User Segmentation - Clustering - MiniBatch KMeans

        Params:
         - model : Pretained model
         - norm : Normalise features
         - batch_size : Batch Size
         - random_state : Random State

        '''
        self.model = model
        self.norm = norm
        self.batch_size = batch_size
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

        Train MiniBatch KMeans clustering on given features and labels.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("Mini-Batch K-Means - Train")
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
        clustering_data = MiniBatchKMeans(
            n_clusters=N_CLASSES, 
            batch_size=self.batch_size,
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
        cluster_centers_unique = np.array(clustering_data.cluster_centers_[unique_labels])
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

        Segment users based on MiniBatch KMeans clustered features.

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

# KModes
class UserSegmentation_Cluster_KModes(UserSegmentation_Cluster_Base):
    def __init__(self,
    model=None,

    random_state=0,

    **params
    ):
        '''
        User Segmentation - Clustering - KModes

        Params:
         - model : Pretained model
         - random_state : Random State

        '''
        self.model = model
        self.random_state = random_state
        self.__dict__.update(params)
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

        Train KModes clustering on given features and labels.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("K-Modes - Train")
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
        Fs_flat, _ = Array_OneHot2Index(Fs, self.feature_types)
        self.feature_names_flat = self.feature_names
        self.time_data["train"] = Time_Record("Data Preprocess", self.time_data["train"])
        clustering_data = KModes(
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
        feature_points, _ = Array_OneHot2Index(self.model["features"], self.feature_types)
        cluster_labels = np.array(clustering_data.labels_)
        unique_labels = np.unique(cluster_labels)
        cluster_centers_unique = np.array(clustering_data.cluster_centroids_[unique_labels])
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

        Segment users based on KModes clustered features.

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
        Fs_flat, _ = Array_OneHot2Index(Fs, self.feature_types)
        Ls_indices = np.array(self.model["model"].predict(Fs_flat))
        Ls[np.arange(Ls.shape[0]), Ls_indices] = 1.0

        return Ls

# KPrototypes
class UserSegmentation_Cluster_KPrototypes(UserSegmentation_Cluster_Base):
    def __init__(self,
    model=None,

    norm=True,
    random_state=0,

    **params
    ):
        '''
        User Segmentation - Clustering - KPrototypes

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

        Train KPrototypes clustering on given features and labels.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("K-Prototypes - Train")
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
        Fs_flat, self.category_column_indices = Array_OneHot2Index(Fs, self.feature_types)
        self.numerical_column_indices = np.array([i for i in range(Fs_flat.shape[-1]) if i not in self.category_column_indices])
        self.feature_names_flat = self.feature_names
        if self.norm:
            Fs_flat_numerical = Fs_flat[:, self.numerical_column_indices]
            Fs_flat_numerical, self.norm_params = Array_Normalize(Fs_flat_numerical)
            Fs_flat[:, self.numerical_column_indices] = Fs_flat_numerical
        self.time_data["train"] = Time_Record("Data Preprocess", self.time_data["train"])
        clustering_data = KPrototypes(
            n_clusters=N_CLASSES, 
            random_state=self.random_state
        ).fit(Fs_flat, categorical=self.category_column_indices)
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
        feature_points, _ = Array_OneHot2Index(self.model["features"], self.feature_types)
        if self.norm:
            feature_points_numerical = feature_points[:, self.numerical_column_indices]
            feature_points_numerical, _ = Array_Normalize(feature_points_numerical, self.norm_params)
            feature_points[:, self.numerical_column_indices] = feature_points_numerical
        cluster_labels = np.array(clustering_data.labels_)
        unique_labels = np.unique(cluster_labels)
        cluster_centers_unique = np.array(clustering_data.cluster_centroids_[unique_labels])
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

        Segment users based on KPrototypes clustered features.

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
        Fs_flat, _ = Array_OneHot2Index(Fs, self.feature_types)
        if self.norm:
            Fs_flat_numerical = Fs_flat[:, self.numerical_column_indices]
            Fs_flat_numerical, self.norm_params = Array_Normalize(Fs_flat_numerical)
            Fs_flat[:, self.numerical_column_indices] = Fs_flat_numerical
        Ls_indices = np.array(self.model["model"].predict(Fs_flat, categorical=self.category_column_indices))
        Ls[np.arange(Ls.shape[0]), Ls_indices] = 1.0

        return Ls

# Main Vars
SEG_FUNCS = {
    "KMeans++": {
        "class": UserSegmentation_Cluster_KMeans,
        "params": {
            "model": None,
            "norm": False,
            "random_state": 0
        }
    },
    "MiniBatchKMeans++": {
        "class": UserSegmentation_Cluster_MiniBatchKMeans,
        "params": {
            "model": None,
            "norm": False,
            "batch_size": 1024,
            "random_state": 0
        }
    },
    "KModes": {
        "class": UserSegmentation_Cluster_KModes,
        "params": {
            "model": None,
            "random_state": 0
        }
    },
    "KPrototypes": {
        "class": UserSegmentation_Cluster_KPrototypes,
        "params": {
            "model": None,
            "norm": False,
            "random_state": 0
        }
    }
}