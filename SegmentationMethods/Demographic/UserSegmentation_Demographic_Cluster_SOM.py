"""
User Segmentation - Demographic - Cluster - Self Organizing Maps

Pipeline Steps:
 - Cluster input features using clustering algorithm and store cluster centers (number of clusters is given as input)
 - Predict label of new features using stored cluster centers
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.cluster import KMeans

from .Utils import *

# Main Classes
# SOM
class UserSegmentation_Cluster_SOM(UserSegmentation_Cluster_Base):
    def __init__(self,
    model=None,

    norm=True,
    random_state=0,
    som_dim=(10, 10),
    som_iterations=100,
    som_sigma=1.0,
    som_learning_rate=0.5,
    som_neighborhood_function="gaussian",
    som_activation_distance="euclidean",
    som_verbose=False,

    **params
    ):
        '''
        User Segmentation - Clustering - SOM

        Params:
         - model : Pretained model
         - norm : Normalize features
         - random_state : Random State
         - som_dim : Dimension of SOM
         - som_iterations : Number of iterations for SOM
         - som_sigma : Sigma of SOM
         - som_learning_rate : Learning Rate of SOM
         - som_neighborhood_function : Neighborhood Function of SOM ["gaussian", "mexican_hat", "bubble", "triangle"]
         - som_activation_distance : Activation Distance of SOM ["euclidean", "cosine", "manhattan", "chebyshev"]
         - som_verbose : Whether to print SOM training progress

        '''
        self.model = model
        self.norm = norm
        self.random_state = random_state
        self.som_dim = som_dim
        self.som_iterations = som_iterations
        self.som_sigma = som_sigma
        self.som_learning_rate = som_learning_rate
        self.som_neighborhood_function = som_neighborhood_function
        self.som_activation_distance = som_activation_distance
        self.som_verbose = som_verbose
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

        Train SOM clustering on given features and labels.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("SOM - Train")
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
        ## Train SOM
        som = MiniSom(
            x=self.som_dim[0], y=self.som_dim[1], 
            input_len=Fs_flat.shape[-1],
            sigma=self.som_sigma, learning_rate=self.som_learning_rate,
            neighborhood_function=self.som_neighborhood_function, activation_distance=self.som_activation_distance,
            random_seed=self.random_state
        )
        som.random_weights_init(Fs_flat)
        som.train_random(Fs_flat, self.som_iterations, verbose=self.som_verbose)
        som_labels = np.array([som.winner(x) for x in Fs_flat])
        ## Apply K-Means to reduce clusters
        ### OLD METHOD
        # som_clusters = som.win_map(Fs_flat)
        # som_clusters_points = np.array([[int(k[0]), int(k[1])] for k in som_clusters.keys()])
        ### OLD METHOD
        som_clusters_points = som.get_weights().reshape(-1, Fs_flat.shape[-1])
        kmeans_data = KMeans(
            n_clusters=N_CLASSES, 
            random_state=self.random_state
        ).fit(som_clusters_points)
        ### OLD METHOD
        # labels_ = np.array(kmeans_data.predict(som_labels))
        ### OLD METHOD
        self.label_map = np.reshape(kmeans_data.labels_, self.som_dim)
        labels_ = np.array([self.label_map[sl[0], sl[1]] for sl in som_labels])
        self.time_data["train"] = Time_Record("Model Training", self.time_data["train"])
        # Record
        self.model = {
            "model": {
                "som": som,
                "kmeans": kmeans_data
            },
            "n_classes": N_CLASSES,
            "features": Fs,
            "true_labels": Ls.argmax(axis=-1),
            "labels": labels_
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
            "Cluster Map": [],
            "SOM Distance Map": []
        }
        Data = {}
        # Get Data
        feature_names = self.feature_names_flat
        clustering_data = self.model["model"]
        feature_points, _ = Array_Flatten(self.model["features"], self.feature_types)
        if self.norm: feature_points, _ = Array_Normalize(feature_points, self.norm_params)
        cluster_labels = self.model["labels"]
        unique_labels = np.unique(cluster_labels)
        ### OLD METHOD
        # cluster_centers_unique_kmeans = np.array(np.round(clustering_data["kmeans"].cluster_centers_), dtype=int)
        # som_weights = np.array(clustering_data["som"].get_weights())
        # cluster_centers_unique = np.array([som_weights[c[0], c[1]] for c in cluster_centers_unique_kmeans])
        ### OLD METHOD
        cluster_centers_unique = np.array(clustering_data["kmeans"].cluster_centers_)
        # Cluster Map Plot
        if disable_plots:
            fig_map = plt.figure()
        else:
            fig_map = ClusterVis_ClusterMapPlot(feature_points, cluster_labels, cluster_centers_unique, unique_labels, feature_names)["fig"]
        # SOM Distance Map Plot
        if disable_plots:
            fig_som = plt.figure()
        else:
            fig_som = plt.figure()
            plt.imshow(clustering_data["som"].distance_map().T, cmap="bone_r")
            plt.colorbar()
            plt.title("SOM Distance Map")
        ## Record
        Plots["Cluster Map"].append(fig_map)
        Plots["SOM Distance Map"].append(fig_som)
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
        plt.close(fig_som)
        # Record
        VisData["figs"]["plotly_chart"]["Cluster Map"] = Plots["Cluster Map"]
        VisData["figs"]["pyplot"]["SOM Distance Map"] = Plots["SOM Distance Map"]
        VisData["data"] = Data

        return VisData

    def predict(self,
        Fs, 

        **params
        ):
        '''
        Predict

        Segment users based on SOM clustered features.

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
        som_labels = np.array([self.model["model"]["som"].winner(x) for x in Fs_flat])
        Ls_indices = np.array([self.label_map[sl[0], sl[1]] for sl in som_labels])
        Ls[np.arange(Ls.shape[0]), Ls_indices] = 1.0

        return Ls

# Main Vars
SEG_FUNCS = {
    "SOM": {
        "class": UserSegmentation_Cluster_SOM,
        "params": {
            "model": None,
            "norm": False,
            "random_state": 0,
            "som_dim": (10, 10),
            "som_iterations": 100,
            "som_sigma": 1.0,
            "som_learning_rate": 0.5,
            "som_neighborhood_function": "gaussian",
            "som_activation_distance": "euclidean",
            "som_verbose": False
        }
    }
}