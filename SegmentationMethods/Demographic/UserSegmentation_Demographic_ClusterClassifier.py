"""
User Segmentation - Demographic - Cluster-Classifier

Pipeline Steps:
 - Cluster input features using clustering algorithm
 - Train a classifier on clustered features (label is the cluster index)
 - Predict label of new features using trained classifier
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

from . import UserSegmentation_Demographic_Cluster_K

from .Utils import *

# Main Functions
# KMeans
class UserSegmentation_Combined_KMeans_DT(UserSegmentation_Cluster_Base):
    def __init__(self,
    model={"cluster_model": None, "classifier_model": None},

    norm=True,
    random_state=0,
    decision_tree_criterion="gini",
    cluster_algo_params=UserSegmentation_Demographic_Cluster_K.SEG_FUNCS["KMeans++"]["params"],

    **params
    ):
        '''
        User Segmentation - Combined - KMeans++ + Decision Tree

        Params:
         - model : Pretained model
         - norm : Normalise features
         - random_state : Random State
         - decision_tree_criterion : Criterion for Decision Tree ["gini", "entropy"]
         - cluster_algo_params : Parameters for Clustering Algorithm

        '''
        self.model = model
        self.norm = norm
        self.random_state = random_state
        self.decision_tree_criterion = decision_tree_criterion
        self.cluster_algo_params = cluster_algo_params
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

        Perform KMeans++ clustering on given features and labels.
        Then, train a Decision Tree classifier on the clustered features.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - model : Trained Models
                - cluster_model : KMeans Model
                - classifier_model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("KMeans-DecisionTree - Train")
        ## Features Info Init
        for k in features_info.keys():
            if len(features_info[k]) == 0: features_info[k] = [{"name": f"Feature_{i}", "type": {"type": "number"}} for i in range(Fs[k].shape[1])]
        self.feature_types = [fi["type"] for fi in features_info["demographic"]]
        self.feature_names = [fi["name"] for fi in features_info["demographic"]]
        ## Data Init
        Fs_INPUT = Fs
        Ls_INPUT = Ls
        Fs = np.array(Fs["demographic"])
        Ls = np.array(Ls)
        N_CLASSES = Ls.shape[-1]
        # Get Clusters
        Fs_flat, self.features_flat_pos = Array_Flatten(Fs, self.feature_types)
        self.feature_names_flat = Array_FlatFeaturesNameMap(self.feature_names, self.feature_types, self.features_flat_pos)
        if self.norm: Fs_flat, self.norm_params = Array_Normalize(Fs_flat)
        self.time_data["train"] = Time_Record("Data Preprocess", self.time_data["train"])
        clustering_model = UserSegmentation_Demographic_Cluster_K.SEG_FUNCS["KMeans++"]["class"](**self.cluster_algo_params)
        clustering_model.train(Fs_INPUT, Ls_INPUT, features_info)
        clustering_data = clustering_model.model["model"]
        self.time_data["train"] = Time_Record("Cluster Model Training", self.time_data["train"])
        cluster_labels = np.array(clustering_data.labels_)
        ## Record
        self.model["cluster_model"] = {
            "model": clustering_data,
            "n_classes": N_CLASSES,
            "features": Fs,
            "true_labels": Ls.argmax(axis=-1)
        }
        # Train Classifier
        classifier_data = DecisionTreeClassifier(
            criterion=self.decision_tree_criterion,
            random_state=self.random_state
        ).fit(Fs_flat, cluster_labels)
        self.time_data["train"] = Time_Record("Classifier Model Training", self.time_data["train"])
        ## Record
        self.model["classifier_model"] = {
            "model": classifier_data
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
        clustering_data = self.model["cluster_model"]["model"]
        feature_points, _ = Array_Flatten(self.model["cluster_model"]["features"], self.feature_types)
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
            **ClusterEval_External(self.model["cluster_model"]["true_labels"], cluster_labels, unique_labels),
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
        N_CLASSES = self.model["cluster_model"]["n_classes"]
        Fs = np.array(Fs["demographic"])
        Ls = np.zeros((Fs.shape[0], N_CLASSES))
        # Get Closest Clusters
        Fs_flat, _ = Array_Flatten(Fs, self.feature_types)
        if self.norm: Fs_flat, _ = Array_Normalize(Fs_flat, self.norm_params)
        Ls_indices = np.array(self.model["classifier_model"]["model"].predict(Fs_flat))
        Ls[np.arange(Ls.shape[0]), Ls_indices] = 1.0

        return Ls

# Main Vars
SEG_FUNCS = {
    "KMeans + Decision Tree": {
        "class": UserSegmentation_Combined_KMeans_DT,
        "params": {
            "model": {
                "cluster_model": None,
                "classifier_model": None
            },
            "norm": False,
            "random_state": 0,
            "decision_tree_criterion": "gini",
            "cluster_algo_params": {
                **UserSegmentation_Demographic_Cluster_K.SEG_FUNCS["KMeans++"]["params"]
            }
        }
    }
}