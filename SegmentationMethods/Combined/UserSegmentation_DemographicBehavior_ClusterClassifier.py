"""
User Segmentation - Demographic and Behavior Combined - Basic
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import NMF

from .Utils import *

# Main Functions
# KMeans
class UserSegmentation_DB_NMF_DT(UserSegmentation_Cluster_Base):
    def __init__(self,
    model={"nmf_model": None, "classifier_model": None},

    norm=True,
    random_state=0,
    nmf_init="random",
    nmf_solver="cd",
    nmf_beta_loss="frobenius",
    top_n_samples_per_pattern=5,
    decision_tree_criterion="gini",

    **params
    ):
        '''
        User Segmentation - Demographic and Behavior Combined - Non-Negative Matrix Factorization + Decision Tree

        Params:
         - model : Pretained model
         - norm : Normalise Features
         - random_state : Random State
         - nmf_init : Initialization method for NMF ["random", "nndsvd", "nndsvda", "nndsvdar"]
         - nmf_solver : Solver for NMF ["cd", "mu"]
         - nmf_beta_loss : Beta loss for NMF ["frobenius", "kullback-leibler", "itakura-saito"]
         - top_n_samples_per_pattern : Number of top impactful samples to use for each pattern (If > 1, then top n samples used, If [0.0, 1.0] then top 100*n% samples used])
         - decision_tree_criterion : Criterion for Decision Tree ["gini", "entropy"]

        '''
        self.model = dict(model)
        self.norm = norm
        self.random_state = random_state
        self.nmf_init = nmf_init
        self.nmf_solver = nmf_solver
        self.nmf_beta_loss = nmf_beta_loss
        self.top_n_samples_per_pattern = top_n_samples_per_pattern
        self.decision_tree_criterion = decision_tree_criterion
        self.__dict__.update(params)
        # Norm Params
        self.norm_params = {
            "demographic": {},
            "behavior": {}
        }
        # Time Params
        self.time_data = {
            "train": {},
            "predict": {}
        }

    def train(self,
        Fs, Ls, 
        features_info={"demographic": [], "behavior": [], "product": []},
        **params
        ):
        '''
        Train

        Train NMF on given features and labels to find personas.
        Then, use Decision Tree to classify users into these personas.

        Inputs:
            - Fs : Features of Users
                - demographic : (N_Samples, N_Features, Feature_Dim)
                - behavior : (N_Samples, N_Features, Feature_Dim)
            - Ls : Label Distribution of Users (N_Samples, Label_Dim)
            
        Outputs:
            - models : Trained Models
                - nmf_model : NMF Model
                - classifier_model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("NMF-DecisionTree - Train")
        ## Features Info Init
        for k in features_info.keys():
            if len(features_info[k]) == 0: features_info[k] = [{"name": f"Feature_{i}", "type": {"type": "number"}} for i in range(Fs[k].shape[1])]
        self.feature_types = {k: [fi["type"] for fi in features_info[k]] for k in features_info.keys()}
        self.feature_names = {k: [fi["name"] for fi in features_info[k]] for k in features_info.keys()}
        ## Data Init
        Fs = Fs
        Ls = np.array(Ls)
        N_CLASSES = Ls.shape[-1]
        ## Params Init
        self.n_patterns = N_CLASSES
        ## Vars Init
        self.features_flat_pos = {}
        self.feature_names_flat = {}
        # Perform NMF on Behavior Matrix
        ## Compute B and D
        B = Fs["behavior"]
        B, self.features_flat_pos["behavior"] = Array_Flatten(B, self.feature_types["behavior"])
        self.feature_names_flat["behavior"] = Array_FlatFeaturesNameMap(self.feature_names["behavior"], self.feature_types["behavior"], self.features_flat_pos["behavior"])
        ### NORMALISE BEHAVIOR FEATURES
        if self.norm: B, self.norm_params["behavior"] = Array_Normalize(B)
        D_Flat, self.features_flat_pos["demographic"] = Array_Flatten(Fs["demographic"], self.feature_types["demographic"])
        self.feature_names_flat["demographic"] = Array_FlatFeaturesNameMap(self.feature_names["demographic"], self.feature_types["demographic"], self.features_flat_pos["demographic"])
        ### DO NOT NORMALISE DEMOGRAPHIC FEATURES
        # if self.norm: D_Flat, self.norm_params["demographic"] = Array_Normalize(D_Flat)
        ## Get W and H
        self.time_data["train"] = Time_Record("Data Preprocess", self.time_data["train"])
        model_data = NMF(
            n_components=self.n_patterns, 
            init=self.nmf_init,
            solver=self.nmf_solver,
            beta_loss=self.nmf_beta_loss,
            max_iter=1000,
            random_state=self.random_state
        ).fit(B)
        self.time_data["train"] = Time_Record("NMF", self.time_data["train"])
        W = np.array(model_data.transform(B))
        H = np.array(model_data.components_)
        ## Get most impactful samples for each pattern
        top_impact_samples = []
        TOPN = int(self.top_n_samples_per_pattern) if int(self.top_n_samples_per_pattern) > 1.0 \
            else int(self.top_n_samples_per_pattern * B.shape[0])
        for i in range(self.n_patterns):
            top_impact_samples.append(np.argsort(W[:, i])[::-1][:TOPN])
        ## Form Personas for each pattern
        personas = []
        for i in range(self.n_patterns):
            persona = {
                "name": f"Persona {i}",
                "behavior": H[i],
                "demographic": np.mean(D_Flat[top_impact_samples[i]], axis=0)
            }
            personas.append(persona)
        self.time_data["train"] = Time_Record("Persona Generation", self.time_data["train"])
        ## Assign each user to a persona based on closest demographic distance
        Ls_indices = np.zeros((Ls.shape[0],))
        DistanceMatrix = np.zeros((Ls.shape[0], self.n_patterns))
        for i in range(Ls.shape[0]):
            for j in range(self.n_patterns):
                DistanceMatrix[i, j] = np.linalg.norm(D_Flat[i] - personas[j]["demographic"])
            Ls_indices[i] = np.argmin(DistanceMatrix[i])
        self.time_data["train"] = Time_Record("Clustering", self.time_data["train"])
        ## Record
        self.model["nmf_model"] = {
            "model": model_data,
            "W": W,
            "H": H,
            "top_impact_samples": top_impact_samples,
            "personas": personas,
            "n_classes": N_CLASSES,
            "features": Fs,
            "true_labels": Ls.argmax(axis=-1),
            "labels_pred": Ls_indices
        }
        # Train Classifier
        classifier_data = DecisionTreeClassifier(
            criterion=self.decision_tree_criterion,
            random_state=self.random_state
        ).fit(D_Flat, Ls_indices)
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
        feature_points, _ = Array_Flatten(self.model["nmf_model"]["features"]["demographic"], self.feature_types["demographic"])
        ## DO NOT NORMALISE DEMOGRAPHIC FEATURES
        # if self.norm: feature_points, _ = Array_Normalize(feature_points, self.norm_params["demographic"])
        cluster_labels = self.model["nmf_model"]["labels_pred"]
        unique_labels = np.unique(cluster_labels)
        cluster_centers_unique = np.array([p["demographic"] for p in self.model["nmf_model"]["personas"]])
        # Cluster Map Plot
        if disable_plots:
            fig_map = plt.figure()
        else:
            fig_map = ClusterVis_ClusterMapPlot(feature_points, cluster_labels, cluster_centers_unique, unique_labels, feature_names["demographic"])["fig"]
        ## Record
        Plots["Cluster Map"].append(fig_map)
        Data["Cluster Evaluations"] = {
            **ClusterEval_Basic(feature_points, cluster_labels, unique_labels),
            **ClusterEval_External(self.model["nmf_model"]["true_labels"], cluster_labels, unique_labels),
        }
        Data["Cluster Count"] = {
            "N Samples": feature_points.shape[0],
            "N Clusters": cluster_centers_unique.shape[0]
        }
        Data["Cluster Centers"] = pd.DataFrame([
            {feature_names["demographic"][j]: cluster_centers_unique[i, j] for j in range(cluster_centers_unique.shape[-1])}
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
        N_CLASSES = self.model["nmf_model"]["n_classes"]
        Fs = np.array(Fs["demographic"])
        Ls = np.zeros((Fs.shape[0], N_CLASSES))
        # Get Closest Clusters
        Fs_flat, _ = Array_Flatten(Fs, self.feature_types["demographic"])
        if self.norm: Fs_flat, _ = Array_Normalize(Fs_flat, self.norm_params["demographic"])
        Ls_indices = np.array(self.model["classifier_model"]["model"].predict(Fs_flat), dtype=int)
        Ls[np.arange(Ls.shape[0]), Ls_indices] = 1.0

        return Ls

# Main Vars
SEG_FUNCS = {
    "NMF + Decision Tree": {
        "class": UserSegmentation_DB_NMF_DT,
        "params": {
            "model": {
                "nmf_model": None,
                "classifier_model": None
            },
            "norm": False,
            "random_state": 0,
            "nmf_init": "random",
            "nmf_solver": "cd",
            "nmf_beta_loss": "frobenius",
            "top_n_samples_per_pattern": 5,
            "decision_tree_criterion": "gini"
        }
    }
}