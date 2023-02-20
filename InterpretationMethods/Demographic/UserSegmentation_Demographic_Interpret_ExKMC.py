"""
User Segmentation - Demographic - Interpret - ExKMC Algorithms

Pipeline Steps:
 - Train Classifier on Features with assigned Cluster Labels
 - Use Classifier to predict Cluster Labels for each User
 - Intepret Classifier to get Information for each Cluster
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ExKMC import Tree as ExKMC_Tree

from .Utils import *

# Main Classes
# ExKMC
class UserSegmentation_Interpret_ExKMC(UserSegmentation_Interpret_Base):
    def __init__(self,
    clustering_algorithm=None,

    exkmc_max_leaves=None,
    random_state=0,

    **params
    ):
        '''
        User Segmentation - Interpret - ExKMC

        Params:
         - clustering_algorithm : Clustering Algorithm Object used to get Cluster Labels
         - exkmc_max_leaves : Maximum number of leaves in the tree (If None, IMM will be used, else ExKMC)
         - random_state : Random State

        '''
        self.clustering_algorithm = clustering_algorithm
        self.exkmc_max_leaves = exkmc_max_leaves
        self.random_state = random_state
        self.__dict__.update(params)
        # Time Params
        self.time_data = {
            "train": {},
            "predict": {}
        }

    def train(self,
        **params
        ):
        '''
        Train

        Train ExKMC on given features and labels from clustering algorithm.

        Inputs: None
            
        Outputs:
            - interpretation_model : Model used to interpret clusters
        '''
        # Init
        self.time_data["train"] = Time_Record("ExKMC - Train")
        ## Features Info Init
        self.feature_types = self.clustering_algorithm.feature_types
        self.feature_names = self.clustering_algorithm.feature_names
        ## Data Init
        Fs = self.clustering_algorithm.model["features"]
        Ls = self.clustering_algorithm.predict({"demographic": Fs})
        N_CLASSES = Ls.shape[-1]
        # Train Classifier
        Fs_flat, self.features_flat_pos = Array_Flatten(Fs, self.feature_types)
        self.feature_names_flat = Array_FlatFeaturesNameMap(self.feature_names, self.feature_types, self.features_flat_pos)
        self.time_data["train"] = Time_Record("Data Preprocess", self.time_data["train"])
        
        interpreter_model = ExKMC_Tree.Tree(
            k=N_CLASSES,
            max_leaves=self.exkmc_max_leaves,
            random_state=self.random_state
        ).fit(Fs_flat, self.clustering_algorithm.model["model"])
        self.time_data["train"] = Time_Record("Model Training", self.time_data["train"])
        # Record
        self.model = {
            "model": interpreter_model,
            "features_flat": Fs_flat,
            "true_labels": Ls,
            "predicted_labels": interpreter_model.predict(Fs_flat),
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
            "Tree": []
        }
        Data = {}
        # Get Data
        interpreter_model = self.model["model"]
        Fs_flat = self.model["features_flat"]
        # SHAP Plot
        if disable_plots:
            fig_1 = plt.figure()
        else:
            save_path = "_evaluations/temp"
            interpreter_model.plot(filename=save_path, feature_names=self.feature_names_flat)
            fig_1 = plt.figure()
            fig_I = plt.imread(save_path + ".gv.png")
            plt.imshow(fig_I)
        ## Record
        Plots["Tree"].append(fig_1)
        Data["Classifier Evaluations"] = {
            **ClassifierEval_Basic(self.model["true_labels"], self.model["predicted_labels"]),
        }
        Data["Time"] = self.time_data
        ## CleanUp
        plt.close(fig_1)
        # Record
        VisData["figs"]["pyplot"]["Tree"] = Plots["Tree"]
        VisData["data"] = Data

        return VisData

    def predict(self,
        Fs, 

        **params
        ):
        '''
        Predict

        Segment users using model.

        Inputs:
            - Fs : Features of Users (N_Samples, N_Features, Feature_Dim)

        Outputs:
            - Ls : Label Distributions of Users (N_Samples, Label_Dim)
        '''
        # Init
        N_CLASSES = self.model["n_classes"]
        Fs = np.array(Fs["demographic"])
        Ls = None
        # Predict
        Fs_flat, _ = Array_Flatten(Fs, self.feature_types)
        Ls = self.model["model"].predict(Fs_flat)

        return Ls

# Main Vars
INTERPRET_FUNCS = {
    "ExKMC": {
        "class": UserSegmentation_Interpret_ExKMC,
        "params": {
            "clustering_algorithm": None,
            "exkmc_max_leaves": None,
        }
    }
}