"""
User Segmentation - Demographic - Interpret - Classifier-Based Algorithms

Pipeline Steps:
 - Train Classifier on Features with assigned Cluster Labels
 - Use Classifier to predict Cluster Labels for each User
 - Intepret Classifier to get Information for each Cluster
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import shap

from .Utils import *

# Main Classes
# Decision Tree Classifier
class UserSegmentation_Interpret_DT(UserSegmentation_Interpret_Base):
    def __init__(self,
    clustering_algorithm=None,

    decision_tree_criterion="gini",
    random_state=0,

    **params
    ):
        '''
        User Segmentation - Interpret - Decision Tree Classifier

        Params:
         - clustering_algorithm : Clustering Algorithm Object used to get Cluster Labels
         - decision_tree_criterion : Criterion for Decision Tree ["gini", "entropy"]
         - random_state : Random State

        '''
        self.clustering_algorithm = clustering_algorithm
        self.decision_tree_criterion = decision_tree_criterion
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

        Train Decision Tree Classifier on given features and labels from clustering algorithm.

        Inputs: None
            
        Outputs:
            - interpretation_model : Model used to interpret clusters
        '''
        # Init
        self.time_data["train"] = Time_Record("Decision Tree Classifier - Train")
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
        classifier_model = DecisionTreeClassifier(
            criterion=self.decision_tree_criterion,
            random_state=self.random_state
        ).fit(Fs_flat, Ls.argmax(axis=-1))
        self.time_data["train"] = Time_Record("Model Training", self.time_data["train"])
        # Record
        self.model = {
            "model": classifier_model,
            "features_flat": Fs_flat,
            "true_labels": Ls,
            "predicted_labels": classifier_model.predict(Fs_flat),
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
            "SHAP": []
        }
        Data = {}
        # Get Data
        classifier_model = self.model["model"]
        Fs_flat = self.model["features_flat"]
        # SHAP Plot
        if disable_plots:
            fig_1 = plt.figure()
            # fig_2 = plt.figure()
        else:
            fig_1 = plt.figure()
            EXPLAINER = shap.TreeExplainer(classifier_model)
            Fs_flat = Fs_flat[0:5]
            SHAP_VALUES = EXPLAINER.shap_values(Fs_flat)
            shap.summary_plot(SHAP_VALUES, Fs_flat, feature_names=self.feature_names_flat, show=False)
            # fig_1 = shap.force_plot(EXPLAINER.expected_value[0], SHAP_VALUES[0], Fs_flat, feature_names=self.feature_names_flat, show=False)
            # shap.waterfall_plot(SHAP_VALUES[0], Fs_flat, show=False)
        ## Record
        Plots["SHAP"].append(fig_1)
        # Plots["SHAP"].append(fig_2)
        Data["Classifier Evaluations"] = {
            **ClassifierEval_Basic(self.model["true_labels"], self.model["predicted_labels"]),
        }
        Data["Time"] = self.time_data
        ## CleanUp
        plt.close(fig_1)
        # plt.close(fig_2)
        # Record
        VisData["figs"]["plotly_chart"]["SHAP"] = Plots["SHAP"]
        VisData["data"] = Data

        return VisData

    def predict(self,
        Fs, 

        **params
        ):
        '''
        Predict

        Segment users using classifier model.

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
    "Decision Tree": {
        "class": UserSegmentation_Interpret_DT,
        "params": {
            "clustering_algorithm": None,
            "decision_tree_criterion": "gini",
        }
    }
}