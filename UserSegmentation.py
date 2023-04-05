"""
User Segmentation

Embedding/Encoding Stage
Inputs:
 - User Data : Features of the user in various forms (Image, Text, Categorical, etc.)
 - User Label : Label of the user in any form (Image, Text, Categorical, etc.)
Outputs:
 - User Features : Encoded Features of the user (N_Features, Feature_Dim)
 - User Label : Label Distribution of the user (Label_Dim,)

Training Stage
Inputs:
 - User Features : Features of Users (N_Features, Feature_Dim)
 - User Label : Label Distribution of Users (Label_Dim,)
Outputs:
 - Trained Model : Model that can be used to predict labels from features

Testing Stage
Inputs:
 - Trained Model : Model that can be used to predict labels from features
 - User Features : Features of user (N_Features, Feature_Dim)
Outputs:
 - User Label : Label Distribution of user (Label_Dim,)
"""

# Imports
# Segmenter Imports
from SegmentationMethods.Demographic import UserSegmentation_Demographic_Cluster_Density
from SegmentationMethods.Demographic import UserSegmentation_Demographic_Cluster_K, UserSegmentation_Demographic_Cluster_Heirarchical
from SegmentationMethods.Demographic import UserSegmentation_Demographic_Cluster_Fuzzy
from SegmentationMethods.Demographic import UserSegmentation_Demographic_Cluster_SOM
from SegmentationMethods.Demographic import UserSegmentation_Demographic_Cluster_Graph
from SegmentationMethods.Demographic import UserSegmentation_Demographic_ClusterClassifier
from SegmentationMethods.Combined import UserSegmentation_DemographicBehavior_ClusterClassifier
# Interpreter Imports
from InterpretationMethods.Demographic import UserSegmentation_Demographic_Interpret_Classifier
# from InterpretationMethods.Demographic import UserSegmentation_Demographic_Interpret_ExKMC
# Dataset Imports
from Data.Datasets.CreditCard_1 import DatasetUtils as DatasetUtils_CreditCard_1
from Data.Datasets.CaravanInsuranceChallenge import DatasetUtils as DatasetUtils_CaravanInsuranceChallenge
from Data.Datasets.MallCustomers import DatasetUtils as DatasetUtils_MallCustomers
from Data.Datasets.BankCustomers_1 import DatasetUtils as DatasetUtils_BankCustomers_1
from Data.Datasets.YoutubeVideosUsers_1 import DatasetUtils as DatasetUtils_YoutubeVideosUsers_1
from Data.Datasets.YoutubeTrendingVideos_1 import DatasetUtils as DatasetUtils_YoutubeTrendingVideos_1

# Main Functions
def SegmentationClasses_Default(N_CLASSES):
    '''
    Segmentation Classes - Default
    '''
    return ["Class_" + str(i) for i in range(N_CLASSES)]

# Main Vars
SEGMENTATION_MODULES = {
    "demographic": {
        "Density-Based": {
            **UserSegmentation_Demographic_Cluster_Density.SEG_FUNCS,
        },
        "Centroid-Based": {
            **UserSegmentation_Demographic_Cluster_K.SEG_FUNCS,
        },
        "Hierarchy-Based": {
            **UserSegmentation_Demographic_Cluster_Heirarchical.SEG_FUNCS,
        },
        "Graph-Based": {
            **UserSegmentation_Demographic_Cluster_Graph.SEG_FUNCS,
        },
        "Fuzzy-Based": {
            **UserSegmentation_Demographic_Cluster_Fuzzy.SEG_FUNCS,
        },
        "Combined": {
            **UserSegmentation_Demographic_Cluster_SOM.SEG_FUNCS,
            **UserSegmentation_Demographic_ClusterClassifier.SEG_FUNCS,
        }
        
        # "Clustering - Known Cluster Count": {
        #     **UserSegmentation_Demographic_Cluster_K.SEG_FUNCS,
        #     **UserSegmentation_Demographic_Cluster_Fuzzy.SEG_FUNCS,
        #     **UserSegmentation_Demographic_Cluster_Heirarchical.SEG_FUNCS,
        #     **UserSegmentation_Demographic_Cluster_SOM.SEG_FUNCS,
        #     **UserSegmentation_Demographic_Cluster_Graph.SEG_FUNCS

        # },
        # "Clustering - Unknown Cluster Count": {
        #     **UserSegmentation_Demographic_Cluster_Density.SEG_FUNCS
        # },
        # "Cluster-Classifier": {
        #     **UserSegmentation_Demographic_ClusterClassifier.SEG_FUNCS
        # }
    },
    "demographic-behavior": {
        "Cluster-Classifier": {
            **UserSegmentation_DemographicBehavior_ClusterClassifier.SEG_FUNCS
        }
    }
}

INTERPRETATION_MODULES = {
    "demographic": {
        "Classifier": {
            **UserSegmentation_Demographic_Interpret_Classifier.INTERPRET_FUNCS
        },
        # "ExKMC": {
        #     **UserSegmentation_Demographic_Interpret_ExKMC.INTERPRET_FUNCS
        # }
    }
}

DATASETS = {
    "Bank Customers 1": DatasetUtils_BankCustomers_1,
    "Caravan Insurance Challenge": DatasetUtils_CaravanInsuranceChallenge,
    "Credit Card 1": DatasetUtils_CreditCard_1,
    "Mall Customers": DatasetUtils_MallCustomers,
    "Youtube Videos-Users 1": DatasetUtils_YoutubeVideosUsers_1,
    "Youtube Trending-Videos 1": DatasetUtils_YoutubeTrendingVideos_1
}