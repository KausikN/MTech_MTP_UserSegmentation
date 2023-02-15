"""
Streamlit App
"""

# Imports
import os
import time
import json
import pickle
import functools
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from UserSegmentation import *
from Utils.EvalUtils import *

# Main Vars
PATHS = {
    "temp": "Data/Temp/",
    "models": "_models/",
    "credentials_kaggle": "_credentials/kaggle.json",
    "settings": "_appdata/settings.json"
}
SETTINGS = {}

# Progress Classes
class ProgressBar:
    def __init__(self, title, max_value):
        self.max_value = max_value
        self.title = st.sidebar.empty()
        self.bar = st.sidebar.progress(0)
        self.value = -1
        self.update(title)

    def update(self, title):
        self.title.markdown(title)
        self.value += 1
        self.bar.progress(self.value / self.max_value)

    def finish(self):
        self.title.empty()
        self.bar.empty()

# Utils Functions
def name_to_path(name):
    # Convert to Lowercase
    name = name.lower()
    # Remove Special Chars
    for c in [" ", "-", ".", "<", ">", "/", "\\", ",", ";", ":", "'", '"', "|", "*", "?"]:
        name = name.replace(c, "_")

    return name

# Cache Data Functions
# @st.cache
def CacheData_TrainedModel(
    USERINPUT_SegMethod, dataset,
    n_clusters, keep_cols,
    **params
    ):
    '''
    Cache Data - Trained Model
    '''
    # Load Dataset
    DATASET_MODULE = DATASETS[dataset["name"]]
    DATASET = DATASET_MODULE.DATASET_FUNCS["test"](
        keep_cols=keep_cols,
        data_type=dataset["data_type"],
        other_params=dataset["params"]
    )
    Fs, Ls, FEATURES_INFO = DATASET_MODULE.DATASET_FUNCS["encode"](
        DATASET, 
        n_clusters=n_clusters
    )
    # Init Model
    MODEL = USERINPUT_SegMethod["class"](**USERINPUT_SegMethod["params"])
    MODEL_PARAMS = {
        "features_info": FEATURES_INFO
    }
    # Train Model
    MODEL.train(Fs, Ls, **MODEL_PARAMS)

    return MODEL

# UI Functions
def UI_LoadDataset(data_type="demographic", dataset_params=None):
    '''
    Load Dataset
    '''
    st.markdown("## Load Dataset")
    # Select Dataset
    cols = st.columns((1, 3))
    DATASETS_AVAILABLE = [d for d in list(DATASETS.keys()) if data_type in DATASETS[d].DATASET_DATA.keys()]
    USERINPUT_Dataset = cols[0].selectbox("Select Dataset", DATASETS_AVAILABLE)
    DATASET_MODULE = DATASETS[USERINPUT_Dataset]
    ## Load Params
    if dataset_params is None:
        USERINPUT_DatasetParams_str = cols[1].text_area(
            "Params", 
            value=json.dumps(DATASET_MODULE.DATASET_PARAMS, indent=8),
            height=200
        )
        dataset_params = json.loads(USERINPUT_DatasetParams_str)
    # Load Dataset
    DATASET = DATASET_MODULE.DATASET_FUNCS["test"](
        data_type=data_type,
        other_params=dataset_params
    )
    N = DATASET["N"]

    # Options
    cols = st.columns(2)
    USERINPUT_Options = {
        "n_samples": cols[0].markdown(f"Count: **{N}**"),
        "display": cols[1].checkbox("Display Dataset", value=True)
    }

    # Display
    if USERINPUT_Options["display"]:
        USERINPUT_ViewSampleIndex = st.slider(f"View Sample ({N} Samples)", 0, N-1, 0, 1)
        DisplayData = DATASET_MODULE.DATASET_FUNCS["display"](DATASET, [USERINPUT_ViewSampleIndex, USERINPUT_ViewSampleIndex+1]).to_dict()
        st.table([{k: DisplayData[k][list(DisplayData[k].keys())[0]] for k in DisplayData.keys()}])

    DATA = {
        "name": USERINPUT_Dataset,
        "data_type": data_type,
        "module": DATASET_MODULE,
        "dataset": DATASET,
        "params": dataset_params
    }
    return DATA

def UI_LoadSegModel(DATA):
    '''
    Load Model for Segmentation
    '''
    st.markdown("## Load Segmentation Algo")
    # Load Method
    data_type = DATA["data_type"]
    SegModules = SEGMENTATION_MODULES[data_type]
    USERINPUT_SegModule = st.selectbox("Select Segmentation Module", list(SegModules.keys()))
    cols = st.columns((1, 3))
    USERINPUT_SegMethodName = cols[0].selectbox(
        "Select Segmentation Method",
        list(SegModules[USERINPUT_SegModule].keys())
    )
    USERINPUT_SegMethod = SegModules[USERINPUT_SegModule][USERINPUT_SegMethodName]
    # Load Params
    USERINPUT_SegParams_str = cols[1].text_area(
        "Params", 
        value=json.dumps(USERINPUT_SegMethod["params"], indent=8),
        height=200
    )
    USERINPUT_SegParams = json.loads(USERINPUT_SegParams_str)
    USERINPUT_SegMethod = {
        "class": USERINPUT_SegMethod["class"],
        "params": USERINPUT_SegParams
    }
    USERINPUT_SegParams = json.loads(USERINPUT_SegParams_str) # Redo to create new object
    # Other Params
    DatasetData = DATA["module"].DATASET_DATA[data_type]
    KeepCols = DatasetData["cols"]["keep"]
    KeepCols_Default = DatasetData["cols"]["keep_default"]
    USERINPUT_Dataset = {
        "dataset": {
            "name": DATA["name"],
            "data_type": DATA["data_type"],
            "params": DATA["params"]
        }
    }
    USERINPUT_OtherParams = {
        "n_clusters": st.sidebar.number_input("N Clusters", 1, 1000, 10),
        "keep_cols": {
            k:  st.multiselect(
                    "Keep Columns: " + str(k), 
                    list(KeepCols[k]),
                    default=list(KeepCols_Default[k])
                )
            for k in KeepCols.keys()
        }
    }
    # Process Check
    USERINPUT_Process = st.checkbox("Stream Process", value=False)
    if not USERINPUT_Process: USERINPUT_Process = st.button("Process")
    if not USERINPUT_Process: st.stop()
    # Get Trained Model
    USERINPUT_SegModel = CacheData_TrainedModel(USERINPUT_SegMethod, **USERINPUT_Dataset, **USERINPUT_OtherParams)
    # Update Data
    DATA["model_params"] = {
        "module_name": USERINPUT_SegModule,
        "method_name": USERINPUT_SegMethodName,
        "method_params": USERINPUT_SegParams
    }
    DATA["other_params"] = USERINPUT_OtherParams

    # Display Model Visualisations
    VisData = USERINPUT_SegModel.visualise()
    UI_DisplayVisData(VisData)

    return USERINPUT_SegModel, DATA

def UI_LoadUser(DATA):
    '''
    Load User
    '''
    # Init
    DATASET, DATASET_MODULE = DATA["dataset"], DATA["module"]
    # Load User
    st.markdown("## Load User")
    # Select User
    N = DATASET["N"]
    USERINPUT_ViewSampleIndex = st.slider(f"Select User ({N} Samples)", 0, N-1, 0, 1)
    DisplayData = DATASET_MODULE.DATASET_FUNCS["display"](DATASET, [USERINPUT_ViewSampleIndex, USERINPUT_ViewSampleIndex+1]).to_dict()
    st.table([{k: DisplayData[k][list(DisplayData[k].keys())[0]] for k in DisplayData.keys()}])
    # Encode User
    UserData = DATASET_MODULE.DATASET_FUNCS["test"](
        N=[USERINPUT_ViewSampleIndex, USERINPUT_ViewSampleIndex+1],
        keep_cols=DATA["other_params"]["keep_cols"],
        data_type=DATA["data_type"],
        other_params=DATA["params"]
    )
    User_Fs, User_Ls, FEATURES_INFO = DATASET_MODULE.DATASET_FUNCS["encode"](UserData, n_clusters=DATA["other_params"]["n_clusters"])
    USERINPUT_User = {
        "F": User_Fs,
        "L": User_Ls,
        "features_info": FEATURES_INFO
    }

    return USERINPUT_User

def UI_LoadInterpreterModel(DATA_TYPE="demographic"):
    '''
    Load Model for Interpretation
    '''
    st.markdown("## Load Interpretation Algo")
    # Load Method
    USERINPUT_InterpModule = st.selectbox("Select Interpretation Module", list(INTERPRETATION_MODULES[DATA_TYPE].keys()))
    cols = st.columns((1, 3))
    USERINPUT_InterpMethodName = cols[0].selectbox(
        "Select Interpretation Method",
        list(INTERPRETATION_MODULES[DATA_TYPE][USERINPUT_InterpModule].keys())
    )
    USERINPUT_InterpMethod = INTERPRETATION_MODULES[DATA_TYPE][USERINPUT_InterpModule][USERINPUT_InterpMethodName]
    # Load Params
    USERINPUT_InterpParams_str = cols[1].text_area(
        "Params", 
        value=json.dumps(USERINPUT_InterpMethod["params"], indent=8),
        height=200
    )
    USERINPUT_InterpParams = json.loads(USERINPUT_InterpParams_str)
    USERINPUT_InterpMethod = {
        "class": USERINPUT_InterpMethod["class"],
        "params": USERINPUT_InterpParams
    }

    return USERINPUT_InterpMethod


def UI_DisplayLabelDistribution(User_L, label_names=[]):
    '''
    Display Label Distribution
    '''
    # Init
    if len(label_names) == 0: label_names = ["C_"+str(i) for i in range(len(User_L))]
    # Display
    st.markdown("### Label Distribution")
    ## Plots
    fig = plt.figure()
    plt.bar(label_names, User_L)
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Probability")
    plt.close(fig)
    st.pyplot(fig)
    ## Data
    data = {
        "Best Label": {
            "Label": label_names[np.argmax(User_L)],
            "Value": np.max(User_L)
        },
        "Worst Label": {
            "Label": label_names[np.argmin(User_L)],
            "Value": np.min(User_L)
        }
    }
    st.write(data)

def UI_DisplayVisData(OutData):
    '''
    Display Algorithm Visualisation Data
    '''
    # Init
    st.markdown("# Visualisations")
    st.markdown("## Plots")

    # Graphs
    for k in OutData["figs"]["plotly_chart"].keys():
        st.markdown(f"### {k}")
        cols = st.columns(len(OutData["figs"]["plotly_chart"][k]))
        for i in range(len(OutData["figs"]["plotly_chart"][k])):
            if SETTINGS["plots_interactive"]:
                cols[i].plotly_chart(OutData["figs"]["plotly_chart"][k][i])
            else:
                cols[i].pyplot(OutData["figs"]["plotly_chart"][k][i])
    # Plots
    for k in OutData["figs"]["pyplot"].keys():
        st.markdown(f"### {k}")
        cols = st.columns(len(OutData["figs"]["pyplot"][k]))
        for i in range(len(OutData["figs"]["pyplot"][k])):
            cols[i].pyplot(OutData["figs"]["pyplot"][k][i])
    # Data
    st.markdown("## Data")
    for k in OutData["data"].keys():
        st.markdown(f"### {k}")
        st.write(OutData["data"][k])

def UI_InterpretModel(USERINPUT_SegModel, DATA):
    '''
    Interpret Model
    '''
    # Init
    st.markdown("# Interpret Model")
    # Get User
    USERINPUT_User = UI_LoadUser(DATA)
    # Get User Labels
    USERINPUT_User_L = USERINPUT_SegModel.predict(USERINPUT_User["F"])
    # Display User Labels
    UI_DisplayLabelDistribution(USERINPUT_User_L, label_names=USERINPUT_SegModel.label_names)

# Load / Save Model Functions
def Model_SaveModelData(USERINPUT_SegModel, DATA):
    '''
    Model - Save Model and Dataset metadata
    '''
    # Init
    datatype_name = name_to_path(DATA["data_type"])
    data_name = name_to_path(DATA["name"])
    module_name = name_to_path(DATA["model_params"]["module_name"])
    method_name = name_to_path(DATA["model_params"]["method_name"])
    dir_path = os.path.join(PATHS["models"], datatype_name, data_name, module_name, method_name)
    # Create Dirs
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    # Save Model Data
    USERINPUT_SegModel.save(dir_path)
    # Save Dataset Data
    save_params = {
        "data_name": DATA["name"],
        "data_type": DATA["data_type"],
        "dataset_params": DATA["params"],
        "model_params": DATA["model_params"],
        "other_params": DATA["other_params"]
    }
    json.dump(save_params, open(os.path.join(dir_path, "params.json"), "w"), indent=4)
    # Save Session Data
    pickle.dump(DATA["module"].DATASET_SESSION_DATA, open(os.path.join(dir_path, "session_data.p"), "wb"))

def Model_LoadModelData(path):
    '''
    Model - Load Model and Dataset metadata
    '''
    # Init
    # Check Exists
    if not os.path.exists(path): return None, None
    # Load Session Data
    session_data = pickle.load(open(os.path.join(path, "session_data.p"), "rb"))
    # Load Dataset Data
    load_params = json.load(open(os.path.join(path, "params.json"), "r"))
    # Load Model Data
    # Load Model Base
    USERINPUT_SegModelBase = SEGMENTATION_MODULES[load_params["data_type"]][load_params["model_params"]["module_name"]][load_params["model_params"]["method_name"]]
    USERINPUT_SegModel = USERINPUT_SegModelBase["class"](load_params["model_params"]["method_params"])
    USERINPUT_SegModel.load(path)

    return USERINPUT_SegModel, load_params, session_data

# Main Functions
def user_segmentation_train_basic(DATA_TYPE="demographic"):
    # Title
    st.markdown(f"# Segmentation - {DATA_TYPE} - Train")

    # Load Inputs
    # Init
    PROGRESS_BARS = {
        "overall": ProgressBar("Started", 4)
    }
    # Dataset
    PROGRESS_BARS["overall"].update("Loading Dataset...") # 1
    DATA = UI_LoadDataset(DATA_TYPE)

    # Process Inputs
    # Train Model
    PROGRESS_BARS["overall"].update("Training Model...") # 2
    USERINPUT_SegModel, DATA = UI_LoadSegModel(DATA)
    # Save Model
    PROGRESS_BARS["overall"].update("Saving Model...") # 3
    Model_SaveModelData(USERINPUT_SegModel, DATA)
    PROGRESS_BARS["overall"].update("Finished") # 4
    PROGRESS_BARS["overall"].finish()

def user_segmentation_test_basic(DATA_TYPE="demographic"):
    # Title
    st.markdown(f"# Segmentation - {DATA_TYPE} - Test")

    # Load Inputs
    # Init
    PROGRESS_BARS = {
        "overall": ProgressBar("Started", 7)
    }
    # Load Dataset
    PROGRESS_BARS["overall"].update("Loading Dataset...") # 1
    DATA = UI_LoadDataset(DATA_TYPE)
    # Load Model
    PROGRESS_BARS["overall"].update("Loading Model...") # 2
    datatype_name = name_to_path(DATA["data_type"])
    data_name = name_to_path(DATA["name"])
    cols = st.columns(2)
    USERINPUT_SegModule = cols[0].selectbox("Select Segmentation Module", list(SEGMENTATION_MODULES[DATA["data_type"]].keys()))
    USERINPUT_SegMethodName = cols[1].selectbox(
        "Select Segmentation Method",
        list(SEGMENTATION_MODULES[DATA["data_type"]][USERINPUT_SegModule].keys())
    )
    module_name = name_to_path(USERINPUT_SegModule)
    method_name = name_to_path(USERINPUT_SegMethodName)
    dir_path = os.path.join(PATHS["models"], datatype_name, data_name, module_name, method_name)
    if not os.path.exists(dir_path):
        st.error("No Model Found")
        return
    # Load Model
    USERINPUT_SegModel, LOAD_PARAMS, SESSION_DATA = Model_LoadModelData(dir_path)
    DATA["module"].DATASET_SESSION_DATA = SESSION_DATA
    DATA["params"] = LOAD_PARAMS["dataset_params"]
    DATA["model_params"] = LOAD_PARAMS["model_params"]
    DATA["other_params"] = LOAD_PARAMS["other_params"]
    # User
    PROGRESS_BARS["overall"].update("Loading User...") # 3
    USERINPUT_User = UI_LoadUser(DATA)

    # Process Inputs
    # Process Check
    USERINPUT_Process = st.checkbox("Stream Process", value=False)
    if not USERINPUT_Process: USERINPUT_Process = st.button("Process")
    if not USERINPUT_Process: st.stop()
    # Segmentation
    PROGRESS_BARS["overall"].update("Predicting User...") # 4
    User_L = USERINPUT_SegModel.predict(USERINPUT_User["F"])[0]
    # Display Outputs
    PROGRESS_BARS["overall"].update("Visualising User Predictions...") # 5
    st.markdown("## Segmentation Output")
    UI_DisplayLabelDistribution(User_L)
    # Display Model Visualisations
    PROGRESS_BARS["overall"].update("Visualising Model...") # 6
    st.markdown("## Model Visualisation")
    VisData = USERINPUT_SegModel.visualise()
    UI_DisplayVisData(VisData)
    PROGRESS_BARS["overall"].update("Finished") # 7
    PROGRESS_BARS["overall"].finish()

def user_segmentation_interpret_basic(DATA_TYPE="demographic"):
    # Title
    st.markdown(f"# Segmentation - {DATA_TYPE} - Interpret")

    # Load Inputs
    # Init
    PROGRESS_BARS = {
        "overall": ProgressBar("Started", 6)
    }
    # Load Dataset
    PROGRESS_BARS["overall"].update("Loading Dataset...") # 1
    DATA = UI_LoadDataset(DATA_TYPE)
    # Load Model
    PROGRESS_BARS["overall"].update("Loading Model...") # 2
    datatype_name = name_to_path(DATA["data_type"])
    data_name = name_to_path(DATA["name"])
    cols = st.columns(2)
    USERINPUT_SegModule = cols[0].selectbox("Select Segmentation Module", list(SEGMENTATION_MODULES[DATA["data_type"]].keys()))
    USERINPUT_SegMethodName = cols[1].selectbox(
        "Select Segmentation Method",
        list(SEGMENTATION_MODULES[DATA["data_type"]][USERINPUT_SegModule].keys())
    )
    module_name = name_to_path(USERINPUT_SegModule)
    method_name = name_to_path(USERINPUT_SegMethodName)
    dir_path = os.path.join(PATHS["models"], datatype_name, data_name, module_name, method_name)
    if not os.path.exists(dir_path):
        st.error("No Model Found")
        return
    # Select Interpreter Model
    PROGRESS_BARS["overall"].update("Loading Interpreter...") # 3
    USERINPUT_InterpreterMethod = UI_LoadInterpreterModel(DATA_TYPE)

    # Process Inputs
    # Process Check
    USERINPUT_Process = st.checkbox("Stream Process", value=False)
    if not USERINPUT_Process: USERINPUT_Process = st.button("Process")
    if not USERINPUT_Process: st.stop()
    # Load Model
    USERINPUT_SegModel, LOAD_PARAMS, SESSION_DATA = Model_LoadModelData(dir_path)
    DATA["module"].DATASET_SESSION_DATA = SESSION_DATA
    DATA["params"] = LOAD_PARAMS["dataset_params"]
    DATA["model_params"] = LOAD_PARAMS["model_params"]
    DATA["other_params"] = LOAD_PARAMS["other_params"]
    # Load Interpreter
    USERINPUT_InterpreterMethod["params"].update({
        "clustering_algorithm": USERINPUT_SegModel,
    })
    USERINPUT_InterpreterModel = USERINPUT_InterpreterMethod["class"](
        **USERINPUT_InterpreterMethod["params"]
    )
    USERINPUT_InterpreterModel.train()
    # Display Interpreter Visualisations
    PROGRESS_BARS["overall"].update("Visualising Interpreter...") # 4
    st.markdown("## Interpreter Visualisation")
    VisData_Interpreter = USERINPUT_InterpreterModel.visualise()
    UI_DisplayVisData(VisData_Interpreter)
    # Display Model Visualisations
    PROGRESS_BARS["overall"].update("Visualising Model...") # 5
    st.markdown("## Model Visualisation")
    VisData = USERINPUT_SegModel.visualise()
    UI_DisplayVisData(VisData)
    PROGRESS_BARS["overall"].update("Finished") # 6
    PROGRESS_BARS["overall"].finish()

# Mode Vars
APP_MODES = {
    "User Segmentation - Demographic": {
        "Train": functools.partial(user_segmentation_train_basic, DATA_TYPE="demographic"),
        "Test": functools.partial(user_segmentation_test_basic, DATA_TYPE="demographic"),
        "Interpret": functools.partial(user_segmentation_interpret_basic, DATA_TYPE="demographic")
    },
    "User Segmentation - Demographic-Behavior": {
        "Train": functools.partial(user_segmentation_train_basic, DATA_TYPE="demographic-behavior"),
        "Test": functools.partial(user_segmentation_test_basic, DATA_TYPE="demographic-behavior"),
        "Interpret": functools.partial(user_segmentation_interpret_basic, DATA_TYPE="demographic-behavior")
    }
}

# App Functions
def app_main():
    # Title
    st.markdown("# MTech Project - User Segmentation")
    # Mode
    USERINPUT_App = st.sidebar.selectbox(
        "Select App",
        list(APP_MODES.keys())
    )
    USERINPUT_Mode = st.sidebar.selectbox(
        "Select Mode",
        list(APP_MODES[USERINPUT_App].keys())
    )
    APP_MODES[USERINPUT_App][USERINPUT_Mode]()

def app_settings():
    global SETTINGS
    # Title
    st.markdown("# Settings")
    # Load Settings
    if SETTINGS["kaggle"]["username"] == "" or SETTINGS["kaggle"]["key"] == "":
        if os.path.exists(PATHS["credentials_kaggle"]): SETTINGS["kaggle"] = json.load(open(PATHS["credentials_kaggle"], "r"))
    # Settings
    SETTINGS["plots_interactive"] = st.checkbox("Interactive Plots", False)
    SETTINGS["kaggle"] = json.loads(st.text_area("Kaggle", json.dumps(SETTINGS["kaggle"], indent=4), height=250))
    # Save Settings
    if st.button("Save Settings"):
        json.dump(SETTINGS, open(PATHS["settings"], "w"), indent=4)
        # Settings Operations
        os.makedirs(os.path.dirname(PATHS["credentials_kaggle"]), exist_ok=True)
        if not (SETTINGS["kaggle"]["username"] == "" or SETTINGS["kaggle"]["key"] == ""):
            json.dump(SETTINGS["kaggle"], open(PATHS["credentials_kaggle"], "w"))
        st.success("Settings Saved")

# RunCode
if __name__ == "__main__":
    # Assign Objects
    SETTINGS = json.load(open(PATHS["settings"], "r"))
    SETTINGS_ACTIVE = st.sidebar.checkbox("Show Settings", False)
    if SETTINGS_ACTIVE:
        # Run Settings
        app_settings()
    else:
        # Run Main
        app_main()