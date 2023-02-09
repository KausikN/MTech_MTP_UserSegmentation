"""
Streamlit App - Dataset
"""

# Imports
import os
import cv2
import json
import streamlit as st
from tqdm import tqdm

from UserSegmentation import *

# Main Vars
PATHS = {
    "temp": "Data/Temp/"
}
DEFAULT_CMAP = "gray"

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

def UI_VisualiseDataset(DATASET):
    '''
    Standard Visualisations on Dataset
    '''
    st.markdown("## Visualisations")

# Mode Functions
def visualise_dataset():
    # Title
    st.markdown("# Visualise Dataset")

    # Load Inputs
    USERINPUT_Dataset = UI_LoadDataset()

    # Visualise Dataset
    UI_VisualiseDataset(USERINPUT_Dataset)

# Main Vars
MODES = {
    "Visualise Dataset": visualise_dataset
}

# Main Functions
def app_main():
    # Title
    st.markdown("# User Segmentation Dataset Utils")

    # Load Inputs
    # Method
    USERINPUT_Mode = st.sidebar.selectbox("Select Mode", list(MODES.keys()))
    USERINPUT_ModeFunc = MODES[USERINPUT_Mode]
    USERINPUT_ModeFunc()


# RunCode
if __name__ == "__main__":
    # Assign Objects
    
    # Run Main
    app_main()