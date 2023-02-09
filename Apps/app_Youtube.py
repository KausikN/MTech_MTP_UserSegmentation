"""
Streamlit App - Youtube Segmentation
"""

# Imports
import os
import time
import json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from UserSegmentation import *
from Utils.YoutubeUtils import *

# Main Vars
TEMP_PATH = "Data/Temp/"
MODELS_DIR = "_models/"
SETTINGS = {
    "plots_interactive": True
}

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

# Cache Data Functions
@st.cache
def CacheData_YoutubeSearch_Channel_ByName(name):
    '''
    Cache Data - Youtube Search Channel by username
    '''
    # Load Dataset
    return YoutubeAPI_GetChannel_FromUsername(name)

@st.cache
def CacheData_YoutubeSearch_Channel_ByID(ID):
    '''
    Cache Data - Youtube Search Channel by ID
    '''
    # Load Dataset
    return YoutubeAPI_GetChannel_FromID(ID)

# UI Functions
def UI_LoadYoutubeDataset(data_type="demographic"):
    '''
    Load Youtube Dataset
    '''
    st.markdown("## Load Youtube Dataset")
    # Search Youtube Channel
    cols = st.columns(2)
    USERINPUT_SearchType = cols[0].selectbox("Search Type", ["By Name", "By ID"])
    if USERINPUT_SearchType == "By Name":
        USERINPUT_SearchQuery = cols[1].text_input("Youtube Channel Name", "PewDiePie")
        SearchData = CacheData_YoutubeSearch_Channel_ByName(USERINPUT_SearchQuery)
    elif USERINPUT_SearchType == "By ID":
        USERINPUT_SearchQuery = cols[1].text_input("Youtube Channel ID", "UC-lHJZR3Gqxm24_Vd_AJ5Yw")
        SearchData = CacheData_YoutubeSearch_Channel_ByID(USERINPUT_SearchQuery)
    if "items" not in SearchData.keys(): return None
    # Select Youtube Channel
    SearchedChannels = [d["snippet"]["title"] for d in SearchData["items"]]
    USERINPUT_YoutubeChannel = st.selectbox("Select Youtube Channel", SearchedChannels)
    ChannelData = SearchData["items"][SearchedChannels.index(USERINPUT_YoutubeChannel)]

    # Options
    cols = st.columns(2)
    USERINPUT_Options = {
        "display": cols[0].checkbox("Display Dataset", value=True)
    }
    # Display
    if USERINPUT_Options["display"]:
        st.write(ChannelData)

    DATA = {
        "name": USERINPUT_YoutubeChannel,
        "data_type": data_type,
        "name": USERINPUT_YoutubeChannel,
        "dataset": ChannelData,
        "params": {}
    }
    return DATA

# Main Functions
def youtube_user_segmentation():
    # Title
    st.markdown("# Youtube Users Segmentation")

    # Load Inputs
    # Init
    DATA_TYPE = "demographic-behavior" # "demographic-behavior" or "demographic"
    PROGRESS_BARS = {
        "overall": ProgressBar("Started", 6)
    }
    # Dataset
    PROGRESS_BARS["overall"].update("Loading Dataset...") # 1
    DATA = UI_LoadYoutubeDataset(DATA_TYPE)

# Mode Vars
MODES = {
    "Youtube User Segmentation": youtube_user_segmentation
}

# App Functions
def app_main():
    # Title
    st.markdown("# MTech Project - Youtube User Segmentation")
    # Mode
    USERINPUT_Mode = st.sidebar.selectbox(
        "Select Mode",
        list(MODES.keys())
    )
    MODES[USERINPUT_Mode]()

# RunCode
if __name__ == "__main__":
    # Assign Objects
    SETTINGS["plots_interactive"] = st.sidebar.checkbox("Interactive Plots", True)
    # Run Main
    app_main()