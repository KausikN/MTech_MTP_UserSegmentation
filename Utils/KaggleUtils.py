"""
Kaggle Utils
"""

# Imports
import os
import json
import zipfile

# Main Functions
def KaggleUtils_Auth():
    '''
    KaggleUtils - Authenticate API
    '''
    # Load Config
    CUR_DIR = "_credentials/" # os.path.dirname(os.path.abspath(__file__))
    KAGGLE_CONFIG = json.load(open(os.path.join(CUR_DIR, "kaggle.json"), "r"))
    # Set Env Vars
    os.environ["KAGGLE_USERNAME"] = KAGGLE_CONFIG["username"]
    os.environ["KAGGLE_KEY"] = KAGGLE_CONFIG["key"]
    # Authenticate
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_API = KaggleApi()
    KAGGLE_API.authenticate()

    return KAGGLE_API

def KaggleUtils_ListDataset(dataset_id):
    '''
    KaggleUtils - List Dataset Files
    '''
    global KAGGLE_API
    # Check API
    if KAGGLE_API is None: KAGGLE_API = KaggleUtils_Auth()
    # List Dataset
    files = KAGGLE_API.dataset_list_files(dataset_id).files

    return files

def KaggleUtils_DownloadDataset(dataset_id, path, quiet=False, unzip=True):
    '''
    KaggleUtils - Download Dataset
    '''
    global KAGGLE_API
    # Check API
    if KAGGLE_API is None: KAGGLE_API = KaggleUtils_Auth()
    # Download Dataset
    KAGGLE_API.dataset_download_files(dataset_id, path=path, quiet=quiet, unzip=False)
    # Unzip Dataset
    if unzip:
        zip_path = os.path.join(path, dataset_id.split("/")[1] + ".zip")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path)
        os.remove(zip_path)

# Main Vars
KAGGLE_API = None