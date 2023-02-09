"""
Youtube Utils

Extracting Youtube Data:
https://www.thepythoncode.com/article/using-youtube-api-in-python
"""

# Imports
import os
import re
import pickle
import urllib.parse as urlparse
import googleapiclient.errors
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# Main Vars
YOUTUBE_API_DATA = {
    "token_path": os.path.join("_credentials/", "token.pickle"),
    "client_secret_path": os.path.join("_credentials/", "client_secret_1039579620535-64vtoipetapp4qmf0dg4v6n3m1i5h8nc.apps.googleusercontent.com.json"),
    "api_service_name": "youtube",
    "api_version": "v3",
    "scopes": ["https://www.googleapis.com/auth/youtube.readonly"]
}
# Disable OAuthlib's HTTPS verification when running locally.
# *DO NOT* leave this option enabled in production.
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Main Functions
def YoutubeAPI_GetClient():
    '''
    Youtube API - Get Client
    '''
    # Init
    CREDS = None
    # Check if token exists
    if os.path.exists(YOUTUBE_API_DATA["token_path"]):
        with open(YOUTUBE_API_DATA["token_path"], "rb") as token:
            CREDS = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not CREDS or not CREDS.valid:
        if CREDS and CREDS.expired and CREDS.refresh_token:
            CREDS.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                YOUTUBE_API_DATA["client_secret_path"], 
                YOUTUBE_API_DATA["scopes"]
            )
            CREDS = flow.run_local_server(port=0)
        # save the credentials for the next run
        with open(YOUTUBE_API_DATA["token_path"], "wb") as token:
            pickle.dump(CREDS, token)
    # Create Youtube Client
    YOUTUBE_CLIENT = build(
        YOUTUBE_API_DATA["api_service_name"], 
        YOUTUBE_API_DATA["api_version"], 
        credentials=CREDS
    )

    return YOUTUBE_CLIENT

def YoutubeAPI_GetChannel_FromUsername(name):
    '''
    Youtube API - Get Channel ID from Username
    '''
    request = YOUTUBE_CLIENT.channels().list(
        part="snippet,contentDetails,statistics",
        forUsername=name
    )
    response = request.execute()

    return response

def YoutubeAPI_GetChannel_FromID(ID):
    '''
    Youtube API - Get Channel ID from Username
    '''
    request = YOUTUBE_CLIENT.channels().list(
        part="snippet,contentDetails,statistics",
        id=ID
    )
    response = request.execute()

    return response

# RunCode
YOUTUBE_CLIENT = YoutubeAPI_GetClient()