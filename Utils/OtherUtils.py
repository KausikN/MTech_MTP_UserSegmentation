"""
Other Utils
"""

# Imports
import os
import pandas as pd

# Main Functions


# RunCode
if __name__ == "__main__":
    # CSV Columns to List String
    path = "Data/Datasets/YoutubeVideosUsers_1/Data/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv"
    df = pd.read_csv(path)
    print(str(list(df.columns)).replace("'", '"'))