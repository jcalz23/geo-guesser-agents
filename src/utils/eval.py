import os
import json
import math
import sys
import pandas as pd

sys.path.append('../')
from src.constants import MAX_SCORE, MAX_D

def calculate_distance(coord1, coord2):
    """
    Calculate the distance between two latitude and longitude coordinates using the Haversine formula.
    
    Args:
    coord1 (dict): A dictionary containing 'latitude' and 'longitude' for the first coordinate.
    coord2 (dict): A dictionary containing 'latitude' and 'longitude' for the second coordinate.
    
    Returns:
    float: The distance in kilometers between the two coordinates.
    """
    try:
        # Radius of the Earth in kilometers
        R = 6371.0
        
        # Convert latitude and longitude from degrees to radians
        lat1, lon1 = math.radians(float(coord1['latitude'])), math.radians(float(coord1['longitude']))
        lat2, lon2 = math.radians(float(coord2['latitude'])), math.radians(float(coord2['longitude']))
        
        # Difference in coordinates
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine formula
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = round(R * c, 2)
        return distance
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return None
    
def geoguessr_score(distance_km):
    """
    Calculate the GeoGuessr score based on the distance error in kilometers.

    Args:
    distance_km (float): The distance error in kilometers.

    Returns:
    int: The GeoGuessr score.
    """
    try:
        if distance_km < 0:
            raise ValueError("Distance cannot be negative.")
        
        if (distance_km >= MAX_D) | (distance_km == None) | (str(distance_km) == "nan"):
            return 0
        
        # Calculate score using the precise quadratic function
        score = MAX_SCORE * (1 - (distance_km / MAX_D) ** 2)
        return max(0, int(score))
    except:
        print()
        print(distance_km)

def get_latest_run_filenames(agent_list, results_dir):
    result_filenames = {}
    for agent in agent_list:
        runs = [x for x in os.listdir(results_dir) if x.startswith(agent)]
        latest_run = sorted(runs, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))[-1]
        result_filenames[agent] = latest_run
    return result_filenames

def load_results(results_dir, result_filenames):
    df = pd.DataFrame()
    for agent, result_filename in result_filenames.items():
        with open(f"{results_dir}/{result_filename}", "r") as file:
            results = json.load(file)
        new_row = pd.DataFrame({k: v["distance"] for k, v in results.items()}, index=[agent])
        df = new_row if len(df) == 0 else pd.concat([df, new_row])
    return df

def calculate_metrics(df):
    t1_cols = [col for col in df.columns if col[0] == "1"]
    t2_cols = [col for col in df.columns if col[0] == "2"]
    df["mean"] = df[t1_cols + t2_cols].mean(axis=1)
    df["mean_t1"] = df[t1_cols].mean(axis=1)
    df["mean_t2"] = df[t2_cols].mean(axis=1)
    df["min"] = df[t1_cols + t2_cols].min(axis=1)
    return df[sorted(df.columns)]

def calculate_scores(df):
    df["total_score"] = df["total_score_t1"] = df["total_score_t2"] = 0
    for index, row in df.iterrows():
        for col in df.columns:
            if col[0] in ["1", "2"]:
                score = geoguessr_score(df.at[index, col])
                df.at[index, "total_score"] += score
                if int(col) <= 110:
                    df.at[index, "total_score_t1"] += score
                elif int(col) > 110:
                    df.at[index, "total_score_t2"] += score
    return df

def calculate_normalized_scores(df):
    score_columns = ['total_score', 'total_score_t1', 'total_score_t2']
    for index, row in df.iterrows():
        non_na_t1 = df.loc[index, '101':'110'].count()
        non_na_t2 = df.loc[index, '201':'210'].count()
        
        df.at[index, 'normalized_total_score_t1'] = df.at[index, 'total_score_t1'] / non_na_t1 if non_na_t1 > 0 else 0
        df.at[index, 'normalized_total_score_t2'] = df.at[index, 'total_score_t2'] / non_na_t2 if non_na_t2 > 0 else 0
        df.at[index, 'normalized_total_score'] = df.at[index, 'total_score'] / (non_na_t1 + non_na_t2) if (non_na_t1 + non_na_t2) > 0 else 0
    return df
    

