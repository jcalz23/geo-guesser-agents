import math
import sys
sys.path.append('../')
from constants import MAX_SCORE, MAX_D

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
    

