import math

def calculate_distance(coord1, coord2):
    """
    Calculate the distance between two latitude and longitude coordinates using the Haversine formula.
    
    Args:
    coord1 (dict): A dictionary containing 'latitude' and 'longitude' for the first coordinate.
    coord2 (dict): A dictionary containing 'latitude' and 'longitude' for the second coordinate.
    
    Returns:
    float: The distance in kilometers between the two coordinates.
    """
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