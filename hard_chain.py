import os
import json
import requests
import getpass

import googlemaps

from utils.helpers import prep_images
from utils.eval import calculate_distance
from utils.constants import *

# Get keys
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

# Initialize the Google Maps client with your API key
gmaps = googlemaps.Client(key=os.environ["GOOGLE_API_KEY"])

# Define shared prompts
SYS_MESSAGE_PROMPT = """You are an expert at the game GeoGuessr. \
    You are trying to determine the location of an image in world coordinates."""
SYS_MESSAGE = {"role": "system", "content": SYS_MESSAGE_PROMPT}
JSON_PROMPT = """Only return a valid json string (RCF8259). Do provide any other commentary. \
    Do not wrap the JSON in markdown such as ```json. Only use the data from the provided content."""



# Main
def main():
    # Load target locations
    with open(f'./data/master.json', 'r') as f:
        target_locations = json.load(f)

    # Loop through locations
    for key, value in target_locations.items():
        # Load images
        image_inputs = prep_images(key)

    ## Run OpenAI API calls
    # Tool 1: Describe Image w/ GPT4o


    # Calculate distance

    # Save results
