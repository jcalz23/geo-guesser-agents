"""
This script runs the prescribed chain agent, which is a chain of thought agent that uses a
defined series of prompts to infer the location of a street view image.
"""

# Imports
import os
import sys
import json
import logging
import getpass
from datetime import datetime
from typing import Dict, Any
import googlemaps
sys.path.append('../../')
from utils.helpers import call_openai, prep_images, get_top_n_results
from utils.eval import calculate_distance
from utils.constants import *
from prompts import *

# Get keys
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

# Constants
DATA_DIR = "../../../data/"
MODEL = "gpt-4o"

# Initialize the gmaps client, logger
gmaps = googlemaps.Client(key=os.environ["GOOGLE_API_KEY"])
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_target_locations() -> Dict[str, Any]:
    """
    Loads the target locations from the master json file.
    """
    with open(f'{DATA_DIR}master.json', 'r') as f:
        return json.load(f)

def process_location(key: str,
                     target: Any,
                     image_inputs: Dict[str, str],
                     all_results: Dict[str, Any]) -> None:
    """
    Processes a single location with the prescribed chain agent.
    """
    # Load images
    image_dir = f'{DATA_DIR}locations/{key}/'
    image_inputs = prep_images(image_dir)

    ## Run OpenAI API calls
    # Describe image
    prompt_inputs = {"json_prompt": JSON_PROMPT}
    text_results = call_openai(MODEL, SYS_MESSAGE, DESCRIBE_TEXT_PROMPT, prompt_inputs, image_inputs)
    logging.info(f"Text results: {text_results}")

    # Describe image scenery
    scene_results = call_openai(MODEL, SYS_MESSAGE, DESCRIBE_SCENE_PROMPT, prompt_inputs, image_inputs)
    logging.info(f"Scene results: {scene_results}")

    # Come up with potential city candidates
    prompt_inputs["text_results"] = text_results
    prompt_inputs["scene_results"] = scene_results
    candidates = call_openai(MODEL, SYS_MESSAGE, CANDIDATES_PROMPT, prompt_inputs)
    logging.info(f"Candidates: {candidates}")

    # Come up with Google searches
    prompt_inputs["candidates"] = candidates
    queries = call_openai(MODEL, SYS_MESSAGE, SEARCH_PROMPT, prompt_inputs)
    logging.info(f"Queries: {queries}")

    # Run Google Searches
    search_results = {}
    for search_id, query in queries.items():
        result = get_top_n_results(GOOGLE_SEARCH_DESCRIPTION, query)
        search_results[search_id] = result

    # Predict location
    prompt_inputs["search_results"] = search_results
    pred = call_openai(MODEL, SYS_MESSAGE, GEO_GUESSER_PROMPT, prompt_inputs)
    logging.info(f"Pred: {pred}")

    ## Evals
    # Calculate distance
    distance = calculate_distance(pred, target)
    logging.info(f"Distance: {distance} km")

    # Save results
    all_results[key] = {"distance": distance}

def save_results(results: Dict[str, Any], file_path: str) -> None:
    """
    Saves the results to a file.
    """
    with open(file_path, 'w') as file:
        json.dump(results, file)

def main():
    # Load target locations
    target_locations = load_target_locations()

    # Save results to working dir with a runtime value
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    results_file_path = f'{DATA_DIR}/results/runs/prescribed_chain_{current_time}.json'

    # Loop through locations in test set
    all_results = {}
    for key, target in target_locations.items():
        logging.info(f"Processing {key}")
        image_inputs = prep_images(f'{DATA_DIR}locations/{key}/')
        try:
            # Infer location
            process_location(key, target, image_inputs, all_results)
            
            # Save results to file after each iteration
            save_results(all_results, results_file_path)

        except Exception as e:
            logging.error(f"Error processing {key}: {e}", exc_info=True)
            continue

if __name__=="__main__":
    main()
