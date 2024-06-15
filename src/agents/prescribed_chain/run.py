import os
import sys
import json
import getpass
from datetime import datetime

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

# Initialize the Google Maps client with your API key
gmaps = googlemaps.Client(key=os.environ["GOOGLE_API_KEY"])

# Main
def main():
    # Load target locations
    with open(f'{DATA_DIR}master.json', 'r') as f:
        target_locations = json.load(f)

    # Loop through locations
    all_results = {}
    # Save results to working dir with a runtime value
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    results_file_path = f'{DATA_DIR}/results/runs/prescribed_chain_{current_time}.json'
    
    for key, value in target_locations.items():
        print(f"Processing {key}")
        try:
            # Load images
            image_dir = f'{DATA_DIR}locations/{key}/'
            image_inputs = prep_images(image_dir)

            ## Run OpenAI API calls
            # Tool 1: Describe Image w/ GPT4o
            # Describe image
            prompt_inputs = {"json_prompt": JSON_PROMPT}
            text_results = call_openai(MODEL, SYS_MESSAGE, DESCRIBE_TEXT_PROMPT, prompt_inputs, image_inputs)
            print(f"Text results: {text_results}")

            # Describe image scenery
            prompt_inputs = {"json_prompt": JSON_PROMPT}
            scene_results = call_openai(MODEL, SYS_MESSAGE, DESCRIBE_SCENE_PROMPT, prompt_inputs, image_inputs)
            print(f"Scene results: {scene_results}")

            # Come up with potential city candidates
            prompt_inputs = {"json_prompt": JSON_PROMPT, "text_results": text_results, "scene_results": scene_results}
            candidates = call_openai(MODEL, SYS_MESSAGE, CANDIDATES_PROMPT, prompt_inputs)
            print(f"Candidates: {candidates}")

            # Come up with Google searches
            prompt_inputs = {"json_prompt": JSON_PROMPT, "text_results": text_results, "scene_results": scene_results, "candidates": candidates}
            queries = call_openai(MODEL, SYS_MESSAGE, SEARCH_PROMPT, prompt_inputs)
            print(f"Queries: {queries}")
        
            # Run Google Searches
            search_results = {}
            for search_id, query in queries.items():
                result = get_top_n_results(GOOGLE_SEARCH_DESCRIPTION, query)
                search_results[search_id] = result

            # Predict location
            prompt_inputs = {"json_prompt": JSON_PROMPT, "text_results": text_results, "scene_results": scene_results, "candidates": candidates, "search_results": search_results}
            pred = call_openai(MODEL, SYS_MESSAGE, GEO_GUESSER_PROMPT, prompt_inputs)
            print(f"Pred: {pred}")

            ## Evals
            # Calculate distance
            target = target_locations[key]
            distance = calculate_distance(pred, target)
            print(f"Distance: {distance} km")

            # Save results
            all_results[key] = {"distance": distance}

            # Save results to file after each iteration
            with open(results_file_path, 'w') as file:
                json.dump(all_results, file)

        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__=="__main__":
    main()
