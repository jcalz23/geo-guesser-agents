"""
This script defines the PrescribedChainAgent class, which is a chain of thought agent that uses a
defined series of prompts to infer the location of a street view image.
"""

# Imports
import os
import sys
import logging
import getpass
from typing import List, Dict, Any, Tuple
import googlemaps

sys.path.append("..")
from src.utils.helpers import call_openai, get_top_n_results
from src.utils.eval import calculate_distance
from src.constants import *
from src.prompts.prescribed_chain import *


# Get keys
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")


class PrescribedChainAgent:
    """
    A chain of thought agent that uses a defined series of prompts to infer the location of a street view image.
    """
    def __init__(self):
        # Initialize the gmaps client
        self.gmaps = googlemaps.Client(key=os.environ["GOOGLE_API_KEY"])
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the application."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger("googlemaps.client").setLevel(logging.WARNING)
        logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.WARNING)
        logging.getLogger("langchain.callbacks.shared").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def process_location(self, image_list: List[str], target: Dict[str, Any]) -> Tuple[str, float]:
        """
        Processes a single location with the prescribed chain agent.

        Args:
            image_list (List[str]): A list of file paths to the images.
            target (Dict[str, Any]): A dictionary containing the target location information.

        Returns:
            float: The distance between the predicted location and the target location.
        """
        ## Run OpenAI API calls
        # Describe image
        prompt_inputs = {"json_prompt": JSON_PROMPT}
        text_results = call_openai(
            PRESCRIBED_CHAIN_MODEL, SYS_MESSAGE, DESCRIBE_TEXT_PROMPT, prompt_inputs, image_list
            )
        logging.info(f"Step 1a - Extract text from image: {text_results}")
        logging.info("--------------------------------")
        logging.info("--------------------------------")

        # Describe image scenery
        scene_results = call_openai(
            PRESCRIBED_CHAIN_MODEL, SYS_MESSAGE, DESCRIBE_SCENE_PROMPT, prompt_inputs, image_list
            )
        logging.info(f"Step 1b - Describe image scenery: {scene_results}")
        logging.info("--------------------------------")
        logging.info("--------------------------------")

        # Come up with potential city candidates
        prompt_inputs["text_results"] = text_results
        prompt_inputs["scene_results"] = scene_results
        candidates = call_openai(PRESCRIBED_CHAIN_MODEL, SYS_MESSAGE, CANDIDATES_PROMPT, prompt_inputs)
        logging.info(f"Step 2 - Come up with potential city candidates: {candidates}")
        logging.info("--------------------------------")
        logging.info("--------------------------------")

        # Come up with Google searches
        prompt_inputs["candidates"] = candidates
        queries = call_openai(PRESCRIBED_CHAIN_MODEL, SYS_MESSAGE, SEARCH_PROMPT, prompt_inputs)
        logging.info(f"Step 3a - Come up with Google searches: {queries}")
        logging.info("--------------------------------")
        logging.info("--------------------------------")

        # Run Google Searches
        search_results = {}
        for search_id, query in queries.items():
            result = get_top_n_results(GOOGLE_SEARCH_DESCRIPTION, query)
            search_results[search_id] = result
        logging.info(f"Step 3b - Run Google searches: {search_results}")
        logging.info("--------------------------------")
        logging.info("--------------------------------")

        # Predict location
        prompt_inputs["search_results"] = search_results
        pred = call_openai(PRESCRIBED_CHAIN_MODEL, SYS_MESSAGE, GEO_GUESSER_PROMPT, prompt_inputs)
        logging.info(f"Prediction: {pred}\n")

        ## Evals
        if target:
         # Calculate distance
            distance = calculate_distance(pred, target)
            logging.info(f"Distance: {distance} km")
        else:
            distance = None

        return pred, distance
