"""
This script allows users to select an agent and run inference on a test set.
"""
import json
import logging
import argparse
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

from agents.single_agent import SingleAgentRunner
from agents.prescribed_chain import PrescribedChainAgent
from agents.multi_agent_supervisor import MultiAgentSupervisorRunner
from agents.react_agent import ReactAgentRunner
from utils.helpers import prep_images

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentRunner(ABC):
    """Abstract base class for agent runners."""

    @abstractmethod
    def process_location(self, image_list: List[str], target_dict: Dict[str, Any]) -> float:
        """
        Process a single location and return the calculated distance.

        Args:
            image_list (List[str]): The list of image paths.
            target_dict (Dict[str, Any]): The target value for the location.

        Returns:
            float: The calculated distance for the location.
        """
        pass


class InferenceRunner:
    """Class to run inference on a test set using a selected agent."""

    def __init__(self, agent: AgentRunner, agent_name: str,
                 test_set_path: Optional[str] = None, image_dir: Optional[str] = None,
                 target_latitude: Optional[float] = None, target_longitude: Optional[float] = None):
        """
        Initialize the InferenceRunner.

        Args:
            agent (AgentRunner): The agent to use for inference.
            agent_name (str): The name of the agent.
            test_set_path (str): The path to the test set JSON file.
            image_dir (str): The path to a single image directory for inference.
            target_latitude (float): The target latitude for distance calculation.
            target_longitude (float): The target longitude for distance calculation.
        """
        self.agent = agent
        self.agent_name = agent_name
        self.test_set_path = test_set_path
        self.image_dir = image_dir
        self.target_latitude = target_latitude
        self.target_longitude = target_longitude
        self.results_file_path = self._create_results_file_path()

    def _create_results_file_path(self) -> str:
        """
        Create a unique file path for storing results.

        Returns:
            str: The file path for storing results.
        """
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        return f'../data/results/runs/all/{self.agent_name}_{current_time}.json'

    def load_test_set(self) -> Dict[str, Any]:
        """
        Load the test set from the specified JSON file.

        Returns:
            Dict[str, Any]: The loaded test set.
        """
        with open(self.test_set_path, 'r') as f:
            return json.load(f)

    def inference(self, key: str, image_dir: List[str], target_dict: Dict[str, Any],
                  output_dict: Dict[str, Any]) -> Tuple[str, float]:
        """Run inference on a single image using the selected agent.

        Args:
            key (str): The identifier for the location.
            image_dir (List[str]): The directory of images.
            target_dict (Dict[str, Any]): The target value for the location.
            output_dict (Dict[str, Any]): The dictionary to store the results.

        Returns:
            Tuple[str, float]: The predicted location and the distance.
        """
        # Load images from directory
        image_list = prep_images(image_dir)

        # Run inference on the location
        pred, distance = self.agent.process_location(image_list, target_dict)

        # Save results after each location
        output_dict[key] = {"pred": pred, "distance": distance}
        with open(self.results_file_path, 'w') as file:
            json.dump(output_dict, file)

    def eval_single_image(self):
        """Run inference on a single image using the selected agent.

        Args:
            image_dir (str): The directory of images.
            target_latitude (float): The target latitude for distance calculation.
            target_longitude (float): The target longitude for distance calculation.

        Returns:
            Tuple[str, float]: The predicted location and the distance.
        """
        logger.info("process=eval_single_image, msg=Starting inference run")
        # Define inputs
        key = self.image_dir.split('/')[-1]
        target_dict = None
        if self.target_latitude and self.target_longitude:
            target_dict = {"latitude": float(self.target_latitude), "longitude": float(self.target_longitude)}
        
        # Run inference
        self.inference(key, self.image_dir, target_dict, output_dict={})
        logger.info(f"process=eval_single_image, msg=Inference completed. Results saved to {self.results_file_path}")

    def eval_test_set(self):
        """Run inference on the test set using the selected agent.

        Args:
            test_set_path (str): The path to the test set JSON file.

        Returns:
            None
        """
        # Load the test set
        test_set = self.load_test_set()

        # Run inference on each location
        logger.info("process=eval_test_set, msg=Starting inference run")
        output_dict = {}
        for key, target_dict in test_set.items():
            logger.info(f"process=eval_test_set, msg=Processing location: {key}")
            try:
                # Load images
                image_dir = f'../data/locations/{key}/'
                self.inference(key, image_dir, target_dict, output_dict)
            except Exception as e:
                logger.error(f"process=eval_test_set, msg=Error processing {key}: {e}", exc_info=True)
                continue
        logger.info(f"process=eval_test_set, msg=Inference completed. Results saved to {self.results_file_path}")

    def main(self):
        """Run inference on the test set or single image directory using the selected agent."""
        if self.image_dir:
            self.eval_single_image()
        elif self.test_set_path:
            self.eval_test_set()
        else:
            raise ValueError("Either test_set_path or image_dir must be provided.")

def get_agent(agent_name: str) -> AgentRunner:
    """
    Get the agent based on the provided name.

    Args:
        agent_name (str): The name of the agent.

    Returns:
        AgentRunner: The selected agent runner.
    """
    if agent_name.lower() == "single_agent":
        return SingleAgentRunner()
    elif agent_name.lower() == "prescribed_chain":
        return PrescribedChainAgent()
    elif agent_name.lower() == "multi_agent_supervisor":
        return MultiAgentSupervisorRunner()
    elif agent_name.lower() == "react_agent":
        return ReactAgentRunner()
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference on a test set or single image directory using a selected agent.")
    parser.add_argument("--agent_name", help="Name of the agent to use (e.g., 'single_agent')")
    parser.add_argument("--test_set_path", default=None, help="Path to the test set JSON file")
    parser.add_argument("--image_dir", default=None, help="Path to a single image directory for inference")
    parser.add_argument("--target_latitude", default=None, help="Target latitude for distance calculation")
    parser.add_argument("--target_longitude", default=None, help="Target longitude for distance calculation")
    args = parser.parse_args()

    try:
        selected_agent = get_agent(args.agent_name)
        runner = InferenceRunner(selected_agent, args.agent_name, args.test_set_path, args.image_dir,
                                args.target_latitude, args.target_longitude)
        runner.main()
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
