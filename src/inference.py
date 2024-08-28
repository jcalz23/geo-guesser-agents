"""
This script allows users to select an agent and run inference on a test set.
"""
import json
import logging
import argparse
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any

from agents.single_agent import SingleAgentRunner
from agents.prescribed_chain import PrescribedChainAgent
from agents.multi_agent_supervisor import MultiAgentSupervisorRunner
from agents.react_agent import ReactAgentRunner

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentRunner(ABC):
    """Abstract base class for agent runners."""

    @abstractmethod
    def process_location(self, key: str, target_value: Dict[str, Any]) -> float:
        """
        Process a single location and return the calculated distance.

        Args:
            key (str): The identifier for the location.
            target_value (Dict[str, Any]): The target value for the location.

        Returns:
            float: The calculated distance for the location.
        """
        pass


class InferenceRunner:
    """Class to run inference on a test set using a selected agent."""

    def __init__(self, agent: AgentRunner, agent_name: str, test_set_path: str):
        """
        Initialize the InferenceRunner.

        Args:
            agent (AgentRunner): The agent to use for inference.
            agent_name (str): The name of the agent.
            test_set_path (str): The path to the test set JSON file.
        """
        self.agent = agent
        self.agent_name = agent_name
        self.test_set_path = test_set_path
        self.results_file_path = self._create_results_file_path()

    def _create_results_file_path(self) -> str:
        """
        Create a unique file path for storing results.

        Returns:
            str: The file path for storing results.
        """
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        return f'../data/results/runs/{self.agent_name}_{current_time}.json'

    def load_test_set(self) -> Dict[str, Any]:
        """
        Load the test set from the specified JSON file.

        Returns:
            Dict[str, Any]: The loaded test set.
        """
        with open(self.test_set_path, 'r') as f:
            return json.load(f)

    def run_inference(self):
        """Run inference on the test set using the selected agent."""
        # Load the test set
        test_set = self.load_test_set()

        # Run inference on each location
        logger.info("Starting inference run")
        pred_distances = {}
        for key, value in test_set.items():
            logger.info(f"Processing location: {key}")
            try:
                # Run inference on the location
                pred_distances[key] = self.agent.process_location(key, value)

                # Save results after each location
                with open(self.results_file_path, 'w') as file:
                    json.dump(pred_distances, file)

            except Exception as e:
                logger.error(f"Error processing {key}: {e}", exc_info=True)
                continue

        logger.info(f"Inference completed. Results saved to {self.results_file_path}")


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
    parser = argparse.ArgumentParser(description="Run inference on a test set using a selected agent.")
    parser.add_argument("--agent_name", help="Name of the agent to use (e.g., 'single_agent')")
    parser.add_argument("--test_set_path", help="Path to the test set JSON file")
    args = parser.parse_args()

    try:
        selected_agent = get_agent(args.agent_name)
        runner = InferenceRunner(selected_agent, args.agent_name, args.test_set_path)
        runner.run_inference()
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
