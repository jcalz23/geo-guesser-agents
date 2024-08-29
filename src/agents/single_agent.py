"""
This script runs a single agent for processing location data.
"""
import os
import sys
import json
import getpass
import logging
from typing import TypedDict, Annotated, Sequence
import operator

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_community import GooglePlacesTool
from langchain_core.messages import HumanMessage
from langchain.tools import StructuredTool

sys.path.append('../')
from utils.helpers import call_openai, prep_images, get_street_view_image
from utils.eval import calculate_distance
from constants import *
from prompts.single_agent import *

# Set env vars
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
required_env_vars = ['LANGCHAIN_API_KEY', 'TAVILY_API_KEY', 'OPENAI_API_KEY', 'GPLACES_API_KEY']
for var in required_env_vars:
    if var not in os.environ:
        os.environ[var] = getpass.getpass(f"Enter your {var}: ")


# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define runner class
class SingleAgentRunner:
    """
    A class to run a single agent for processing location data.
    """

    def __init__(self):
        """Initialize the SingleAgentRunner with necessary setup."""
        self.setup_logging()
        self.setup_tools()
        self.setup_model()
        self.app = self.create_graph()

    def setup_logging(self):
        """Configure logging for the application."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger("googlemaps.client").setLevel(logging.WARNING)
        logging.getLogger("langchain.callbacks.shared").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
    
    def setup_tools(self):
        """Initialize the tools used by the agent."""
        self.tools = [
            TavilySearchResults(max_results=3),
            GooglePlacesTool(),
            StructuredTool.from_function(get_street_view_image)
        ]

    def setup_model(self):
        """Set up the language model with tools."""
        self.model = ChatOpenAI(model=SINGLE_AGENT_MODEL, temperature=SINGLE_AGENT_TEMPERATURE)
        self.model = self.model.bind_tools(self.tools)

    def should_continue(self, state):
        """Determine if the agent should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        return "end" if not last_message.tool_calls else "continue"

    def call_model(self, state):
        """Call the model with the current state."""
        messages = state["messages"]
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def create_graph(self):
        """Create the graph for the agent."""
        # Define the function to execute tools
        tool_node = ToolNode(self.tools)

        # Define the graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("action", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", # Start with agent
            self.should_continue, # Decide what to call next or end
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")
        app = workflow.compile()

        return app

    def process_location(self, key, target_value):
        """Process a single location."""
        # Prep initial inputs with prompt + images
        image_dir = f'../data/locations/{key}/'
        image_inputs = prep_images(image_dir, max_size=(600,600), quality=85)
        inputs = {"messages": [HumanMessage(content=text_input + image_inputs)]}

        # Log the total number of tokens in the input
        total_tokens = self.model.get_num_tokens(str(inputs["messages"]))
        logging.info(f"Total tokens in input: {total_tokens}")
        
        # Log the start of processing for this location
        logging.info(f"Starting processing for location: {key}")
        logging.info("Graph execution stream:")

        # Iterate through each step of the graph execution
        for step, output in enumerate(self.app.stream(inputs), start=1):
            for node, content in output.items():
                # Log the step number and node name
                if 'data:image/jpeg;base64' in str(content):
                    logging.info(f"Step {step} - Node: {node} - [Sent image to LLM for analysis]")
                else:
                    logging.info(f"Step {step} - Node: {node} - {content}")
            logging.info("--------------------------------")
            logging.info("--------------------------------")

        # Log the completion of graph execution
        logging.info("Graph execution completed")

        # Use LLM to format final output in expected json
        prompt_inputs = {"pred": str(output), "json_prompt": JSON_PROMPT}
        sys_message = {"role": "system", "content": "Return valid json given input."}
        pred = call_openai(SIN, sys_message, PRED_FORMAT_PROMPT_TEMPLATE, prompt_inputs)

        # Log the prediction in a readable JSON format
        logging.info(f"Prediction for {key}: {json.dumps(pred)}")

        # Calculate and log the distance
        distance = calculate_distance(pred, target_value)
        logging.info(f"Calculated distance for {key}: {distance} km")
        
        return distance
