"""
This module runs a multi-agent supervisor for predicting location from a street view image.
"""
import os
import sys
import logging
import getpass
import functools
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_community import GooglePlacesTool
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.tools import StructuredTool

sys.path.append("..")
from utils.helpers import call_openai, prep_images, get_street_view_image
from utils.eval import calculate_distance
from constants import *
from prompts.multi_agent_supervisor import *


class AgentState(TypedDict):
    """Type definition for agent state."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

class MultiAgentSupervisorRunner:
    """
    A class to run a multi-agent supervisor for processing location data.
    """

    def __init__(self):
        """Initialize the MultiAgentSupervisorRunner with necessary setup."""
        self.setup_logging()
        self.setup_env_vars()
        self.setup_tools()
        self.setup_model()
        self.graph = self.create_graph()

    def setup_logging(self):
        """Configure logging for the application."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        for logger_name in ["googlemaps.client", "langchain.callbacks.shared", "httpx"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    def setup_env_vars(self):
        """Set up environment variables."""
        os.environ['LANGCHAIN_TRACING_V2'] = 'false'
        required_env_vars = ['LANGCHAIN_API_KEY', 'TAVILY_API_KEY', 'OPENAI_API_KEY', 'GPLACES_API_KEY']
        for var in required_env_vars:
            if var not in os.environ:
                os.environ[var] = getpass.getpass(f"Enter your {var}: ")

    def setup_tools(self):
        """Initialize the tools used by the agents."""
        self.tools = [
            TavilySearchResults(max_results=3),
            GooglePlacesTool(),
            StructuredTool.from_function(get_street_view_image)
        ]

    def setup_model(self):
        """Set up the language model."""
        self.model = ChatOpenAI(model=MULTI_AGENT_MODEL, temperature=MULTI_AGENT_TEMPERATURE)

    def create_agent(self, tools: list, system_prompt: str) -> AgentExecutor:
        """
        Create an agent with specified tools and system prompt.

        Args:
            tools (list): List of tools for the agent.
            system_prompt (str): System prompt for the agent.

        Returns:
            AgentExecutor: The created agent executor.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_openai_tools_agent(self.model, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools)

    def agent_node(self, state: AgentState, agent: AgentExecutor, name: str) -> dict:
        """
        Define an agent node for the graph.

        Args:
            state (AgentState): Current state of the agent.
            agent (AgentExecutor): The agent executor.
            name (str): Name of the agent.

        Returns:
            dict: Updated state with new message.
        """
        result = agent.invoke(state)
        return {"messages": [HumanMessage(content=result["output"], name=name)]}

    def create_supervisor_chain(self):
        """Create the supervisor chain for routing between agents."""
        # Define options and function definition
        options = ["FINISH"] + MULTI_AGENT_MEMBERS
        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "anyOf": [
                            {"enum": options},
                        ],
                    }
                },
                "required": ["next"],
            },
        }

        # Create supervisor prompt
        supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_SYS_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            SUPERVISOR_USER_PROMPT,
        ]).partial(options=str(options), MULTI_AGENT_MEMBERS=", ".join(MULTI_AGENT_MEMBERS))

        # Create supervisor chain
        return (
            supervisor_prompt
            | self.model.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
        )

    def create_graph(self):
        """Create the graph for the multi-agent system."""
        # Create agents
        research_agent = self.create_agent([self.tools[0]], "You are a web researcher.")
        research_node = functools.partial(self.agent_node, agent=research_agent, name="Researcher")
        map_searcher_agent = self.create_agent([self.tools[1]], "You are a map searcher.")
        map_searcher_node = functools.partial(self.agent_node, agent=map_searcher_agent, name="Map_Searcher")
        street_viewer_agent = self.create_agent([self.tools[2]], "You are a street view analyst.")
        street_viewer_node = functools.partial(self.agent_node, agent=street_viewer_agent, name="Street_Viewer")

        # Create supervisor chain
        supervisor_chain = self.create_supervisor_chain()

        # Create graph
        workflow = StateGraph(AgentState)
        workflow.add_node("Researcher", research_node)
        workflow.add_node("Map_Searcher", map_searcher_node)
        workflow.add_node("Street_Viewer", street_viewer_node)
        workflow.add_node("supervisor", supervisor_chain)

        # Add edges from each agent to supervisor
        for member in MULTI_AGENT_MEMBERS:
            workflow.add_edge(member, "supervisor")

        # Add conditional edges from supervisor to agents or end
        conditional_map = {k: k for k in MULTI_AGENT_MEMBERS}
        conditional_map["FINISH"] = END
        workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

        # Set entry point and compile graph
        workflow.set_entry_point("supervisor")
        return workflow.compile()

    def process_location(self, key: str, target_value: dict) -> float:
        """
        Process a single location.

        Args:
            key (str): The location key.
            target_value (dict): The target value for the location.

        Returns:
            float: The calculated distance.
        """
        # Load images
        image_dir = f'../data/locations/{key}/'
        image_inputs = prep_images(image_dir)

        # Define initial input
        text_input = [{"type": "text", "text": text_prompt}]
        inputs = {"messages": [HumanMessage(content=text_input + image_inputs)]}

        # Run the graph and log outputs
        for output in self.graph.stream(inputs):
            if "FINISH" not in str(output):
                prev_output = output
                logging.info(output)
                logging.info("----------------------------------")
                logging.info("----------------------------------")

        # Format the final output
        prompt_inputs = {"pred": str(prev_output), "json_prompt": JSON_PROMPT}
        sys_message = {"role": "system", "content": "Return valid json given input."}
        pred = call_openai(MODEL, sys_message, PRED_FORMAT_PROMPT_TEMPLATE, prompt_inputs)
        logging.info(f"Prediction: {pred}")

        # Calculate the distance
        distance = calculate_distance(pred, target_value)
        logging.info(f"Distance: {distance} km")

        return distance
