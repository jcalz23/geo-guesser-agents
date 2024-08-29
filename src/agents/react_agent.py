"""
React Agent for GeoGuessr location identification
"""
# Imports
import os
import sys
import logging
import getpass
from typing import Any, Dict, List, TypedDict, Union, Literal, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_community import GooglePlacesTool
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain.tools import StructuredTool

sys.path.append("..")
from utils.helpers import prep_images, call_openai, get_street_view_image
from utils.eval import calculate_distance
from constants import *
from prompts.react_agent import *


class State(TypedDict):
    """State for the React Agent."""
    input: str
    images: List
    plan: List[str]
    past_steps_results: List[Dict[str, Any]]
    num_steps_used: int
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class Response(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

class ReactAgentRunner:
    def __init__(self):
        self.setup_logging()
        self.setup_env_vars()
        self.setup_tools()
        self.setup_agents()
        self.graph = self.create_graph()

    def setup_tools(self):
        """Initialize the tools used by the agents."""
        self.tools = [
            TavilySearchResults(max_results=3),
            GooglePlacesTool(),
            StructuredTool.from_function(
                get_street_view_image,
                name="get_street_view_image",
                description="Get a Google Street View image for a given latitude and longitude. Input should be a dictionary with 'latitude' and 'longitude' keys."
            )
        ]

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

    def setup_agents(self):
        """Set up the language model."""
        # Define agent executor, planner, replanner chains
        llm = ChatOpenAI(model=REACT_MODEL, temperature=REACT_TEMPERATURE)
        self.agent_executor = create_react_agent(llm, self.tools, messages_modifier=REACT_PROMPT)
        self.planner = PLANNER_PROMPT | llm.with_structured_output(Plan)
        self.replanner = REPLANNER_PROMPT | llm.with_structured_output(Act)   

    def plan_step(self, state: State):
        """Plan step for the React Agent"""
        plan = self.planner.invoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps, "num_steps_used": state["num_steps_used"] + 1}

    def execute_step(self, state: State):
        """Execute step for the React Agent, where the agent is given a plan and executes the first step"""
        # Get the plan and format it
        plan = state["plan"]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

        # Get the first task and format it into a message
        task = plan[0]
        task_formatted = f"""Here is the plan: {plan_str}\n\nYour job is to execute step 1, {task}."""
        messages = [("user", task_formatted)]

        # Handle images in message
        if "use_images" in task:
            messages.append(("user", state["images"]))

        # Execute the step
        agent_response = self.agent_executor.invoke({"messages": messages})

        # Return the updated steps for next iteration
        step_result = {
            "num_steps_used": state["num_steps_used"] + 1
        }
        step_result["past_steps_results"] = state["past_steps_results"] + [{
            "step_num": state["num_steps_used"],
            "step": task,
            "result": agent_response["messages"][-1].content
        }]
        return step_result

    def replan_step(self, state: State):
        """Replan step for the React Agent"""
        # Replan
        output = self.replanner.invoke({
            "input": state["input"],
            "plan": state["plan"],
            "past_steps_results": state["past_steps_results"],
            "remaining_steps": REACT_RECURSION_LIMIT - state["num_steps_used"] - 1
        })

        # Determine if the output is a response or a plan
        if isinstance(output.action, Response):
            output = {"response": output.action.response}
        else:
            output = {"plan": output.action.steps, "num_steps_used": state["num_steps_used"] + 1}

        return output

    def should_end(self, state: State) -> Literal["agent", "__end__"]:
        """Determine if the agent should continue or end"""
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"

    def create_graph(self):
        """Create the graph for the React Agent"""
        # Define workflow
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("planner", self.plan_step)
        workflow.add_node("agent", self.execute_step)
        workflow.add_node("replan", self.replan_step)
        workflow.set_entry_point("planner")

        # Add edges
        workflow.add_edge("planner", "agent")
        workflow.add_edge("agent", "replan")
        workflow.add_conditional_edges("replan", self.should_end)

        # Compile
        app = workflow.compile()
        return app

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

        # Create initial input
        text_prompt = INITIAL_PROMPT_TEMPLATE.format(
            json_prompt=JSON_PROMPT, recursion_limit=REACT_RECURSION_LIMIT
            )
        inputs = {"input": text_prompt, "images": image_inputs, "num_steps_used": 0, "past_steps_results": []}

        # Run app with streaming
        for event in self.graph.stream(inputs, config={"recursion_limit": REACT_RECURSION_LIMIT}):
            for k, v in event.items():
                if k != "__end__":
                    if 'data:image/jpeg;base64' in str(v):
                        logging.info("Sent image to LLM for analysis")
                    else:
                        logging.info(v)
                    logging.info("----------------------------------")
                    logging.info("----------------------------------")

        # Format the final output
        prompt_inputs = {"pred": str(event), "json_prompt": JSON_PROMPT}
        sys_message = {"role": "system", "content": "Return valid json given input."}
        pred = call_openai(REACT_MODEL, sys_message, PRED_FORMAT_PROMPT_TEMPLATE, prompt_inputs)

        # Calculate the distance
        distance = calculate_distance(pred, target_value)
        logging.info(f"Distance: {distance} km")

        return distance
