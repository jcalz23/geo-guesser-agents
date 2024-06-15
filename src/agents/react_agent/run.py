# Imports
import os
import sys
import json
import getpass
from datetime import datetime
from typing import Annotated, Any, Dict, List, Tuple, Optional, Sequence, TypedDict, Union, Literal
import operator

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import GooglePlacesTool
from langchain import hub
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain.tools import StructuredTool

sys.path.append('../../')
from utils.helpers import prep_images, call_openai, get_street_view_image
from utils.eval import calculate_distance
from utils.constants import *
from prompts import *

# Set env vars
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
if "LANGCHAIN_API_KEY" not in os.environ:
    os.environ['LANGCHAIN_API_KEY'] = getpass.getpass("Enter your Langchain API key: ")
if "TAVILY_API_KEY" not in os.environ:
    os.environ['TAVILY_API_KEY'] = getpass.getpass("Enter your Tavily API key: ")
if "OPENAI_API_KEY" not in os.environ:
    os.environ['OPENAI_API_KEY'] = getpass.getpass("Enter your OpenAI API key: ")
if "GPLACES_API_KEY" not in os.environ:
    os.environ["GPLACES_API_KEY"] = getpass.getpass("Enter your Google Places API key: ")

# Constants
DATA_DIR = "../../../data/"
MODEL = "gpt-4o"
CONFIG = {"recursion_limit": 20}

# Define tools
search_tool = TavilySearchResults(max_results=3)
places_tool = GooglePlacesTool()
street_view = StructuredTool.from_function(get_street_view_image)
tools = [search_tool, places_tool, street_view]

# Define LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent_executor = create_react_agent(llm, tools, messages_modifier=REACT_PROMPT)

# Define State
class PlanExecute(TypedDict):
    input: str
    images: List
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

# Define Plan
class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

# Define Response class
class Response(BaseModel):
    """Response to user."""
    response: str

# Define Act class
class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

# Define planner chain
planner = PLANNER_PROMPT | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Plan)

# Define replanner chain
replanner = REPLANNER_PROMPT | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)

# Plan step
def plan_step(state: PlanExecute):
    plan = planner.invoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}

# Execute step
def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    messages = [("user", task_formatted)]
    if "use_images" in task:
        messages.append(("user", state["images"]))
    agent_response = agent_executor.invoke(
        {"messages": messages}
    )
    return {
        "past_steps": (task, agent_response["messages"][-1].content),
    }

# Replan step
def replan_step(state: PlanExecute):
    output = replanner.invoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}

# Determine if the workflow should end
def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
    if "response" in state and state["response"]:
        return "__end__"
    else:
        return "agent"

def create_graph():
    # Define workflow
    workflow = StateGraph(PlanExecute)

    # Add nodes
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.set_entry_point("planner")

    # Add edges
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")
    workflow.add_conditional_edges(
        "replan",
        should_end,
    )

    # Compile
    app = workflow.compile()
    return app

# Main
def main():
    # Load target locations
    with open(f'{DATA_DIR}master.json', 'r') as f:
        target_locations = json.load(f)

    # Save results to working dir with a runtime value
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    results_file_path = f'{DATA_DIR}/results/runs/react_agent_{current_time}.json'

    # Create graph
    app = create_graph()

    # Loop through locations
    all_results = {}
    for key, value in target_locations.items():
        print(f"Processing {key}")
        try:
            # Load images
            image_dir = f'{DATA_DIR}locations/{key}/'
            image_inputs = prep_images(image_dir)

            # Run app with streaming
            inputs = {"input": text_prompt, "images": image_inputs}
            for event in app.stream(inputs, config=CONFIG):
                for k, v in event.items():
                    if k != "__end__":
                        print(v)

            # Format output
            prompt_inputs = {"pred": str(event), "json_prompt": JSON_PROMPT}
            sys_message = {"role": "system", "content": "Return valid json given input."}
            pred = call_openai(MODEL, sys_message, PRED_FORMAT_PROMPT_TEMPLATE, prompt_inputs)
            print(f"pred: {pred}")

            ## Evals
            # Calculate distance
            target = target_locations[key]
            distance = calculate_distance(pred, target)
            print(f"Distance: {distance} km")

            # Save results
            all_results[key] = distance

            # Save results to working dir with a runtime value
            with open(results_file_path, 'w') as file:
                json.dump(all_results, file)
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue

if __name__=="__main__":
    main()
