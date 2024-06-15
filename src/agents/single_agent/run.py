import os
import sys
import json
import getpass
from datetime import datetime
from typing import TypedDict, Annotated, Sequence

import operator
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import GooglePlacesTool
from langchain_core.messages import HumanMessage
from langchain.tools import StructuredTool

sys.path.append('../../')
from utils.helpers import call_openai, prep_images, get_street_view_image
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

# Define tools
search_tool = TavilySearchResults(max_results=3)
places_tool = GooglePlacesTool()
street_view = StructuredTool.from_function(get_street_view_image)
tools = [search_tool, places_tool, street_view]

# Define LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)
model = model.bind_tools(tools)

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def create_graph():
    # Define the function to execute tools
    tool_node = ToolNode(tools)

    # Define the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", # Start with agent
        should_continue, # Decide what to call next or end
        {
            "continue": "action",
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")
    app = workflow.compile()

    return app

# Main
def main():
    # Load target locations
    with open(f'{DATA_DIR}master.json', 'r') as f:
        target_locations = json.load(f)

    # Define results file path
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    results_file_path = f'{DATA_DIR}/results/runs/single_agent_new_{current_time}.json'

    # Loop through locations
    all_results = {}
    for key, value in target_locations.items():
        print(f"Processing {key}")
        try:
            # Load images
            image_dir = f'{DATA_DIR}locations/{key}/'
            image_inputs = prep_images(image_dir)

            # Run app with streaming
            app = create_graph()
            inputs = {"messages": [HumanMessage(content=text_input + image_inputs)]}
            for output in app.stream(inputs):
                for k, v in output.items():
                    print(f"Output from node '{k}':")
                print("---")
                print(v)
            print("\n---\n")

            # Format output
            prompt_inputs = {"pred": str(output), "json_prompt": JSON_PROMPT}
            sys_message = {"role": "system", "content": "Return valid json given input."}
            pred = call_openai(MODEL, sys_message, PRED_FORMAT_PROMPT_TEMPLATE, prompt_inputs)
            print(f"Prediction: {pred}")

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
            print(f"Error: {e}")
            continue

if __name__=="__main__":
    main()
