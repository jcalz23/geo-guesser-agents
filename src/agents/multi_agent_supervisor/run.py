import os
import sys
import json
import getpass
import operator
import functools
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import GooglePlacesTool
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
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
members = ["Researcher", "Map_Searcher"]

# Define tools
search_tool = TavilySearchResults(max_results=3)
places_tool = GooglePlacesTool()
street_view = StructuredTool.from_function(get_street_view_image)
tools = [search_tool, places_tool, street_view]

# Helper Fns
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

def create_supervisor_chain(llm):
    # Supervisor picks next agent to process or to end
    options = ["FINISH"] + members
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
    supervisor_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_SYS_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            SUPERVISOR_USER_PROMPT,
        ]
    ).partial(options=str(options), members=", ".join(members))

    # Define chain
    supervisor_chain = (
        supervisor_prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    return supervisor_chain

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def create_graph():
    # Define llm
    llm = ChatOpenAI(model=MODEL, temperature=0)

    # Researcher
    research_agent = create_agent(llm, [search_tool], "You are a web researcher.")
    research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

    # Map_Searcher
    map_searcher_agent = create_agent(llm, [places_tool], "You are a map searcher.")
    map_searcher_node = functools.partial(agent_node, agent=map_searcher_agent, name="Map_Searcher")

    # Init supervisor chain
    supervisor_chain = create_supervisor_chain(llm)

    # Create graph
    workflow = StateGraph(AgentState)
    workflow.add_node("Researcher", research_node)
    workflow.add_node("Map_Searcher", map_searcher_node)
    workflow.add_node("supervisor", supervisor_chain)

    # Eeach worker reports back to supervisor when done
    for member in members:
        workflow.add_edge(member, "supervisor")

    # The supervisor populates the "next" field in the graph state
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

    # Add entrypoint and compile
    workflow.set_entry_point("supervisor")
    graph = workflow.compile()

    return graph

# Main
def main():
    # Load target locations
    with open(f'{DATA_DIR}master.json', 'r') as f:
        target_locations = json.load(f)

    # Save results to working dir with a runtime value
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    results_file_path = f'{DATA_DIR}/results/runs/multi_agent_supervisor_{current_time}.json'

    # Loop through locations
    all_results = {}
    for key, value in target_locations.items():
        try:
            print(f"Processing {key}")
            #try:
            # Load images
            image_dir = f'{DATA_DIR}locations/{key}/'
            image_inputs = prep_images(image_dir)
            text_input = [{"type": "text", "text": text_prompt}]
            inputs = {"messages": [HumanMessage(content=text_input + image_inputs)]}

            # Run graph
            graph = create_graph()
            for output in graph.stream(inputs):
                if "FINISH" not in str(output):
                    prev_output = output
                    print(output)
                    print("----")
            
            # Format output
            prompt_inputs = {"pred": str(prev_output), "json_prompt": JSON_PROMPT}
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
            print(f"Error processing {key}: {e}")
            continue

if __name__=="__main__":
    main()
