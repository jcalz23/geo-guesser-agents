from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain import hub

# Json formatting prompt
JSON_PROMPT = "Only return a valid json string (RCF8259). Do provide any other commentary. Do not wrap the JSON in markdown such as ```json. Only use the data from the provided content."

# Initial user prompt
INITIAL_PROMPT_TEMPLATE = """USER: Given a set of streetview images from a vehicle, your task is to determine the coordinates from which the picture was taken. It can be anywhere in the world. You have a recursion limit of {recursion_limit}, so make sure to return a prediction before that is reached.

Return json with a single guess for the city and the coordinates following the below example. {json_prompt}
output={{"location": "City, State, Country", "latitude": "123.123123", "longitude": "456.456456"}}

AGENT: output="""


# Define prompt for react agent by using template and adding custom system message
REACT_PROMPT = hub.pull("wfh/react-agent-executor")
custom_system_message = SystemMessagePromptTemplate.from_template(
    """You are an AI assistant tasked with analyzing images and determining their location. 
    You have access to several tools to help you:
    1. TavilySearchResults: Use this to search for information online.
    2. GooglePlacesTool: Use this to find information about specific places.
    3. get_street_view_image: Use this to get a Google Street View image for a specific location. You need to provide latitude and longitude.

    When you need to use the get_street_view_image tool, first use the other tools to determine the latitude and longitude of the location you're interested in.

    Always think step-by-step and use the tools available to you to gather the necessary information before making a final determination."""
)
REACT_PROMPT.messages[0] = custom_system_message


# Planner prompts
PLANNER_SYS_MESSAGE = """For the given objective, come up with a simple step by step plan. This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 

If the task requires observing the initial images, include the string "(use_images)" at the end of the task but remain verbose and specific with the full task description.
{json_prompt}
"""

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", PLANNER_SYS_MESSAGE.format(json_prompt=JSON_PROMPT)),
        ("placeholder", "{messages}"),
    ]
)


# Replanner prompts
REPLANNER_INSTRUCTION = """For the given objective, come up with a simple step by step plan. This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. If the task requires observing the images, include the string "(use_images)" at the end of the task but remain verbose and specific with the full task description. The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""" 
REPLANNER_UPDATE = """Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.."""
REPLANNER_RETURN = """
You have reached the recursion limit and must return a response to user now. Follow the format in this example:
output={{"location": "City, State, Country", "latitude": "123.123123", "longitude": "456.456456"}}
"""
REPLANNER_PROMPT = ChatPromptTemplate.from_template(
    """{instruction}
    Your objective was this:
    {input}

    Your original plan was this:
    {plan}

    Here is a list of the previous steps and their results:
    {past_steps_results}

    You have {remaining_steps} steps remaining to return the final response.

    {next_step}
    """
    )

# Format the prediction
PRED_FORMAT_PROMPT_TEMPLATE = """USER: Given this LLM output with a prediction, convert it to the expected format defined below. Prediction: {pred} \
    Return json with a single guess for the city and the coordinates following the below example. {json_prompt} \
        - output={{"city": "Orland Park, IL, USA", "latitude": "42.0099", "longitude": "-87.62317"}}

AGENT: output="""
