from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

REACT_PROMPT = hub.pull("wfh/react-agent-executor")

PLANNER_SYS_MESSAGE = """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
        If the task requires observing the images, include the string "(use_images)" at the end of the task but remain verbose and specific with the full task description. \
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps. \
                Return json with a single guess for the city and the coordinates following the below example. \
                    Only return a valid json string (RCF8259). Do provide any other commentary. Do not wrap the JSON in markdown such as ```json. Only use the data from the provided content. \
                        - output={{"city": "Orland Park, IL, USA", "latitude": "42.0099", "longitude": "-87.62317"}}"""
PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", PLANNER_SYS_MESSAGE),
        ("placeholder", "{messages}"),
    ]
)

REPLANNER_PROMPT = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
        This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
            If the task requires observing the images, include the string "(use_images)" at the end of the task but remain verbose and specific with the full task description. \
                The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    Your objective was this:
    {input}

    Your original plan was this:
    {plan}

    You have currently done the follow steps:
    {past_steps}

    You have a max step limit of 10.

    Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. \
        Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
    )

JSON_PROMPT = "Only return a valid json string (RCF8259). Do provide any other commentary. Do not wrap the JSON in markdown such as ```json. Only use the data from the provided content."
INITIAL_PROMPT_TEMPLATE = """USER: Given a set of streetview images from a vehicle, your task is to determine the \
coordinates from which the picture was taken. It can be anywhere in the world. You have a recursion limit of 10, so make sure to return a guess before that is reached.

Before you submit your final prediction, use the Google Street View tool to get an image of the location you are predicting and compare it to the input images. If the streetview image doesn't match the input image, keep searching.

Return json with a single guess for the city and the coordinates following the below example. {json_prompt}
output={{"city": "Orland Park, IL, USA", "latitude": "42.0099", "longitude": "-87.62317"}}

AGENT: output="""
text_prompt = INITIAL_PROMPT_TEMPLATE.format(json_prompt=JSON_PROMPT)

PRED_FORMAT_PROMPT_TEMPLATE = """USER: Given this LLM output with a prediction, convert it to the expected format defined below. Prediction: {pred} \
    Return json with a single guess for the city and the coordinates following the below example. {json_prompt} \
        - output={{"city": "Orland Park, IL, USA", "latitude": "42.0099", "longitude": "-87.62317"}}

AGENT: output="""
