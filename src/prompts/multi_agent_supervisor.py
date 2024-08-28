SUPERVISOR_SYS_PROMPT = """You are a supervisor tasked with managing a conversation between the following workers:  {members}. Given the following user request,
respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH. \
    Only return a valid json string (RCF8259). Do provide any other commentary. Do not wrap the JSON in markdown such as ```json. Only use the data from the provided content."""
SUPERVISOR_USER_PROMPT = """Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"""

JSON_PROMPT = "Only return a valid json string (RCF8259). Do provide any other commentary. Do not wrap the JSON in markdown such as ```json. Only use the data from the provided content."

PROMPT_TEMPLATE = """USER: Given a set of streetview images from a vehicle, your task is to determine the
coordinates from which the picture was taken. It can be anywhere in the world.

Before you submit your final prediction, use the Google Street View tool to get an image of the location you are predicting and compare it to the input images. If the streetview image doesn't match the input image, keep searching.

Return json with the city and coordinates following the below example. {json_prompt}
- {{"city": "Orland Park, IL, 60467, USA", "latitude": "42.0099", "longitude": "-87.62317"}}

AGENT: """
text_prompt = PROMPT_TEMPLATE.format(json_prompt=JSON_PROMPT)

PRED_FORMAT_PROMPT_TEMPLATE = """USER: Given this LLM output with a prediction, convert it to the expected format defined below. Prediction: {pred} \
    Return json with a single guess for the city and the coordinates following the below example. {json_prompt} \
        - output={{"city": "Orland Park, IL, USA", "latitude": "42.0099", "longitude": "-87.62317"}}

AGENT: output="""