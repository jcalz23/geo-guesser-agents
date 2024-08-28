# Shared prompts
SYS_MESSAGE_PROMPT = """You are an expert at the game GeoGuessr. \
    You are trying to determine the location of an image in world coordinates."""
SYS_MESSAGE = {"role": "system", "content": SYS_MESSAGE_PROMPT}
JSON_PROMPT = """Only return a valid json string (RCF8259). Do provide any other commentary. \
    Do not wrap the JSON in markdown such as ```json. Only use the data from the provided content."""

# Inidividual prompts
DESCRIBE_TEXT_PROMPT = """USER: Your task is to describe this image in detail. Here are some instructions:
- Focus entirely on the text found on the image that might provide location clues.
- Prioritize street signs, stores, landmarks, and other specific features that provide location clues.
- Pay attention to any public transportation, which may provide great local information.
- Ignore advertisements unless they contain local information.
- Ignore the text related to the Geoguesser game.
- Sort the items based on how useful the information might be for someone trying to determine where in the world the image was taken.
- Return a json following the below examples. {json_prompt}
  - output={{"item1_name": "item1_description", "item2_name": "item2_description", ...}}
  - output={{"restaurant": "There is a restaurant displaying 'Stevens Pizza of Oakmont", "street_sign": "One street sign says Orange Ave"}}

AGENT: output="""

DESCRIBE_SCENE_PROMPT = """USER: Your task is to describe this image in detail. Here are some instructions:
- Focus entirely on the scenery the image, whether architecture, nature, or people.
- Ignore the text related to the Geoguesser game.
- Sort the items based on how useful the information might be for someone trying to determine where in the world the image was taken.
- Return a json following the below examples. {json_prompt}
  - output={{"item1_name": "item1_description", "item2_name": "item2_description", ...}}
  - output={{"mountain": "There is a large mountain range in the background of the image with white snowcaps", "trees": "There are Sequioa trees on the mountain", "architecture": "There is a Victorian building in the image", "people": "There are people in the image wearing American clothing"}}

AGENT: output="""

CANDIDATES_PROMPT = """USER: A previous agent has extracted features from a set of four images facing North, South, East, and West.
Your task is to interpret these features and attempt to guess which city the image is from. Here is some helpful information and instruction:
*Information*
- text_results: {text_results}
- scene_results: {scene_results}

*Instructions*
- Provide educated guesses on which city the image may be from.
- Return your top few guesses and their confidence levels from 0-100.
- Return a json following the below examples. {json_prompt}
  - output={{"Tokyo": {{"confidence": 0.7, "reasoning": "The image features a singular large mountain in the background, which could be Mt. Fuji but it is hard to determine with confidence."}}}}

AGENT: output="""

SEARCH_PROMPT = """USER: A previous agent has extracted features from an image. 
Your task is to come up with a set of Google searches based on the features extraction from the image. Here is some helpful information and instruction:
*Information*
- text_results: {text_results}
- scene_results: {scene_results}
- city_candidates: {candidates}

*Instructions*
- Return Google search queries that are likely to identify a specific address.
- Ensure that search queries are specific and not too vague.
- Focus primarily on landmarks and locations with names. Avoid queries that describe the scenery or generic objects like a FedEx truck.
- Return a json following the below examples. {json_prompt}
  - output={{"search1": "WalMart Steele Street", "search2": "F37 bus route in Germany", ...}}

AGENT: output="""

GOOGLE_SEARCH_DESCRIPTION = "Find the location associated with this landmark"

GEO_GUESSER_PROMPT = """USER: A previous agent has extracted features from an image. 
Your task is to determine the coordinates from which the picture was taken. It can be anywhere in the world. Here is some helpful information and instruction:
*Information*
- text_results: {text_results}
- scene_results: {scene_results}
- city_candidates: {candidates}
- search_results: {search_results}

*Instructions*
You must return a coordinate prediction. Don't return "unknown" or any other text besides coordinates.
Return json with the city and coordinates following the below example. {json_prompt}
output={{"city": "Orland Park, IL, 60467, USA", "latitude": "42.0099", "longitude": "-87.62317"}}

AGENT: output="""