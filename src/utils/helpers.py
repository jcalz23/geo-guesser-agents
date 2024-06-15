import os
import json
import base64
import requests
import getpass
from PIL import Image
from io import BytesIO

from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper

from utils.constants import *

# Get env vars
if "GOOGLE_API_KEY" not in os.environ:
    os.environ['GOOGLE_API_KEY'] = getpass.getpass("Enter your Google API key: ")
if "GOOGLE_CSE_ID" not in os.environ:
    os.environ["GOOGLE_CSE_ID"] = getpass.getpass("Enter your Google Search API key: ")


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def prep_images(image_id_dir):
    """
    Loads images and prepares them for OpenAI API call
    """
    image_inputs = []
    for direction in directions:
        base64_img = encode_image(f"{image_id_dir}{direction}.png")
        image_input = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
        image_inputs.append(image_input)

    return image_inputs

def call_openai(model, sys_message, text_prompt, prompt_inputs, image_inputs=None, max_tokens=300):
    # Define payload
    text_input = [{"type": "text", "text": text_prompt.format(**prompt_inputs)}]
    if image_inputs:
        messages = text_input + image_inputs
    else:
        messages = text_input
    payload = {
        "model": model,
        "messages": [sys_message, {"role": "user", "content": messages}],
        "max_tokens": max_tokens
    }

    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    result = response.json()['choices'][0]['message']['content']

    return json.loads(result)


# Google Search API
def top_n_results(query):
    search = GoogleSearchAPIWrapper()
    return search.results(query, TOP_N_SEARCHES)

def get_top_n_results(description, query):
    top_n_search_tool = Tool(
        name="google_search",
        description=description,
        func=top_n_results,
    )
    search_results = top_n_search_tool.run(query)
    return search_results

def get_street_view_image(latitude: float, longitude: float, heading: int=90):
    """
    Fetches a Street View image from Google Street View API.
    
    Parameters:
        latitude (str): Latitude of the location (e.g., 40.689247).
        longitude (str): Longitude of the location (e.g., -74.044502).
        heading (int): Direction of the camera in degrees (0-360).
    
    Returns:
        Image: PIL Image object of the Street View image.
    """
    size = "200x100"
    fov = 90
    pitch = 0
    url = f"https://maps.googleapis.com/maps/api/streetview?size={size}&location={float(latitude)},{float(longitude)}&heading={heading}&pitch={pitch}&fov={int(fov)}&key={os.environ["GPLACES_API_KEY"]}"
    response = requests.get(url)
    
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_str}"
        return data_url
    else:
        raise Exception("Failed to fetch Street View image")