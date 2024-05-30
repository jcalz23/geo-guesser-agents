import os
import json
import base64
import requests
import getpass

from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper

from utils.constants import *






# Get env vars
os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")
os.environ["GOOGLE_CSE_ID"] = getpass.getpass("Enter your Google CSE ID: ")








def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def prep_images(id):
    """
    Loads images and prepares them for OpenAI API call
    """
    image_dir = f'./data/{id}/'
    image_inputs = []
    for direction in directions:
        base64_img = encode_image(f"{image_dir}{direction}.png")
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