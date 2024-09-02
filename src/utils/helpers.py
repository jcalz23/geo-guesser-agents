import os
import sys
import json
import base64
import logging
import requests
from PIL import Image
from io import BytesIO
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
sys.path.append('../')
from src.constants import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compress_image(img, max_size=(800, 800), quality=85):
    """
    Compresses image to reduce token count

    Parameters:
        img (PIL.Image): Image to compress
        max_size (tuple): Maximum size of image
        quality (int): Quality of image

    Returns:
        str: Compressed image
    """
    # Compress image to reduce token count
    img.thumbnail(max_size)
    if img.mode in ('RGBA', 'LA'):
        background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
        background.paste(img, img.split()[-1])
        img = background
    buffer = BytesIO()
    img.convert('RGB').save(buffer, format='JPEG', quality=quality, optimize=True)
    return buffer.getvalue()

def crop_image(img, crop_percentage=0.8):
    """
    Crops an image to the specified percentage, keeping the center portion.

    Parameters:
        img (PIL.Image): Image to crop
        crop_percentage (float): Percentage of original image to keep (0.8 = 80%)

    Returns:
        PIL.Image: Cropped image
    """
    width, height = img.size
    crop_width = int(width * crop_percentage)
    crop_height = int(height * crop_percentage)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return img.crop((left, top, right, bottom))

def prep_images(image_id_dir, max_size=(512, 512), quality=85, crop_percentage=0.8):
    """
    Loads images, crops them, compresses them, and prepares them for OpenAI API call

    Parameters:
        image_id_dir (str): Directory of images
        max_size (tuple): Maximum size of image after compression
        quality (int): Quality of compressed image
        crop_percentage (float): Percentage of original image to keep (0.8 = 80%)

    Returns:
        list: List of image inputs
    """
    image_inputs = []
    for image_file in os.listdir(image_id_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_id_dir, image_file)
            with Image.open(image_path) as img:
                # Crop the image
                cropped_img = crop_image(img, crop_percentage)

                # Compress the cropped image
                compressed_img = compress_image(cropped_img, max_size, quality)
                # cropped_img.thumbnail(max_size)
                # if cropped_img.mode in ('RGBA', 'LA'):
                #     background = Image.new(cropped_img.mode[:-1], cropped_img.size, (255, 255, 255))
                #     background.paste(cropped_img, cropped_img.split()[-1])
                #     cropped_img = background
                # buffer = BytesIO()
                # cropped_img.convert('RGB').save(buffer, format='JPEG', quality=quality, optimize=True)
                # compressed_img = buffer.getvalue()

            encoded_img = base64.b64encode(compressed_img).decode('utf-8')
            image_input = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}
            image_inputs.append(image_input)

    return image_inputs

def call_openai(model, sys_message, text_prompt, prompt_inputs, image_inputs=None, max_tokens=300):
    """
    Calls OpenAI API

    Parameters:
        model (str): Model to use
        sys_message (str): System message
        text_prompt (str): Text prompt
        prompt_inputs (dict): Prompt inputs
        image_inputs (list): List of image inputs
        max_tokens (int): Maximum number of tokens

    Returns:
        str: Response from OpenAI API
    """
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
    try:
        return json.loads(result)
    except Exception as e:
        logger.info(f"Error parsing JSON: {e}", exc_info=True)
        return result

def top_n_results(query):
    """
    Fetches top N search results from Google Search API

    Parameters:
        query (str): Query to search for

    Returns:
        str: Top N search results
    """
    search = GoogleSearchAPIWrapper()
    return search.results(query, TOP_N_SEARCHES)

def get_top_n_results(description, query):
    """
    Fetches top N search results from Google Search API

    Parameters:
        description (str): Description of tool
        query (str): Query to search for

    Returns:
        Tool: Tool object
    """
    top_n_search_tool = Tool(
        name="google_search",
        description=description,
        func=top_n_results,
    )
    search_results = top_n_search_tool.run(query)
    return search_results

def get_street_view_image(latitude: float, longitude: float, heading: int=90):
    """
    Fetches a Street View image from Google Street View API and compresses it.
    
    Parameters:
        latitude (str): Latitude of the location (e.g., 40.689247).
        longitude (str): Longitude of the location (e.g., -74.044502).
        heading (int): Direction of the camera in degrees (0-360).

    Returns:
        str: Base64 encoded compressed Street View image.
    """
    size = "200x200"
    fov = 90
    pitch = 0
    url = f"https://maps.googleapis.com/maps/api/streetview?size={size}&location={float(latitude)},{float(longitude)}&heading={heading}&pitch={pitch}&fov={int(fov)}&key={os.environ['GPLACES_API_KEY']}"
    response = requests.get(url)
    
    if response.status_code == 200:
        # Compress the image
        image = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        image.save(buffered, format="JPEG", optimize=True, quality=85)
        compressed_img = buffered.getvalue()

        # Encode the image
        img_str = base64.b64encode(compressed_img).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_str}"
        return data_url
    else:
        raise Exception("Failed to fetch Street View image")