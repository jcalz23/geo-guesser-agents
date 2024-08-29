import logging
import shutil
import os
import sys
from typing import List
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import InferenceRunner, get_agent

# Initialize logging
log_file = 'app.log'
if os.path.exists(log_file):
    os.remove(log_file)  # Remove existing log file

# Configure root logger
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

# Get logger for this module
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Define upload directory
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

def reset_upload_dir():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    # Reset the upload directory before new uploads
    reset_upload_dir()
    
    filenames = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        filenames.append(file.filename)
    return {"filenames": filenames}

@app.post("/predict")
async def predict(agent_name: str = Form(...)):
    logger.info(f"Received prediction request for agent: {agent_name}")
    selected_agent = get_agent(agent_name)
    runner = InferenceRunner(selected_agent, agent_name, None, UPLOAD_DIR, None, None)
    result = runner.main()
    logger.info(f"Prediction completed. Result: {result}")
    return JSONResponse(content={"status": result})

@app.get("/stream_logs")
async def stream_logs():
    async def log_generator():
        with open('app.log', 'r') as log_file:
            while True:
                line = log_file.readline()
                if not line:
                    await asyncio.sleep(0.1)
                    continue
                yield f"data: {line}\n\n"

    return StreamingResponse(log_generator(), media_type="text/event-stream")
