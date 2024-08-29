import logging
import shutil
import os
import sys
from typing import List
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import asyncio
from fastapi import HTTPException

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
    logger.info("Starting image upload process")
    # Reset the upload directory before new uploads
    reset_upload_dir()
    logger.info(f"Upload directory reset: {UPLOAD_DIR}")
    
    filenames = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        logger.info(f"Uploading file: {file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        filenames.append(file.filename)
        logger.info(f"File uploaded successfully: {file.filename}")
    
    logger.info(f"Upload process completed. Total files uploaded: {len(filenames)}")
    return {"filenames": filenames}

@app.post("/predict")
async def predict(agent_name: str = Form(...)):
    logger.info(f"Received prediction request for agent: {agent_name}")
    try:
        selected_agent = get_agent(agent_name)
        runner = InferenceRunner(selected_agent, agent_name, None, UPLOAD_DIR, None, None)
        result = runner.main()
        logger.info(f"Prediction completed. Result: {result}")
        return Response(status_code=204)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream_logs")
async def stream_logs():
    async def log_generator():
        with open('app.log', 'r') as log_file:
            # Move to the end of the file
            log_file.seek(0, 2)
            while True:
                line = log_file.readline()
                if not line:
                    await asyncio.sleep(0.1)
                    continue
                yield f"data: {line}\n\n"

    return StreamingResponse(log_generator(), media_type="text/event-stream")

@app.post("/clear_logs")
async def clear_logs():
    with open('app.log', 'w'):
        pass  # This will clear the contents of the log file
    logger.info("Logs cleared")
    return JSONResponse(content={"status": "Logs cleared"})
