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

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from src.inference import InferenceRunner, get_agent

# Initialize logging
log_file = 'app.log'
if os.path.exists(log_file):
    os.remove(log_file)  # Remove existing log file

# Configure root logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(message)s'))
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

# Get logger for this module
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Define upload directory
UPLOAD_DIR = os.path.join(parent_dir, "app/uploads")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(parent_dir, "app/templates"))

def reset_upload_dir():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

@app.get("/")
async def root(request: Request):
    """Root endpoint for the app"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload images to the app"""
    logger.info("-----UPLOAD-----")
    # Reset the upload directory before new uploads
    reset_upload_dir()
    logger.info(f"Upload directory reset: {UPLOAD_DIR}")
    
    # Upload files
    filenames = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        logger.info(f"Uploading file: {file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        filenames.append(file.filename)
    logger.info(f"Upload process completed. Total files uploaded: {len(filenames)}")
    return {"filenames": filenames}

@app.post("/predict")
async def predict(agent_name: str = Form(...)):
    """Predict the location of the images"""
    try:
        selected_agent = get_agent(agent_name)
        runner = InferenceRunner(selected_agent, agent_name, None, UPLOAD_DIR, None, None)
        result = runner.main()
        return Response(status_code=204)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream_logs")
async def stream_logs():
    """Stream the logs from the app"""
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
    """Clear the logs from the app"""
    with open('app.log', 'w'):
        pass  # This will clear the contents of the log file
    logger.info("Logs cleared")
    return JSONResponse(content={"status": "Logs cleared"})
