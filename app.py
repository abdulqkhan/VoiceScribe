import os
import sys
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests
import tempfile
import boto3
from botocore.client import Config
import whisper
import logging
import subprocess
import threading
import uuid
import torch
from urllib.error import URLError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure logging to console
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Starting application...")

# Load environment variables from .env file
load_dotenv()

logger.info("Environment variables loaded.")

app = Flask(__name__)

S3_ENDPOINT = os.getenv('S3_ENDPOINT')
S3_BUCKET = os.getenv('S3_BUCKET')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')

logger.info(f"S3_ENDPOINT: {S3_ENDPOINT}")
logger.info(f"S3_BUCKET: {S3_BUCKET}")
logger.info(f"S3_ACCESS_KEY: {'*' * len(S3_ACCESS_KEY) if S3_ACCESS_KEY else 'Not set'}")
logger.info(f"S3_SECRET_KEY: {'*' * len(S3_SECRET_KEY) if S3_SECRET_KEY else 'Not set'}")

if not all([S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY]):
    logger.error("One or more required environment variables are not set!")
    raise ValueError("Missing required environment variables")

try:
    s3_client = boto3.client('s3',
                             endpoint_url=S3_ENDPOINT,
                             aws_access_key_id=S3_ACCESS_KEY,
                             aws_secret_access_key=S3_SECRET_KEY,
                             config=Config(signature_version='s3v4'))
    logger.info("S3 client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {str(e)}")
    raise

# In-memory job storage
jobs = {}

def get_public_url(filename):
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{filename}"

def upload_file(file_path, object_name):
    try:
        logger.info(f"Uploading file {file_path} to S3 as {object_name}")
        s3_client.upload_file(file_path, S3_BUCKET, object_name)
        logger.info(f"Upload successful: {object_name}")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise

def format_timestamp(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"

def process_video(job_id, video_filename):
    try:
        video_url = get_public_url(video_filename)
        logger.info(f"Starting process for video: {video_url}")
        
        # Stream and convert video to MP3 using FFmpeg
        mp3_filename = f"{os.path.splitext(video_filename)[0]}.mp3"
        mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
        
        logger.info(f"Converting video to MP3: {mp3_path}")
        ffmpeg_command = [
            'ffmpeg',
            '-i', video_url,
            '-vn',  # Disable video
            '-acodec', 'libmp3lame',
            '-ar', '44100',
            '-ab', '192k',
            '-y',  # Overwrite output file if it exists
            mp3_path
        ]
        
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg error: {stderr.decode()}")
        
        logger.info("MP3 conversion complete")
        
        # Upload MP3 to S3
        logger.info(f"Uploading MP3 to S3: {mp3_filename}")
        upload_file(mp3_path, mp3_filename)
        
        # Load Whisper model
        logger.info("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        try:
            model = whisper.load_model("base", device=device)
            logger.info(f"Whisper model loaded successfully on {device}")
        except URLError as e:
            logger.error(f"Network error while loading Whisper model: {str(e)}")
            raise Exception("Failed to download Whisper model due to network error. Please check your internet connection and try again.")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise Exception(f"Failed to load Whisper model: {str(e)}")

        # Transcribe audio
        logger.info("Transcribing audio...")
        result = model.transcribe(mp3_path)
        logger.info("Transcription complete")
        
        # Process transcription with timestamps
        transcription_with_timestamps = []
        for segment in result["segments"]:
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"]
            transcription_with_timestamps.append(f"[{start_time} - {end_time}] {text}")
        
        # Join the transcription segments
        full_transcription = "\n".join(transcription_with_timestamps)
        
        # Upload transcription to S3
        transcription_filename = f"{os.path.splitext(video_filename)[0]}_transcription.txt"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_transcription_file:
            temp_transcription_file.write(full_transcription)
            temp_transcription_path = temp_transcription_file.name
        
        logger.info(f"Uploading transcription to S3: {transcription_filename}")
        upload_file(temp_transcription_path, transcription_filename)
        
        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        os.unlink(mp3_path)
        os.unlink(temp_transcription_path)
        logger.info("Temporary files cleaned up")

        jobs[job_id] = {
            'status': 'completed',
            'result': {
                'message': 'Video converted, transcribed, and uploaded successfully',
                'mp3_url': get_public_url(mp3_filename),
                'transcription_url': get_public_url(transcription_filename)
            }
        }
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        jobs[job_id] = {'status': 'failed', 'error': str(e)}

@app.route('/convert_and_transcribe', methods=['POST'])
def convert_and_transcribe():
    logger.info("Convert and transcribe endpoint called.")
    video_filename = request.json.get('video_filename')
    if not video_filename:
        logger.error("No video filename provided")
        return jsonify({'error': 'No video filename provided'}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'processing'}
    
    # Start the processing in a new thread
    thread = threading.Thread(target=process_video, args=(job_id, video_filename))
    thread.start()
    
    return jsonify({
        'message': 'Task started successfully',
        'job_id': job_id
    }), 202  # 202 Accepted

@app.route('/job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job['status'] == 'completed':
        return jsonify(job['result']), 200
    elif job['status'] == 'failed':
        return jsonify({'error': job['error']}), 500
    else:
        return jsonify({'status': 'processing'}), 202

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check endpoint called.")
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000)
else:
    logger.info("Flask application imported, not running directly.")