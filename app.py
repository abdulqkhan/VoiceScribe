import os
from flask import Flask, request, jsonify
from pydub import AudioSegment
import requests
import tempfile
from dotenv import load_dotenv
import traceback
import boto3
from botocore.client import Config
import whisper
import logging

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S3_ENDPOINT = os.getenv('S3_ENDPOINT')
S3_BUCKET = os.getenv('S3_BUCKET')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')

logger.info(f"S3_ENDPOINT: {S3_ENDPOINT}")
logger.info(f"S3_BUCKET: {S3_BUCKET}")

s3_client = boto3.client('s3',
                         endpoint_url=S3_ENDPOINT,
                         aws_access_key_id=S3_ACCESS_KEY,
                         aws_secret_access_key=S3_SECRET_KEY,
                         config=Config(signature_version='s3v4'))

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

@app.route('/convert_and_transcribe', methods=['POST'])
def convert_and_transcribe():
    video_filename = request.json.get('video_filename')
    if not video_filename:
        return jsonify({'error': 'No video filename provided'}), 400

    video_url = get_public_url(video_filename)

    try:
        logger.info(f"Starting process for video: {video_url}")
        
        # Download video from public URL
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_filename)[1]) as temp_video_file:
            logger.info(f"Downloading video from: {video_url}")
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                temp_video_file.write(chunk)
            temp_video_path = temp_video_file.name
        logger.info(f"Video downloaded to: {temp_video_path}")

        # Convert to MP3
        logger.info("Converting video to MP3...")
        audio = AudioSegment.from_file(temp_video_path)
        mp3_filename = f"{os.path.splitext(video_filename)[0]}.mp3"
        mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
        audio.export(mp3_path, format="mp3")
        logger.info(f"MP3 conversion complete: {mp3_path}")
        
        # Upload MP3 to S3
        logger.info(f"Uploading MP3 to S3: {mp3_filename}")
        upload_file(mp3_path, mp3_filename)
        
        # Load Whisper model
        logger.info("Loading Whisper model...")
        model = whisper.load_model("base")
        logger.info("Whisper model loaded")

        # Transcribe audio
        logger.info("Transcribing audio...")
        result = model.transcribe(mp3_path)
        logger.info("Transcription complete")
        
        # Upload transcription to MinIO
        transcription_filename = f"{os.path.splitext(video_filename)[0]}_transcription.txt"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_transcription_file:
            temp_transcription_file.write(result["text"])
            temp_transcription_path = temp_transcription_file.name
        logger.info(f"Transcription saved to: {temp_transcription_path}")

        logger.info(f"Uploading transcription to S3: {transcription_filename}")
        upload_file(temp_transcription_path, transcription_filename)
        
        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        os.unlink(temp_video_path)
        os.unlink(mp3_path)
        os.unlink(temp_transcription_path)
        logger.info("Temporary files cleaned up")

        return jsonify({
            'message': 'Video converted, transcribed, and uploaded successfully',
            'mp3_url': get_public_url(mp3_filename),
            'transcription_url': get_public_url(transcription_filename)
        }), 200

    except requests.RequestException as e:
        logger.error(f"Request error: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)