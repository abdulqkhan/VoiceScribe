import os
from flask import Flask, request, jsonify
from pydub import AudioSegment
import requests
import tempfile
from dotenv import load_dotenv
import traceback
import boto3
from botocore.client import Config

load_dotenv()

app = Flask(__name__)

S3_ENDPOINT = os.getenv('S3_ENDPOINT')
S3_BUCKET = os.getenv('S3_BUCKET')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')

app.logger.info(f"S3_ENDPOINT: {S3_ENDPOINT}")
app.logger.info(f"S3_BUCKET: {S3_BUCKET}")

s3_client = boto3.client('s3',
                         endpoint_url=S3_ENDPOINT,
                         aws_access_key_id=S3_ACCESS_KEY,
                         aws_secret_access_key=S3_SECRET_KEY,
                         config=Config(signature_version='s3v4'))

def get_public_url(filename):
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{filename}"

def upload_file(file_path, object_name):
    try:
        s3_client.upload_file(file_path, S3_BUCKET, object_name)
    except Exception as e:
        app.logger.error(f"Upload error: {e}")
        raise

@app.route('/convert', methods=['POST'])
def convert_to_wav():
    video_filename = request.json.get('video_filename')
    if not video_filename:
        return jsonify({'error': 'No video filename provided'}), 400

    video_url = get_public_url(video_filename)

    try:
        app.logger.info(f"Attempting to process video: {video_url}")
        
        # Download video from public URL
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_filename)[1]) as temp_video_file:
            app.logger.info(f"Downloading file from: {video_url}")
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                temp_video_file.write(chunk)
            temp_video_path = temp_video_file.name

        # Convert to WAV
        app.logger.info("Converting to WAV...")
        audio = AudioSegment.from_file(temp_video_path)
        wav_filename = f"{os.path.splitext(video_filename)[0]}.wav"
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        audio.export(wav_path, format="wav")
        
        # Upload WAV to S3
        app.logger.info(f"Uploading WAV to S3: {wav_filename}")
        upload_file(wav_path, wav_filename)
        
        # Clean up temporary files
        os.unlink(temp_video_path)
        os.unlink(wav_path)

        return jsonify({
            'message': 'Video converted and uploaded successfully',
            'wav_url': get_public_url(wav_filename)
        }), 200

    except requests.RequestException as e:
        app.logger.error(f"Request error: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)