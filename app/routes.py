from flask import Flask, render_template, request, jsonify
import threading
import os
import uuid
import subprocess
from app.services import process_audio
from app.utils import configure_logging
import tempfile
from werkzeug.utils import secure_filename
from app.services import upload_file
from app.utils import authenticate
from config.settings import S3_ENDPOINT, S3_BUCKET,ALLOWED_FILE_SOURCES

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MiB
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'wmv', 'flv', 'webm'}

logger = configure_logging()

app = Flask(__name__)

jobs = {}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def scale_video(input_path, output_path, target_size):
    logger.info(f"Starting video scaling process: {input_path} -> {output_path}")
    try:
        # Get video information
        logger.debug("Probing video file for information...")
        probe_command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', input_path]
        probe_output = subprocess.check_output(probe_command, universal_newlines=True)
        video_info = json.loads(probe_output)
        logger.debug(f"Video info: {json.dumps(video_info, indent=2)}")
        
        # Calculate scaling factor
        original_size = os.path.getsize(input_path)
        scale_factor = (target_size / original_size) ** 0.5
        logger.info(f"Original size: {original_size / (1024 * 1024):.2f} MiB, Scale factor: {scale_factor:.2f}")
        
        # Get original resolution
        width = int(video_info['streams'][0]['width'])
        height = int(video_info['streams'][0]['height'])
        logger.info(f"Original resolution: {width}x{height}")
        
        # Calculate new resolution
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        logger.info(f"New resolution: {new_width}x{new_height}")
        
        ffmpeg_command = [
            'ffmpeg',
            '-i', input_path,
            '-vf', f'scale={new_width}:{new_height}',
            '-c:v', 'libx264',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',
            output_path
        ]
        
        logger.debug(f"FFmpeg command: {' '.join(ffmpeg_command)}")
        logger.info("Starting FFmpeg encoding process...")
        
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        # Log FFmpeg output in real-time
        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.debug(output.strip())
        
        rc = process.poll()
        if rc != 0:
            logger.error(f"FFmpeg process failed with return code {rc}")
            raise subprocess.CalledProcessError(rc, ffmpeg_command)
        
        new_size = os.path.getsize(output_path)
        logger.info(f"Video scaled successfully. New size: {new_size / (1024 * 1024):.2f} MiB")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error scaling video: {e}")
        raise


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    logger.info("Upload endpoint called")
    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        logger.info(f"Processing file: {filename}")
        
        temp_file_path = os.path.join(tempfile.gettempdir(), filename)
        logger.debug(f"Saving file to temporary location: {temp_file_path}")
        file.save(temp_file_path)

        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Received file: {filename}, Size: {file_size / (1024 * 1024):.2f} MiB")

        if file_size > MAX_FILE_SIZE:
            logger.info(f"File size exceeds {MAX_FILE_SIZE / (1024 * 1024)} MiB. Scaling down.")
            try:
                scaled_file_path = os.path.join(tempfile.gettempdir(), f"scaled_{filename}")
                scale_video(temp_file_path, scaled_file_path, MAX_FILE_SIZE)
                logger.debug(f"Removing original file: {temp_file_path}")
                os.remove(temp_file_path)  # Remove original file
                temp_file_path = scaled_file_path  # Use scaled file for upload
                logger.info(f"Scaled file size: {os.path.getsize(temp_file_path) / (1024 * 1024):.2f} MiB")
            except Exception as e:
                logger.error(f"Error scaling video: {str(e)}")
                return jsonify({'error': 'Error processing video'}), 500

        try:
            logger.info(f"Uploading file to MinIO: {filename}")
            upload_file(temp_file_path, filename)
            file_url = f"{S3_ENDPOINT}/{S3_BUCKET}/{filename}"
            logger.info(f"File uploaded successfully: {file_url}")
            return jsonify({'message': 'File uploaded successfully', 'file_url': file_url}), 200
        except Exception as e:
            logger.error(f"Error uploading file to MinIO: {str(e)}")
            return jsonify({'error': 'Error uploading file'}), 500
        finally:
            if os.path.exists(temp_file_path):
                logger.debug(f"Removing temporary file: {temp_file_path}")
                os.remove(temp_file_path)
                logger.info("Temporary file removed")
    else:
        logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type'}), 400
                        
@app.route('/convert_and_transcribe', methods=['POST'])
@authenticate
def convert_and_transcribe():
    logger.info("Convert and transcribe endpoint called.")
    data = request.json
    filename = data.get('filename')
    file_source = data.get('file_source', 'S3')  # Default to S3 if not specified

    if not filename:
        logger.error("No filename provided")
        return jsonify({'error': 'No filename provided'}), 400

    if file_source not in ALLOWED_FILE_SOURCES:
        logger.error(f"Invalid file source: {file_source}")
        return jsonify({'error': f'Invalid file source. Allowed sources are: {", ".join(ALLOWED_FILE_SOURCES)}'}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'queued'}
    
    thread = threading.Thread(target=process_audio, args=(job_id, filename))
    thread.start()
    
    return jsonify({
        'message': 'Task started successfully',
        'job_id': job_id
    }), 202

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