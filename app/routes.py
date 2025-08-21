from flask import Flask, render_template, request, jsonify
import threading
import uuid
from urllib.parse import unquote

from app.services import process_audio, process_upload
from app.utils import configure_logging
from werkzeug.utils import secure_filename
from app.utils import authenticate
from config.settings import ALLOWED_FILE_SOURCES,ALLOWED_EXTENSIONS,API_KEY



logger = configure_logging()

app = Flask(__name__)

jobs = {}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/check_api_key', methods=['POST'])
def check_api_key():
    data = request.json
    provided_api_key = data.get('api_key')
    if provided_api_key == API_KEY:
        return jsonify({'valid': True}), 200
    else:
        return jsonify({'valid': False}), 401

@app.route('/upload', methods=['POST'])

@authenticate
def upload():
    logger.info("Upload endpoint called")
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        logger.info(f"Received file: {filename}")
        
        try:
            file_url = process_upload(file, filename)
            return jsonify({'message': 'File uploaded successfully', 'file_url': file_url}), 200
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            return jsonify({'error': f'Error processing upload: {str(e)}'}), 500
    
    logger.error("Unexpected error in file upload")
    return jsonify({'error': 'Unexpected error in file upload'}), 500

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
    
    # URL decode the filename to handle special characters from webhooks
    filename = unquote(filename)
    logger.info(f"Processing file (after URL decode): {filename}")

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