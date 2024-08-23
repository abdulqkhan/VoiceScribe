from flask import Flask, render_template, request, jsonify
import threading
import os
import uuid
from app.services import process_audio, send_webhook_alert
from app.utils import configure_logging
import tempfile
from werkzeug.utils import secure_filename
from app.services import upload_file
from config.settings import S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY, WEBHOOK_URL

logger = configure_logging()

app = Flask(__name__)

jobs = {}

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        temp_file_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_file_path)

        try:
            upload_file(temp_file_path, filename)
            file_url = f"{S3_ENDPOINT}/{S3_BUCKET}/{filename}"
            return jsonify({'message': 'File uploaded successfully', 'file_url': file_url}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
@app.route('/convert_and_transcribe', methods=['POST'])
def convert_and_transcribe():
    logger.info("Convert and transcribe endpoint called.")
    filename = request.json.get('filename')
    if not filename:
        logger.error("No filename provided")
        return jsonify({'error': 'No filename provided'}), 400

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