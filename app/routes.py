from flask import Flask, render_template, request, jsonify
import threading
import uuid
from urllib.parse import unquote_plus
from datetime import timedelta

from app.services import process_audio, process_upload, s3_client
from app.utils import configure_logging, jobs
from werkzeug.utils import secure_filename
from app.utils import authenticate
from config.settings import ALLOWED_FILE_SOURCES,ALLOWED_EXTENSIONS,API_KEY,S3_BUCKET



logger = configure_logging()

app = Flask(__name__)
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

@app.route('/upload_and_process', methods=['POST'])
@authenticate
def upload_and_process():
    """
    Upload file and start transcription process with email tracking.
    
    This endpoint is designed for external systems that need to:
    1. Upload a file and get an immediate job_id
    2. Monitor job completion via job_id
    3. Send email notifications when processing completes
    
    The email is stored with the job for external notification services
    - this app does not send emails directly.
    """
    logger.info("Upload and process endpoint called")
    
    # Get email from form data - required for external notification services
    email = request.form.get('email')
    if not email:
        logger.error("No email provided")
        return jsonify({'error': 'Email is required'}), 400
    
    # Check if file is present
    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    logger.info(f"Received file: {filename} for email: {email}")
    
    try:
        # Upload file first
        file_url = process_upload(file, filename)
        logger.info(f"File uploaded successfully: {file_url}")
        
        # Create job directly without using test_client
        job_id = str(uuid.uuid4())
        job_data = {
            'status': 'queued',
            'filename': filename,
            'processing_source': 'manual',
            'is_repurpose': False,
            'email': email
        }
        jobs[job_id] = job_data
        logger.info(f"Created job {job_id} with data: {job_data}")
        
        # Start processing thread directly
        thread = threading.Thread(target=process_audio, args=(job_id, filename))
        thread.start()
        
        logger.info(f"Job {job_id} created and processing started")
        
        return jsonify({
            'message': 'File uploaded and processing started',
            'job_id': job_id,
            'filename': filename,
            'email': email
        }), 202
        
    except Exception as e:
        logger.error(f"Error in upload_and_process: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/repurpose', methods=['POST'])
@authenticate
def repurpose():
    """
    Upload file and start transcription with repurpose instructions.
    
    This endpoint processes files normally through transcription,
    then marks them for AI-powered content transformation.
    External systems monitor for repurpose jobs and handle the AI processing.
    """
    logger.info("Repurpose endpoint called")
    
    # Get required fields from form data
    email = request.form.get('email')
    repurpose_message = request.form.get('repurpose_message')
    video_url = request.form.get('video_url')  # Optional: original video URL
    
    if not email:
        logger.error("No email provided")
        return jsonify({'error': 'Email is required'}), 400
    
    if not repurpose_message:
        logger.error("No repurpose message provided")
        return jsonify({'error': 'Repurpose message is required'}), 400
    
    # Check if file is present
    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    logger.info(f"Received file: {filename} for repurpose with email: {email}")
    logger.info(f"Repurpose message: {repurpose_message}")
    logger.info(f"Video URL: {video_url}")
    
    try:
        # Upload file first
        file_url = process_upload(file, filename)
        logger.info(f"File uploaded successfully: {file_url}")
        
        # Create job directly without using test_client
        job_id = str(uuid.uuid4())
        job_data = {
            'status': 'queued',
            'filename': filename,
            'processing_source': 'manual',
            'is_repurpose': True,
            'email': email,
            'repurpose_message': repurpose_message
        }
        
        # Add video_url if provided
        if video_url:
            job_data['video_url'] = video_url
            logger.info(f"Added video_url to job data: {video_url}")
        jobs[job_id] = job_data
        logger.info(f"Created repurpose job {job_id} with data: {job_data}")
        
        # Start processing thread directly
        thread = threading.Thread(target=process_audio, args=(job_id, filename))
        thread.start()
        
        logger.info(f"Job {job_id} created and repurpose processing started")
        
        return jsonify({
            'message': 'File uploaded and repurpose processing started',
            'job_id': job_id,
            'filename': filename,
            'email': email,
            'repurpose_message': repurpose_message
        }), 202
        
    except Exception as e:
        logger.error(f"Error in repurpose: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

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
    # unquote_plus handles both %20 and + as spaces
    filename = unquote_plus(filename)
    # Convert three dots to ellipsis character to match MinIO filename
    filename = filename.replace('...', '…')
    logger.info(f"Processing file (after URL decode): {filename}")

    if file_source not in ALLOWED_FILE_SOURCES:
        logger.error(f"Invalid file source: {file_source}")
        return jsonify({'error': f'Invalid file source. Allowed sources are: {", ".join(ALLOWED_FILE_SOURCES)}'}), 400

    # Check if filename is already being processed by manual upload endpoints
    for job_data in jobs.values():
        if (job_data.get('filename') == filename and 
            job_data.get('processing_source') == 'manual' and
            job_data.get('status') in ['queued', 'processing']):
            logger.info(f"Skipping {filename} - already being processed manually")
            return jsonify({'message': 'File already being processed', 'skipped': True}), 200

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'queued', 
        'filename': filename,
        'processing_source': 'auto',  # Flag this as auto-processing
        'is_repurpose': False,  # Default for regular transcription jobs
        'email': None
    }
    
    thread = threading.Thread(target=process_audio, args=(job_id, filename))
    thread.start()
    
    return jsonify({
        'message': 'Task started successfully',
        'job_id': job_id
    }), 202

@app.route('/jobs/active', methods=['GET'])
def get_active_jobs():
    active_jobs = []
    for job_id, job_data in jobs.items():
        if job_data['status'] in ['queued', 'processing']:
            active_jobs.append({
                'job_id': job_id,
                'status': job_data['status'],
                'filename': job_data.get('filename'),
                'email': job_data.get('email')
            })
    
    return jsonify({
        'active_jobs': active_jobs,
        'count': len(active_jobs)
    }), 200

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

@app.route('/is_processing/<filename>', methods=['GET'])
def is_processing(filename):
    """Check if a filename is already being processed by manual upload endpoints"""
    for job_data in jobs.values():
        if (job_data.get('filename') == filename and 
            job_data.get('processing_source') == 'manual' and
            job_data.get('status') in ['queued', 'processing']):
            return jsonify({'skip': True}), 200
    return jsonify({'skip': False}), 200

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check endpoint called.")
    return jsonify({"status": "healthy"}), 200

@app.route('/api/generate-signed-url', methods=['POST'])
@authenticate
def generate_signed_url():
    try:
        # Get parameters from request
        data = request.get_json()
        filename = data.get('filename')
        bucket = data.get('bucket', S3_BUCKET)  # Default to S3_BUCKET
        expires_in = data.get('expires_in', 86400)  # Default to 24 hours (in seconds)
        
        if not filename:
            return jsonify({'error': 'Filename required'}), 400
        
        # Validate expires_in (max 7 days for S3 signed URLs)
        if expires_in > 604800:  # 7 days in seconds
            expires_in = 604800
        
        # Generate presigned URL
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': filename},
            ExpiresIn=expires_in
        )
        
        # Convert seconds to human readable
        hours = expires_in / 3600
        expires_text = f"{hours:.1f} hours" if hours < 48 else f"{hours/24:.1f} days"
        
        return jsonify({
            'success': True,
            'signed_url': url,
            'expires_in': expires_text,
            'expires_in_seconds': expires_in,
            'filename': filename,
            'bucket': bucket
        })
        
    except Exception as e:
        logger.error(f"Error generating signed URL for {bucket}/{filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000)
else:
    logger.info("Flask application imported, not running directly.")