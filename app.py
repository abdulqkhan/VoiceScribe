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
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pydub import AudioSegment
from pydub.silence import detect_silence
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
WEBHOOK_URL = os.getenv('WEBHOOK_URL')

logger.info(f"S3_ENDPOINT: {S3_ENDPOINT}")
logger.info(f"S3_BUCKET: {S3_BUCKET}")
logger.info(f"S3_ACCESS_KEY: {'*' * len(S3_ACCESS_KEY) if S3_ACCESS_KEY else 'Not set'}")
logger.info(f"S3_SECRET_KEY: {'*' * len(S3_SECRET_KEY) if S3_SECRET_KEY else 'Not set'}")
logger.info(f"WEBHOOK_URL: {'Set' if WEBHOOK_URL else 'Not set'}")

if not WEBHOOK_URL:
    logger.warning("WEBHOOK_URL is not set. Webhook notifications will be disabled.")
    
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
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours, seconds = divmod(int(seconds), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def create_srt_content(segments):
    srt_content = ""
    for i, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_content.strip()

def get_audio_duration(audio_file_path):
    """
    Get the duration of an audio file in seconds.
    
    :param audio_file_path: Path to the audio file
    :return: Duration in seconds
    """
    audio = AudioSegment.from_file(audio_file_path)
    return len(audio) / 1000.0  

def analyze_audio_silence(audio_file, min_silence_len=1000, silence_thresh=-40):
    """
    Analyze audio file for silent parts.
    
    :param audio_file: Path to the audio file
    :param min_silence_len: Minimum length of silence in milliseconds
    :param silence_thresh: Silence threshold in dB
    :return: List of silent parts as (start, end) tuples in seconds
    """
    audio = AudioSegment.from_file(audio_file)
    silent_parts = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    # Convert milliseconds to seconds
    return [(start/1000, end/1000) for start, end in silent_parts]

def combine_analyses(whisper_result, silent_parts, total_duration):
    """
    Combine Whisper transcription result with detected silent parts.
    
    :param whisper_result: Result from Whisper transcription
    :param silent_parts: List of silent parts from audio analysis
    :param total_duration: Total duration of the audio/video in seconds
    :return: Combined analysis dictionary
    """
    analysis = {
        'total_duration': format_timestamp(total_duration),
        'total_segments': len(whisper_result['segments']),
        'silent_parts': [],
        'transcribed_segments': [],
        'repeated_sentences': [],
        'flagged_for_deletion': []
    }
    
    # Add silent parts
    for start, end in silent_parts:
        analysis['silent_parts'].append({
            'start': format_timestamp(start),
            'end': format_timestamp(end),
            'duration': round(end - start, 2)
        })
    
    # Process Whisper segments
    for segment in whisper_result['segments']:
        analysis['transcribed_segments'].append({
            'start': format_timestamp(segment['start']),
            'end': format_timestamp(segment['end']),
            'text': segment['text'].strip()
        })
    
    # Identify repeated sentences and phrases to delete (implementation left as an exercise)
    
    return analysis


def generate_fcpxml(analysis, original_filename):
    root = ET.Element("fcpxml", version="1.10")
    
    resources = ET.SubElement(root, "resources")
    format_elem = ET.SubElement(resources, "format", 
                  id="r1", 
                  name="FFVideoFormat2048x1024p60", 
                  frameDuration="100/6000s",
                  width="2048",
                  height="1024",
                  colorSpace="1-1-1 (Rec. 709)")
    
    # Use the original video filename
    asset_name = os.path.basename(original_filename)
    asset_src = f"file://localhost/{asset_name}"
    asset = ET.SubElement(resources, "asset",
                          id="r2",
                          name=asset_name,
                          uid="A3A6C0DC87FB2BEE9C7C05F00B652360",
                          start="0s",
                          duration=analysis['total_duration'],
                          hasVideo="1",
                          format="r3",
                          hasAudio="1",
                          audioSources="1",
                          audioChannels="2",
                          audioRate="48000")
    
    # Add media-rep element
    ET.SubElement(asset, "media-rep", kind="original-media", src=asset_src)
    
    library = ET.SubElement(root, "library", location="file:///Users/amelinemamin/Movies/Untitled.fcpbundle/")
    event = ET.SubElement(library, "event", name="22-08-2024", uid="383A3AAB-CFA0-4B8A-B951-BFA8055E151E")
    project = ET.SubElement(event, "project", name="Untitled Project", uid="BAA5D122-1C6F-41B9-B156-F00CCAAAC3D1", modDate="2024-08-22 15:04:04 +0200")
    
    # Calculate total frames
    total_seconds = sum(float(s['duration']) for s in analysis['silent_parts'])
    total_frames = int(total_seconds * 60)  # Assuming 60 fps
    
    sequence = ET.SubElement(project, "sequence",
                             format="r1",
                             duration=f"{total_frames}/60s",
                             tcStart="0s",
                             tcFormat="NDF",
                             audioLayout="stereo",
                             audioRate="48k")
    
    spine = ET.SubElement(sequence, "spine")
    
    # Add video clips, excluding silent parts
    current_time = 0
    for i, silent in enumerate(analysis['silent_parts']):
        silent_start = float(silent['start'].split(':')[-1].replace(',', '.'))
        silent_duration = float(silent['duration'])
        
        if silent_start > current_time:
            clip_duration = silent_start - current_time
            ET.SubElement(spine, "asset-clip",
                          ref="r2",
                          offset=f"{current_time}s",
                          name=f"Clip {i+1}",
                          duration=f"{clip_duration}s",
                          start=f"{current_time}s",
                          audioRole="dialogue")
        
        current_time = silent_start + silent_duration
    
    # Add final clip if needed
    total_duration = float(analysis['total_duration'].split(':')[-1].replace(',', '.'))
    if current_time < total_duration:
        final_duration = total_duration - current_time
        ET.SubElement(spine, "asset-clip",
                      ref="r2",
                      offset=f"{current_time}s",
                      name="Final Clip",
                      duration=f"{final_duration}s",
                      start=f"{current_time}s",
                      audioRole="dialogue")

    xml_str = ET.tostring(root, encoding='unicode')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
    
    # Remove the automatic XML declaration added by toprettyxml
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip() != '<?xml version="1.0" ?>'])
    
    # Ensure the XML declaration is included and at the start
    pretty_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n{pretty_xml.strip()}'
    
    return pretty_xml

def send_webhook_alert(job_id, status, result=None):
    if not WEBHOOK_URL:
        logger.warning(f"Webhook URL not set. Skipping webhook alert for job {job_id}.")
        return

    payload = {
        'job_id': job_id,
        'status': status
    }
    if result:
        payload['result'] = result

    logger.info(f"Attempting to send webhook alert for job {job_id} to {WEBHOOK_URL}")
    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=10)  # Add a timeout
        logger.info(f"Webhook response status code: {response.status_code}")
        logger.info(f"Webhook response content: {response.text}")
        if response.status_code == 200:
            logger.info(f"Webhook alert sent successfully for job {job_id}")
        else:
            logger.warning(f"Failed to send webhook alert for job {job_id}. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending webhook alert for job {job_id}: {str(e)}")

def analyze_whisper_output(result, delete_phrase="bad take", silence_threshold=1):
    logging.info("Starting analyze_whisper_output")
    segments = result['segments']
    logging.info(f"Number of segments: {len(segments)}")
    
    analysis = {
        'total_duration': format_timestamp(segments[-1]['end']),
        'total_segments': len(segments),
        'flagged_for_deletion': [],
        'silent_parts': [],
        'repeated_sentences': [],
    }

    def add_silent_part(start, end):
        duration = end - start
        logging.info(f"Checking silence: start={start}, end={end}, duration={duration}")
        if duration > silence_threshold:
            silent_part = {
                'start': format_timestamp(start),
                'end': format_timestamp(end),
                'duration': round(duration, 2)
            }
            analysis['silent_parts'].append(silent_part)
            logging.info(f"Added silent part: {silent_part}")

    # Check for gaps between segments
    for i in range(len(segments)):
        if i == 0:
            # Check for initial silence
            if segments[i]['start'] > silence_threshold:
                add_silent_part(0, segments[i]['start'])
        else:
            # Check for silence between segments
            gap = segments[i]['start'] - segments[i-1]['end']
            if gap > silence_threshold:
                add_silent_part(segments[i-1]['end'], segments[i]['start'])

        # Check for delete phrase
        text = segments[i]['text'].strip()
        if text.lower().endswith(delete_phrase.lower()):
            analysis['flagged_for_deletion'].append({
                'start': format_timestamp(segments[i]['start']),
                'end': format_timestamp(segments[i]['end']),
                'text': text
            })

    # Find repeated sentences
    sentences = [{'text': s['text'].strip(), 'start': format_timestamp(s['start']), 'end': format_timestamp(s['end'])} for s in segments]
    for i, sentence in enumerate(sentences):
        if any(sentence['text'].lower() == s['text'].lower() for s in sentences[i+1:]):
            repeated = [sentence] + [s for s in sentences[i+1:] if s['text'].lower() == sentence['text'].lower()]
            if repeated not in analysis['repeated_sentences']:
                analysis['repeated_sentences'].append(repeated)

    logging.info(f"Total silent parts detected: {len(analysis['silent_parts'])}")
    return analysis


def generate_analysis_report(analysis):
    report = f"Video Analysis Report\n"
    report += f"====================\n\n"
    report += f"Total Duration: {analysis['total_duration']}\n"
    report += f"Total Segments: {analysis['total_segments']}\n\n"

    if analysis['silent_parts']:
        report += f"Silent Parts (>1 second):\n"
        for i, silent in enumerate(analysis['silent_parts']):
            if i == 0 and silent['start'] == "00:00:00,000":
                report += f"  - Initial silence: [{silent['start']} - {silent['end']}] Duration: {silent['duration']} seconds\n"
            else:
                report += f"  - [{silent['start']} - {silent['end']}] Duration: {silent['duration']} seconds\n"
        report += "\n"

    if analysis['repeated_sentences']:
        report += f"Repeated Sentences:\n"
        for repeat in analysis['repeated_sentences']:
            report += f"  - '{repeat[0]['text']}' (Repeated {len(repeat)} times)\n"
            for occur in repeat:
                report += f"    [{occur['start']} - {occur['end']}]\n"
        report += "\n"

    if analysis['flagged_for_deletion']:
        report += f"Segments Flagged for Deletion:\n"
        for flagged in analysis['flagged_for_deletion']:
            report += f"  - [{flagged['start']} - {flagged['end']}] {flagged['text']}\n"
        report += "\n"

    return report

def process_audio(job_id, filename):
    try:
        file_url = get_public_url(filename)
        logger.info(f"Starting process for file: {file_url}")
        
        # Determine if the file is MP3 or needs conversion
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension == '.mp3':
            mp3_filename = filename
            mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
            logger.info(f"Downloading MP3 file: {mp3_path}")
            response = requests.get(file_url)
            with open(mp3_path, 'wb') as f:
                f.write(response.content)
        else:
            # Convert video to MP3 using FFmpeg
            mp3_filename = f"{os.path.splitext(filename)[0]}.mp3"
            mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
            
            logger.info(f"Converting file to MP3: {mp3_path}")
            ffmpeg_command = [
                'ffmpeg',
                '-i', file_url,
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
        
        # Upload MP3 to S3 if it was converted
        if file_extension != '.mp3':
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
        
        # Create SRT content
        srt_content = create_srt_content(result["segments"])
        
        # Save and upload SRT file
        srt_filename = f"{os.path.splitext(filename)[0]}.srt"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.srt') as temp_srt_file:
            temp_srt_file.write(srt_content)
            temp_srt_path = temp_srt_file.name
        
        logger.info(f"Uploading SRT to S3: {srt_filename}")
        upload_file(temp_srt_path, srt_filename)
        
        # Process transcription with timestamps (for full transcription file)
        transcription_with_timestamps = []
        for segment in result["segments"]:
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"]
            transcription_with_timestamps.append(f"[{start_time} - {end_time}] {text}")
        
        # Join the transcription segments
        full_transcription = "\n".join(transcription_with_timestamps)
        
        # Upload transcription to S3
        transcription_filename = f"{os.path.splitext(filename)[0]}_transcription.txt"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_transcription_file:
            temp_transcription_file.write(full_transcription)
            temp_transcription_path = temp_transcription_file.name
        
        logger.info(f"Uploading transcription to S3: {transcription_filename}")
        upload_file(temp_transcription_path, transcription_filename)
        
        # Analyze audio for silent parts
        silent_parts = analyze_audio_silence(mp3_path)

        # Transcribe with Whisper
        whisper_result = model.transcribe(mp3_path)

        # Combine analyses
        total_duration = get_audio_duration(mp3_path)  # Implement this function to get total duration
        analysis_result = combine_analyses(whisper_result, silent_parts, total_duration)

        # Generate human-readable report
        report = generate_analysis_report(analysis_result)
        
        # Save and upload report file
        report_filename = f"{os.path.splitext(filename)[0]}_report.txt"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_report_file:
            temp_report_file.write(report)
            temp_report_path = temp_report_file.name
        
        logger.info(f"Uploading analysis report to S3: {report_filename}")
        upload_file(temp_report_path, report_filename)

        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        os.unlink(mp3_path)
        os.unlink(temp_srt_path)
        os.unlink(temp_transcription_path)
        os.unlink(temp_report_path)
        logger.info("Temporary files cleaned up")

        jobs[job_id] = {
            'status': 'completed',
            'result': {
                'message': 'File processed, transcribed, and analyzed successfully',
                'original_filename': filename,
                'mp3_filename': mp3_filename,
                'transcription_filename': transcription_filename,
                'srt_filename': srt_filename,
                'report_filename': report_filename
            }
        }
        logger.info(f"Job {job_id} completed. Sending webhook alert.")
        send_webhook_alert(job_id, 'completed', jobs[job_id]['result'])
        logger.info(f"Webhook alert sent for job {job_id}")
        
    except Exception as e:
        logger.error(f"An error occurred in process_audio: {str(e)}")
        jobs[job_id] = {'status': 'failed', 'error': str(e)}
        
        logger.info(f"Job {job_id} failed. Sending webhook alert.")
        send_webhook_alert(job_id, 'failed', {'error': str(e)})
        logger.info(f"Webhook alert sent for failed job {job_id}")
        
@app.route('/convert_and_transcribe', methods=['POST'])
def convert_and_transcribe():
    logger.info("Convert and transcribe endpoint called.")
    filename = request.json.get('filename')
    if not filename:
        logger.error("No filename provided")
        return jsonify({'error': 'No filename provided'}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'processing'}
    
    # Start the processing in a new thread
    thread = threading.Thread(target=process_audio, args=(job_id, filename))
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