import os
import io
import tempfile
import boto3
from botocore.client import Config
import whisper
import requests
import subprocess
import numpy as np
import soundfile as sf
import torch
from urllib.error import URLError
from pydub import AudioSegment
from pydub.silence import detect_silence
from app.utils import configure_logging, jobs
from config.settings import S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY, WEBHOOK_URL

logger = configure_logging()

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
    audio = AudioSegment.from_file(audio_file_path)
    return len(audio) / 1000.0

def analyze_audio_silence(audio_file, min_silence_len=1000, silence_thresh=-40):
    audio = AudioSegment.from_file(audio_file)
    silent_parts = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return [(start/1000, end/1000) for start, end in silent_parts]

def combine_analyses(whisper_result, silent_parts, total_duration):
    analysis = {
        'total_duration': format_timestamp(total_duration),
        'total_segments': len(whisper_result['segments']),
        'silent_parts': [],
        'transcribed_segments': [],
        'repeated_sentences': [],
        'flagged_for_deletion': []
    }
    
    current_time = 0
    silent_parts_index = 0
    segment_text_map = {}
    
    for segment in whisper_result['segments']:
        # Add silent parts before the current segment
        while silent_parts_index < len(silent_parts) and silent_parts[silent_parts_index][1] <= segment['start']:
            start, end = silent_parts[silent_parts_index]
            if start > current_time:
                analysis['silent_parts'].append({
                    'start': format_timestamp(start),
                    'end': format_timestamp(end),
                    'duration': round(end - start, 2)
                })
            current_time = end
            silent_parts_index += 1
        
        # Add the current segment
        segment_info = {
            'start': format_timestamp(segment['start']),
            'end': format_timestamp(segment['end']),
            'text': segment['text'].strip()
        }
        analysis['transcribed_segments'].append(segment_info)
        current_time = segment['end']
        
        # Track repeated sentences
        text = segment_info['text']
        if text in segment_text_map:
            segment_text_map[text].append(segment_info)
        else:
            segment_text_map[text] = [segment_info]
    
    # Add remaining silent parts
    while silent_parts_index < len(silent_parts):
        start, end = silent_parts[silent_parts_index]
        if start > current_time:
            analysis['silent_parts'].append({
                'start': format_timestamp(start),
                'end': format_timestamp(end),
                'duration': round(end - start, 2)
            })
        current_time = end
        silent_parts_index += 1
    
    # Identify repeated sentences
    for text, occurrences in segment_text_map.items():
        if len(occurrences) > 1:
            analysis['repeated_sentences'].append(occurrences)
    
    # Specific phrase detection
    specific_phrases = ["cut","bad take","redo"]
    for phrase in specific_phrases:
        if phrase in segment_text_map:
            analysis['repeated_sentences'].append(segment_text_map[phrase])
    
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
        
        # Stream the file from S3
        s3_object = s3_client.get_object(Bucket=S3_BUCKET, Key=filename)
        file_stream = io.BytesIO(s3_object['Body'].read())
        
        # Determine the file type
        file_extension = os.path.splitext(filename)[1].lower()
        
        # Convert to WAV if necessary (Whisper works well with WAV)
        if file_extension != '.wav':
            logger.info(f"Converting file to WAV")
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_input_file:
                temp_input_file.write(file_stream.getvalue())
                temp_input_file.flush()
                
                temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_output_file.close()
                
                ffmpeg_command = [
                    'ffmpeg',
                    '-i', temp_input_file.name,
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',
                    temp_output_file.name
                ]
                
                process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    raise Exception(f"FFmpeg error: {stderr.decode()}")
                
                audio_path = temp_output_file.name
                
                # Clean up input temporary file
                os.unlink(temp_input_file.name)
        else:
            # If it's already a WAV, save it to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
                temp_wav_file.write(file_stream.getvalue())
                audio_path = temp_wav_file.name
        
        logger.info("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        try:
            model = whisper.load_model("base", device=device)
            logger.info(f"Whisper model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise Exception(f"Failed to load Whisper model: {str(e)}")

        logger.info("Transcribing audio...")
        result = model.transcribe(audio_path)
        logger.info("Transcription complete")
        
        srt_content = create_srt_content(result["segments"])
        
        srt_filename = f"{os.path.splitext(filename)[0]}.srt"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.srt') as temp_srt_file:
            temp_srt_file.write(srt_content)
            temp_srt_path = temp_srt_file.name
        
        logger.info(f"Uploading SRT to S3: {srt_filename}")
        upload_file(temp_srt_path, srt_filename)
        
        transcription_with_timestamps = []
        for segment in result["segments"]:
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"]
            transcription_with_timestamps.append(f"[{start_time} - {end_time}] {text}")
        
        full_transcription = "\n".join(transcription_with_timestamps)
        
        transcription_filename = f"{os.path.splitext(filename)[0]}_transcription.txt"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_transcription_file:
            temp_transcription_file.write(full_transcription)
            temp_transcription_path = temp_transcription_file.name
        
        logger.info(f"Uploading transcription to S3: {transcription_filename}")
        upload_file(temp_transcription_path, transcription_filename)
        
        logger.info("Analyzing audio silence...")
        silent_parts = analyze_audio_silence(audio_path)
        logger.info("Audio silence analysis complete")
        
        logger.info("Getting audio duration...")
        total_duration = get_audio_duration(audio_path)
        logger.info(f"Audio duration: {total_duration} seconds")
        
        logger.info("Combining analyses...")
        analysis_result = combine_analyses(result, silent_parts, total_duration)
        logger.info("Analysis combination complete")
        
        logger.info("Generating analysis report...")
        report = generate_analysis_report(analysis_result)
        logger.info("Analysis report generated")
        
        report_filename = f"{os.path.splitext(filename)[0]}_report.txt"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_report_file:
            temp_report_file.write(report)
            temp_report_path = temp_report_file.name
        
        logger.info(f"Uploading analysis report to S3: {report_filename}")
        upload_file(temp_report_path, report_filename)

        logger.info("Cleaning up temporary files...")
        os.unlink(audio_path)
        os.unlink(temp_srt_path)
        os.unlink(temp_transcription_path)
        os.unlink(temp_report_path)
        logger.info("Temporary files cleaned up")

        jobs[job_id] = {
            'status': 'completed',
            'result': {
                'message': 'File processed, transcribed, and analyzed successfully',
                'original_filename': filename,
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
        response = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        logger.info(f"Webhook response status code: {response.status_code}")
        logger.info(f"Webhook response content: {response.text}")
        if response.status_code == 200:
            logger.info(f"Webhook alert sent successfully for job {job_id}")
        else:
            logger.warning(f"Failed to send webhook alert for job {job_id}. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending webhook alert for job {job_id}: {str(e)}")