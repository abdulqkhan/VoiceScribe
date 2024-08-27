import os
import tempfile
import boto3
from botocore.client import Config
import whisper
import requests
import subprocess
import json
import torch
import tempfile
from pydub import AudioSegment
from pydub.silence import detect_silence
from app.utils import configure_logging, jobs
from config.settings import S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY, WEBHOOK_URL, MAX_FILE_SIZE, API_KEY
logger = configure_logging()
s3_client = boto3.client('s3',
                         endpoint_url=S3_ENDPOINT,
                         aws_access_key_id=S3_ACCESS_KEY,
                         aws_secret_access_key=S3_SECRET_KEY,
                         config=Config(signature_version='s3v4'))

def get_public_url(filename):
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{filename}"

def get_file_size_from_public_url(url):
    try:
        response = requests.head(url)
        response.raise_for_status()
        return int(response.headers.get('Content-Length', 0))
    except requests.RequestException as e:
        logger.error(f"Error accessing file at {url}: {str(e)}")
        raise

def download_file_from_public_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        raise

def upload_file(file_path, object_name):
    try:
        logger.info(f"Uploading file {file_path} to S3 as {object_name}")
        s3_client.upload_file(file_path, S3_BUCKET, object_name)
        logger.info(f"Upload successful: {object_name}")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise

def process_upload(file, filename):
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
        except subprocess.CalledProcessError as e:
            logger.exception(f"Error scaling video: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error scaling video: {str(e)}")
            raise

    try:
        logger.info(f"Uploading file to S3: {filename}")
        upload_file(temp_file_path, filename)
        file_url = f"{S3_ENDPOINT}/{S3_BUCKET}/{filename}"
        logger.info(f"File uploaded successfully: {file_url}")
        return file_url
    except Exception as e:
        logger.exception(f"Error uploading file to S3: {str(e)}")
        raise
    finally:
        if os.path.exists(temp_file_path):
            logger.debug(f"Removing temporary file: {temp_file_path}")
            os.remove(temp_file_path)
            logger.info("Temporary file removed")
            
def scale_video(input_path, output_path, target_size):
    logger.info(f"Starting video scaling process: {input_path} -> {output_path}")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    current_input = input_path

    try:
        # Get video information
        probe_command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', current_input]
        probe_output = subprocess.check_output(probe_command, universal_newlines=True)
        video_info = json.loads(probe_output)

        # Calculate target bitrate
        duration = float(video_info['format']['duration'])
        target_bitrate = int((target_size * 8) / duration * 0.95)  # 95% of the target size for video

        # Set a minimum bitrate to ensure some video quality
        min_bitrate = 100000  # 100 kbps
        target_bitrate = max(target_bitrate, min_bitrate)

        # Calculate new resolution (reduced by half)
        width = int(int(video_info['streams'][0]['width']) / 2)
        height = int(int(video_info['streams'][0]['height']) / 2)

        # Ensure even dimensions
        width = width - (width % 2)
        height = height - (height % 2)

        ffmpeg_command = [
            'ffmpeg',
            '-i', current_input,
            '-vf', f'scale={width}:{height}',
            '-c:v', 'libx264',
            '-b:v', f'{target_bitrate}',
            '-maxrate', f'{target_bitrate}',
            '-bufsize', f'{target_bitrate*2}',
            '-preset', 'slow',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-y',
            temp_file
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
            stderr = process.stderr.read()
            logger.error(f"FFmpeg process failed with return code {rc}")
            logger.error(f"FFmpeg error output: {stderr}")
            raise subprocess.CalledProcessError(rc, ffmpeg_command, stderr)

        new_size = os.path.getsize(temp_file)
        logger.info(f"Scaled video size: {new_size / (1024 * 1024):.2f} MiB")

        if new_size <= target_size:
            logger.info("Target size reached. Scaling complete.")
            os.rename(temp_file, output_path)
        else:
            logger.warning(f"File still too large ({new_size / (1024 * 1024):.2f} MiB). Further compression may be needed.")
            os.rename(temp_file, output_path)

    except Exception as e:
        logger.exception(f"Error in scale_video: {str(e)}")
        raise
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        logger.info("Temporary files cleaned up")

    logger.info(f"Video scaling complete. Final size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MiB")
    
def format_timestamp(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours, seconds = divmod(int(seconds), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def scale_video_if_needed(input_file, output_file, target_size_mb=25):
    """Scale the video if it's larger than the target size."""
    input_size = os.path.getsize(input_file) / (1024 * 1024)  # Size in MB
    
    if input_size <= target_size_mb:
        # If the file is already small enough, just copy it
        os.rename(input_file, output_file)
        return
    
    # Calculate the scale factor
    scale_factor = (target_size_mb / input_size) ** 0.5
    
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_file,
        '-vf', f'scale=iw*{scale_factor:.2f}:ih*{scale_factor:.2f}',
        '-b:v', f'{target_size_mb/2:.0f}M',  # Allocate half the target size to video
        '-b:a', '128k',
        '-progress', '-',
        '-y',
        output_file
    ]
    
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    for line in process.stdout:
        if 'out_time_ms' in line:
            try:
                time_ms = int(line.split('=')[1])
                progress = (time_ms / 1000000) / get_audio_duration(input_file) * 100
                logger.info(f"Video conversion progress: {progress:.2f}%")
            except ValueError:
                # Skip lines that can't be parsed
                continue
    
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        raise Exception(f"FFmpeg error: {stderr}")

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
    temp_files = []  # List to keep track of temporary files
    try:
        file_url = get_public_url(filename)
        logger.info(f"Starting process for file: {file_url}")
        
        # Get file size
        try:
            file_size_mb = get_file_size_from_public_url(file_url) / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
        except Exception as e:
            raise Exception(f"Failed to get file size: {str(e)}")
        
        # Download the file
        try:
            file_content = download_file_from_public_url(file_url)
        except Exception as e:
            raise Exception(f"Failed to download file: {str(e)}")
        
        # Process the file content
        temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        temp_files.append(temp_input_file.name)
        temp_input_file.write(file_content)
        temp_input_file.close()
        
        temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_files.append(temp_output_file.name)
        temp_output_file.close()
        
        # Scale video if needed
        logger.info(f"Scaling video if needed: {temp_input_file.name} -> {temp_output_file.name}")
        scale_video_if_needed(temp_input_file.name, temp_output_file.name)
        
        scaled_video_path = temp_output_file.name
        
        # Extract audio from the scaled-down video
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        temp_files.append(audio_path)
        ffmpeg_command = [
            'ffmpeg',
            '-i', scaled_video_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            audio_path
        ]
        
        logger.info(f"Extracting audio: {' '.join(ffmpeg_command)}")
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg error: {stderr.decode()}")
        
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
        temp_srt_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.srt')
        temp_files.append(temp_srt_file.name)
        temp_srt_file.write(srt_content)
        temp_srt_file.close()
        
        logger.info(f"Uploading SRT to S3: {srt_filename}")
        upload_file(temp_srt_file.name, srt_filename)
        
        transcription_with_timestamps = []
        for segment in result["segments"]:
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"]
            transcription_with_timestamps.append(f"[{start_time} - {end_time}] {text}")
        
        full_transcription = "\n".join(transcription_with_timestamps)
        
        transcription_filename = f"{os.path.splitext(filename)[0]}_transcription.txt"
        temp_transcription_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_files.append(temp_transcription_file.name)
        temp_transcription_file.write(full_transcription)
        temp_transcription_file.close()
        
        logger.info(f"Uploading transcription to S3: {transcription_filename}")
        upload_file(temp_transcription_file.name, transcription_filename)
        
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
        temp_report_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_files.append(temp_report_file.name)
        temp_report_file.write(report)
        temp_report_file.close()
        
        logger.info(f"Uploading analysis report to S3: {report_filename}")
        upload_file(temp_report_file.name, report_filename)

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
    
    finally:
        logger.info("Cleaning up temporary files...")
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
                logger.info(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {str(e)}")
        logger.info("Temporary files cleanup completed")
            
        
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