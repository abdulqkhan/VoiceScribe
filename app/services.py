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
from urllib.parse import quote
from pydub import AudioSegment
from difflib import SequenceMatcher
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
    # URL encode the filename to handle special characters
    encoded_filename = quote(filename, safe='')
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{encoded_filename}"

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

    try:
        # Get video information
        probe_command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', input_path]
        probe_output = subprocess.check_output(probe_command, universal_newlines=True)
        video_info = json.loads(probe_output)

        # Calculate target bitrate
        duration = float(video_info['format']['duration'])
        target_total_bitrate = int((target_size * 8) / duration * 0.95)  # 95% of the target size for video+audio

        # Allocate bitrate for video and audio
        target_video_bitrate = int(target_total_bitrate * 0.95)  # 95% for video
        target_audio_bitrate = int(target_total_bitrate * 0.05)  # 5% for audio

        # Set minimum bitrates
        min_video_bitrate = 100000  # 100 kbps
        min_audio_bitrate = 32000   # 32 kbps
        target_video_bitrate = max(target_video_bitrate, min_video_bitrate)
        target_audio_bitrate = max(target_audio_bitrate, min_audio_bitrate)

        # Find the video stream
        video_stream = next((stream for stream in video_info['streams'] if stream['codec_type'] == 'video'), None)
        
        if video_stream is None:
            raise ValueError("No video stream found in the input file")

        # Get original dimensions, defaulting to 1280x720 if not found
        original_width = int(video_stream.get('width', 1280))
        original_height = int(video_stream.get('height', 720))
        aspect_ratio = original_width / original_height

        # Calculate new resolution
        if aspect_ratio > 1:  # Landscape
            new_width = min(original_width, 854)  # Max width of 854 (480p)
            new_height = int(new_width / aspect_ratio)
        else:  # Portrait
            new_height = min(original_height, 854)  # Max height of 854
            new_width = int(new_height * aspect_ratio)

        # Ensure even dimensions
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)

        ffmpeg_command = [
            'ffmpeg',
            '-i', input_path,
            '-vf', f'scale={new_width}:{new_height}',
            '-c:v', 'libx264',
            '-b:v', f'{target_video_bitrate}',
            '-maxrate', f'{target_video_bitrate}',
            '-bufsize', f'{target_video_bitrate*2}',
            '-preset', 'slower',  # Use 'slower' preset for better compression
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', f'{target_audio_bitrate}',
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
            logger.warning(f"File still too large ({new_size / (1024 * 1024):.2f} MiB). Attempting further compression.")
            
            # If still too large, try one more pass with more aggressive settings
            ffmpeg_command = [
                'ffmpeg',
                '-i', temp_file,
                '-c:v', 'libx264',
                '-b:v', f'{target_video_bitrate * 0.8}',  # Reduce bitrate by 20%
                '-maxrate', f'{target_video_bitrate * 0.8}',
                '-bufsize', f'{target_video_bitrate * 1.5}',
                '-preset', 'veryslow',  # Use 'veryslow' preset for maximum compression
                '-crf', '28',  # Increase CRF for more compression
                '-c:a', 'aac',
                '-b:a', f'{target_audio_bitrate}',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]

            logger.debug(f"Second pass FFmpeg command: {' '.join(ffmpeg_command)}")
            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)

        final_size = os.path.getsize(output_path)
        logger.info(f"Final video size: {final_size / (1024 * 1024):.2f} MiB")

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

def analyze_audio_silence(audio_file, min_silence_len=2000, silence_thresh=-35):
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
        'flagged_for_review': []
    }
    
    current_time = 0
    silent_parts_index = 0
    segment_text_map = {}
    
    for segment in whisper_result['segments']:
        # Silent parts detection (unchanged)
        while silent_parts_index < len(silent_parts) and silent_parts[silent_parts_index][1] <= segment['start']:
            start, end = silent_parts[silent_parts_index]
            if start > current_time:
                duration = end - start
                if duration >= 2:
                    analysis['silent_parts'].append({
                        'start': format_timestamp(start),
                        'end': format_timestamp(end),
                        'duration': round(duration, 2)
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
        
        # Track similar sentences
        for prev_text, prev_occurrences in segment_text_map.items():
            similarity = SequenceMatcher(None, segment_info['text'].lower(), prev_text.lower()).ratio()
            if similarity > 0.7:  # Adjust this threshold as needed
                prev_occurrences.append(segment_info)
                break
        else:
            segment_text_map[segment_info['text']] = [segment_info]
    
    # Identify repeated or similar sentences
    for text, occurrences in segment_text_map.items():
        if len(occurrences) > 1:
            analysis['repeated_sentences'].append(occurrences)
    
    # Flag segments for review
    filler_words = ["um", "uh", "er", "ah", "like", "you know"]
    for i, segment in enumerate(analysis['transcribed_segments']):
        text = segment['text'].lower()
        
        # Check for filler words
        if any(word in text.split() for word in filler_words):
            analysis['flagged_for_review'].append({'reason': 'Filler words', 'segment': segment})
        
        # Check for hesitations or corrections
        if i > 0:
            prev_text = analysis['transcribed_segments'][i-1]['text'].lower()
            if text.startswith(prev_text[:10]):  # Check if this segment starts similarly to the previous one
                analysis['flagged_for_review'].append({'reason': 'Possible correction', 'segment': segment})
    
    return analysis

def generate_analysis_report(analysis):
    report = f"Video Analysis Report\n"
    report += f"====================\n\n"
    report += f"Total Duration: {analysis['total_duration']}\n"
    report += f"Total Segments: {analysis['total_segments']}\n\n"

    if analysis['silent_parts']:
        report += f"Silent Parts (>2 seconds):\n"
        for i, silent in enumerate(analysis['silent_parts'], 1):
            report += f"  {i}. [{silent['start']} - {silent['end']}] Duration: {silent['duration']} seconds\n"
        report += f"\nTotal silent parts: {len(analysis['silent_parts'])}\n\n"
    else:
        report += "No significant silent parts detected.\n\n"

    if analysis['repeated_sentences']:
        report += f"Similar or Repeated Sentences:\n"
        for i, repeat in enumerate(analysis['repeated_sentences'], 1):
            report += f"  {i}. Similar phrases found {len(repeat)} times:\n"
            for occur in repeat:
                report += f"    [{occur['start']} - {occur['end']}] {occur['text']}\n"
        report += f"\nTotal groups of similar sentences: {len(analysis['repeated_sentences'])}\n\n"
    else:
        report += "No similar or repeated sentences detected.\n\n"

    if analysis['flagged_for_review']:
        report += f"Segments Flagged for Review:\n"
        for i, flagged in enumerate(analysis['flagged_for_review'], 1):
            report += f"  {i}. [{flagged['segment']['start']} - {flagged['segment']['end']}] {flagged['segment']['text']}\n"
            report += f"     Reason: {flagged['reason']}\n"
        report += f"\nTotal segments flagged for review: {len(analysis['flagged_for_review'])}\n\n"
    else:
        report += "No segments flagged for review.\n\n"

    return report

def process_audio(job_id, filename):
    temp_files = []  # List to keep track of temporary files
    try:
        file_url = get_public_url(filename)
        logger.info(f"Starting process for file: {file_url}")

        # Get file size
        file_size_mb = get_file_size_from_public_url(file_url) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

        # Download the file
        file_content = download_file_from_public_url(file_url)

        # Process the file content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_input_file:
            temp_files.append(temp_input_file.name)
            temp_input_file.write(file_content)

        # Check if the file has a video stream
        probe_command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', temp_input_file.name]
        probe_output = subprocess.check_output(probe_command, universal_newlines=True)
        file_info = json.loads(probe_output)
        
        has_video = any(stream['codec_type'] == 'video' for stream in file_info.get('streams', []))
        has_audio = any(stream['codec_type'] == 'audio' for stream in file_info.get('streams', []))
        
        if not has_audio:
            raise ValueError("No audio stream found in the input file")
        
        logger.info(f"File analysis - Has video: {has_video}, Has audio: {has_audio}")
        
        # Determine the source file for audio extraction
        if has_video:
            # Scale video first
            scaled_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            temp_files.append(scaled_video_path)
            
            logger.info(f"Video stream detected. Scaling video: {temp_input_file.name} -> {scaled_video_path}")
            scale_video(temp_input_file.name, scaled_video_path, 25 * 1024 * 1024)  # 25 MiB target size
            source_file = scaled_video_path
        else:
            # Use original file for audio-only files
            logger.info("Audio-only file detected. Skipping video scaling.")
            source_file = temp_input_file.name

        # Extract/convert audio to WAV format
        audio_path = tempfile.mkstemp(suffix='.wav')[1]
        temp_files.append(audio_path)
        ffmpeg_command = [
            'ffmpeg',
            '-i', source_file,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            audio_path
        ]

        logger.info(f"Extracting/converting audio: {' '.join(ffmpeg_command)}")
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)

        logger.info("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        model = whisper.load_model("base", device=device)
        logger.info(f"Whisper model loaded successfully on {device}")

        logger.info("Transcribing audio...")
        result = model.transcribe(audio_path)
        logger.info("Transcription complete")

        srt_content = create_srt_content(result["segments"])

        srt_filename = f"{os.path.splitext(filename)[0]}.srt"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.srt') as temp_srt_file:
            temp_files.append(temp_srt_file.name)
            temp_srt_file.write(srt_content)

        logger.info(f"Uploading SRT to S3: {srt_filename}")
        upload_file(temp_srt_file.name, srt_filename)

        transcription_with_timestamps = [
            f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}] {segment['text']}"
            for segment in result["segments"]
        ]
        full_transcription = "\n".join(transcription_with_timestamps)

        transcription_filename = f"{os.path.splitext(filename)[0]}_transcription.txt"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_transcription_file:
            temp_files.append(temp_transcription_file.name)
            temp_transcription_file.write(full_transcription)

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
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_report_file:
            temp_files.append(temp_report_file.name)
            temp_report_file.write(report)

        logger.info(f"Uploading analysis report to S3: {report_filename}")
        upload_file(temp_report_file.name, report_filename)

        # Create plain text output (without timestamps)
        plain_text = result["text"]
        plain_text_filename = f"{os.path.splitext(filename)[0]}_plain.txt"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_plain_file:
            temp_files.append(temp_plain_file.name)
            temp_plain_file.write(plain_text)

        logger.info(f"Uploading plain text to S3: {plain_text_filename}")
        upload_file(temp_plain_file.name, plain_text_filename)

        # Update job with completion data while preserving original fields
        job_data = jobs[job_id].copy()  # Preserve original job data
        job_data.update({
            'status': 'completed',
            'result': {
                'message': 'File processed, transcribed, and analyzed successfully',
                'original_filename': filename,
                'original_file_url': get_public_url(filename),
                'transcription_filename': transcription_filename,
                'transcription_url': get_public_url(transcription_filename),
                'srt_filename': srt_filename,
                'srt_url': get_public_url(srt_filename),
                'report_filename': report_filename,
                'report_url': get_public_url(report_filename),
                'plain_text_filename': plain_text_filename,
                'plain_text_url': get_public_url(plain_text_filename)
            }
        })
        jobs[job_id] = job_data
        logger.info(f"Job {job_id} completed. Sending webhook alert.")
        send_webhook_alert(job_id, 'completed', jobs[job_id]['result'])
        logger.info(f"Webhook alert sent for job {job_id}")

    except Exception as e:
        logger.error(f"An error occurred in process_audio: {str(e)}")
        # Update job with error while preserving original job data
        if job_id in jobs:
            job_data = jobs[job_id].copy()
            job_data.update({'status': 'failed', 'error': str(e)})
            jobs[job_id] = job_data
        else:
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

    logger.info(f"Process completed for job {job_id}")
         
def send_webhook_alert(job_id, status, result=None):
    if not WEBHOOK_URL:
        logger.warning(f"Webhook URL not set. Skipping webhook alert for job {job_id}.")
        return

    job = jobs.get(job_id, {})
    
    # Log the job data for debugging
    logger.info(f"Job data for {job_id}: {job}")
    
    # Always include these fields for consistent n8n processing
    payload = {
        'job_id': job_id,
        'status': status,
        'is_repurpose': job.get('is_repurpose', False),
        'email': job.get('email', None)
    }
    
    # Include repurpose_message only if it exists
    if job.get('repurpose_message'):
        payload['repurpose_message'] = job.get('repurpose_message')
    
    # Include video_url only if it exists (original YouTube URL)
    if job.get('video_url'):
        payload['video_url'] = job.get('video_url')
        logger.info(f"Including video_url in webhook payload: {job.get('video_url')}")
    else:
        logger.info("No video_url found in job data, not including in webhook payload")
    
    if result:
        payload['result'] = result

    # Log the complete payload being sent
    logger.info(f"Webhook payload for job {job_id}: {json.dumps(payload, indent=2)}")
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