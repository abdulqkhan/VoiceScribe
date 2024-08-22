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
import concurrent.futures
from urllib.error import URLError
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.silence import detect_nonsilent
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

def stream_transcribe_chunks(audio_chunks):
    total_chunks = len(audio_chunks)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_workers = torch.cuda.device_count() if device == "cuda" else os.cpu_count()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        results = [None] * total_chunks
        next_chunk_to_process = 0
        
        while next_chunk_to_process < total_chunks or futures:
            # Submit new tasks
            while next_chunk_to_process < total_chunks and len(futures) < max_workers:
                future = executor.submit(transcribe_chunk, audio_chunks[next_chunk_to_process], next_chunk_to_process, total_chunks, device)
                futures[future] = next_chunk_to_process
                next_chunk_to_process += 1
            
            # Process completed tasks
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                chunk_index, result = future.result()
                results[chunk_index] = result
                del futures[future]
                
                # Report progress
                progress = (chunk_index + 1) / total_chunks * 100
                logger.info(f"Transcription progress: {progress:.2f}%")
    
    return [r for r in results if r is not None]
def split_audio_file(audio_path, chunk_duration=300):  # chunk_duration in seconds (5 minutes)
    audio = AudioSegment.from_wav(audio_path)
    total_duration = len(audio) / 1000  # Duration in seconds
    chunks = []

    for i in range(0, int(total_duration), chunk_duration):
        start = i * 1000  # pydub works with milliseconds
        end = (i + chunk_duration) * 1000
        chunk = audio[start:end]
        
        chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        chunk.export(chunk_file.name, format="wav")
        chunks.append(chunk_file.name)

    return chunks, total_duration
def transcribe_chunk(chunk_path, chunk_index, total_chunks, device):
    logger.info(f"Transcribing chunk {chunk_index+1}/{total_chunks}")
    retries = 3
    for attempt in range(retries):
        try:
            model = whisper.load_model("base", device=device)
            result = model.transcribe(chunk_path)
            os.unlink(chunk_path)  # Remove the chunk file after processing
            return chunk_index, result
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Error transcribing chunk {chunk_index+1} (attempt {attempt+1}): {str(e)}. Retrying...")
            else:
                logger.error(f"Failed to transcribe chunk {chunk_index+1} after {retries} attempts: {str(e)}")
                return chunk_index, None
def adaptive_split_audio(audio_path, min_silence_len=1000, silence_thresh=-40, min_chunk_size=10000, max_chunk_size=30000):
    audio = AudioSegment.from_wav(audio_path)
    non_silent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    chunks = []
    start = 0
    for non_silent_start, non_silent_end in non_silent_ranges:
        if non_silent_start - start > max_chunk_size:
            # If silence is too long, create a chunk
            chunk = audio[start:non_silent_start]
            chunks.append(chunk)
            start = non_silent_start
        elif non_silent_end - start > max_chunk_size:
            # If chunk would be too long, split at a silence point
            split_point = start + max_chunk_size
            chunk = audio[start:split_point]
            chunks.append(chunk)
            start = split_point
    
    # Add the last chunk
    if len(audio) - start > min_chunk_size:
        chunks.append(audio[start:])
    
    # Save chunks to temporary files
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        chunk.export(chunk_file.name, format="wav")
        chunk_files.append(chunk_file.name)
    
    return chunk_files, len(audio) / 1000.0  # Return chunk files and total duration in seconds
def parallel_transcribe_chunks(audio_chunks):
    total_chunks = len(audio_chunks)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_chunk = {executor.submit(transcribe_chunk, chunk, i, total_chunks): i 
                           for i, chunk in enumerate(audio_chunks)}
        results = []
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                result = future.result()
                if result is not None:
                    results.append((chunk_index, result))
            except Exception as exc:
                logger.error(f"Chunk {chunk_index} generated an exception: {exc}")
    
    # Sort results by chunk index to maintain correct order
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results if r[1] is not None]

def process_audio(job_id, filename):
    try:
        file_url = get_public_url(filename)
        logger.info(f"Starting process for file: {file_url}")
        
        # Stream the file from S3
        s3_object = s3_client.get_object(Bucket=S3_BUCKET, Key=filename)
        file_stream = io.BytesIO(s3_object['Body'].read())
        
        # Determine the file type and convert to WAV if necessary
        file_extension = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
            if file_extension != '.wav':
                logger.info(f"Converting file to WAV")
                audio = AudioSegment.from_file(file_stream, format=file_extension[1:])
                audio.export(temp_wav_file.name, format="wav")
            else:
                temp_wav_file.write(file_stream.getvalue())
            audio_path = temp_wav_file.name
        
        logger.info("Adaptively splitting audio file into chunks...")
        audio_chunks, total_duration = adaptive_split_audio(audio_path)
        logger.info(f"Audio file split into {len(audio_chunks)} chunks")

        logger.info("Transcribing audio chunks in parallel...")
        results = stream_transcribe_chunks(audio_chunks)
        logger.info("Parallel transcription of all chunks complete")

        # Combine results from all chunks
        combined_segments = []
        combined_text = []
        time_offset = 0
        for result in results:
            combined_text.append(result["text"])
            for segment in result["segments"]:
                adjusted_segment = segment.copy()
                adjusted_segment["start"] += time_offset
                adjusted_segment["end"] += time_offset
                combined_segments.append(adjusted_segment)
            time_offset += result["segments"][-1]["end"] if result["segments"] else 0

        combined_result = {
            "text": " ".join(combined_text),
            "segments": combined_segments
        }
        
        # Create and upload SRT content
        srt_content = create_srt_content(combined_result["segments"])
        srt_filename = f"{os.path.splitext(filename)[0]}.srt"
        upload_string_to_s3(srt_content, srt_filename)
        
        # Create and upload full transcription
        full_transcription = "\n".join([f"[{format_timestamp(seg['start'])} - {format_timestamp(seg['end'])}] {seg['text']}" for seg in combined_result["segments"]])
        transcription_filename = f"{os.path.splitext(filename)[0]}_transcription.txt"
        upload_string_to_s3(full_transcription, transcription_filename)
        
        # Generate and upload analysis report
        report = generate_analysis_report(combined_result, total_duration)
        report_filename = f"{os.path.splitext(filename)[0]}_report.txt"
        upload_string_to_s3(report, report_filename)

        logger.info("Cleaning up temporary files...")
        os.unlink(audio_path)
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
        
    except Exception as e:
        logger.error(f"An error occurred in process_audio: {str(e)}")
        jobs[job_id] = {'status': 'failed', 'error': str(e)}
        send_webhook_alert(job_id, 'failed', {'error': str(e)})
        
def upload_string_to_s3(content, filename):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(content)
        temp_file.flush()
        logger.info(f"Uploading {filename} to S3")
        upload_file(temp_file.name, filename)
    os.unlink(temp_file.name)        
        
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