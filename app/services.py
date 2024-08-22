import os
import io
import tempfile
import boto3
from botocore.client import Config
import whisper
import requests
import time
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

def generate_analysis_report(combined_result, total_duration):
    report = f"Video Analysis Report\n"
    report += f"====================\n\n"
    report += f"Total Duration: {format_timestamp(total_duration)}\n"
    report += f"Total Segments: {len(combined_result['segments'])}\n\n"

    # Analyze silent parts
    silent_parts = []
    for i, segment in enumerate(combined_result['segments']):
        if i > 0:
            gap = segment['start'] - combined_result['segments'][i-1]['end']
            if gap > 1:  # Consider gaps longer than 1 second as silent parts
                silent_parts.append({
                    'start': format_timestamp(combined_result['segments'][i-1]['end']),
                    'end': format_timestamp(segment['start']),
                    'duration': gap
                })

    if silent_parts:
        report += f"Silent Parts (>1 second):\n"
        for i, silent in enumerate(silent_parts):
            report += f"  - [{silent['start']} - {silent['end']}] Duration: {silent['duration']:.2f} seconds\n"
        report += "\n"

    # Analyze repeated sentences
    sentence_occurrences = {}
    for segment in combined_result['segments']:
        text = segment['text'].strip().lower()
        if text in sentence_occurrences:
            sentence_occurrences[text].append(segment)
        else:
            sentence_occurrences[text] = [segment]

    repeated_sentences = {text: occurrences for text, occurrences in sentence_occurrences.items() if len(occurrences) > 1}

    if repeated_sentences:
        report += f"Repeated Sentences:\n"
        for text, occurrences in repeated_sentences.items():
            report += f"  - '{text}' (Repeated {len(occurrences)} times)\n"
            for occur in occurrences:
                report += f"    [{format_timestamp(occur['start'])} - {format_timestamp(occur['end'])}]\n"
        report += "\n"

    # You can add more analysis here as needed

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
def sequential_transcribe_chunks(audio_chunks, job_id):
    total_chunks = len(audio_chunks)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        model = whisper.load_model("base", device=device)
        logger.info(f"Whisper model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {str(e)}")
        raise

    results = []
    for i, chunk_path in enumerate(audio_chunks):
        chunk_start_time = time.time()
        logger.info(f"Starting transcription of chunk {i+1}/{total_chunks}")
        try:
            result = model.transcribe(chunk_path)
            results.append(result)
            os.unlink(chunk_path)  # Remove the chunk file after processing
            chunk_end_time = time.time()
            chunk_duration = chunk_end_time - chunk_start_time
            logger.info(f"Completed transcription of chunk {i+1}/{total_chunks} in {chunk_duration:.2f} seconds")
            
            # Update job status
            progress = ((i + 1) / total_chunks) * 100
            update_job_status(job_id, f"Transcribing: {progress:.2f}% complete")
            
        except Exception as e:
            logger.error(f"Error transcribing chunk {i+1}: {str(e)}")
            # Continue with the next chunk instead of stopping the entire process
    
    return results

def process_audio(job_id, filename):
    try:
        file_url = get_public_url(filename)
        logger.info(f"Starting process for file: {file_url}")
        
        # Initialize job status
        jobs[job_id] = {'status': 'started'}
        update_job_status(job_id, "Started")
        
        # Stream the file from S3
        try:
            s3_object = s3_client.get_object(Bucket=S3_BUCKET, Key=filename)
            file_stream = io.BytesIO(s3_object['Body'].read())
        except Exception as e:
            logger.error(f"Error streaming file from S3: {str(e)}")
            raise

        # Determine the file type and convert to WAV if necessary
        file_extension = os.path.splitext(filename)[1].lower()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
                if file_extension != '.wav':
                    logger.info(f"Converting file to WAV")
                    update_job_status(job_id, "Converting to WAV")
                    audio = AudioSegment.from_file(file_stream, format=file_extension[1:])
                    audio.export(temp_wav_file.name, format="wav")
                else:
                    temp_wav_file.write(file_stream.getvalue())
                audio_path = temp_wav_file.name
        except Exception as e:
            logger.error(f"Error converting file to WAV: {str(e)}")
            raise

        logger.info("Adaptively splitting audio file into chunks...")
        update_job_status(job_id, "Splitting audio")
        try:
            audio_chunks, total_duration = adaptive_split_audio(audio_path)
            logger.info(f"Audio file split into {len(audio_chunks)} chunks")
        except Exception as e:
            logger.error(f"Error splitting audio into chunks: {str(e)}")
            raise

        logger.info("Transcribing audio chunks sequentially...")
        update_job_status(job_id, "Transcribing")
        try:
            results = sequential_transcribe_chunks(audio_chunks, job_id)
            logger.info("Sequential transcription of all chunks complete")
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise

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
        update_job_status(job_id, "Creating SRT")
        try:
            srt_content = create_srt_content(combined_result["segments"])
            srt_filename = f"{os.path.splitext(filename)[0]}.srt"
            upload_string_to_s3(srt_content, srt_filename)
        except Exception as e:
            logger.error(f"Error creating or uploading SRT: {str(e)}")
            raise

        # Create and upload full transcription
        update_job_status(job_id, "Creating full transcription")
        try:
            full_transcription = "\n".join([f"[{format_timestamp(seg['start'])} - {format_timestamp(seg['end'])}] {seg['text']}" for seg in combined_result["segments"]])
            transcription_filename = f"{os.path.splitext(filename)[0]}_transcription.txt"
            upload_string_to_s3(full_transcription, transcription_filename)
        except Exception as e:
            logger.error(f"Error creating or uploading full transcription: {str(e)}")
            raise

        # Generate and upload analysis report
        update_job_status(job_id, "Generating analysis report")
        try:
            report = generate_analysis_report(combined_result, total_duration)
            report_filename = f"{os.path.splitext(filename)[0]}_report.txt"
            upload_string_to_s3(report, report_filename)
        except Exception as e:
            logger.error(f"Error generating or uploading analysis report: {str(e)}")
            raise

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

def update_job_status(job_id, status):
    if job_id in jobs:
        jobs[job_id]['status'] = status
        logger.info(f"Job {job_id} status updated: {status}")
    else:
        logger.error(f"Attempted to update status for non-existent job {job_id}")
    # Optionally, you could send a webhook update here as well
    
        
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