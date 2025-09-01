# VoiceScribe API Documentation

## Base URL
```
http://your-server:5000
```

## Authentication
Most endpoints require API key authentication. Include the API key in the request headers:
```
X-API-Key: your_api_key_here
```

## Endpoints

### 1. Health Check
**GET** `/health`

Check if the service is running.

**Response:**
```json
{
  "status": "healthy"
}
```

---

### 2. Check API Key
**POST** `/check_api_key`

Validate an API key.

**Request Body:**
```json
{
  "api_key": "your_api_key_here"
}
```

**Response:**
- Success (200):
```json
{
  "valid": true
}
```
- Invalid (401):
```json
{
  "valid": false
}
```

---

### 3. Upload File
**POST** `/upload` (Requires Authentication)

Upload a file to S3 storage.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File upload with field name `file`

**Response (200):**
```json
{
  "message": "File uploaded successfully",
  "file_url": "https://s3-endpoint/bucket/filename"
}
```

---

### 4. Upload and Process
**POST** `/upload_and_process` (Requires Authentication)

Upload a file and start transcription process with email tracking.

**Request:**
- Content-Type: `multipart/form-data`
- Fields:
  - `file` (required): The audio/video file to process
  - `email` (required): Email address for notifications

**Response (202):**
```json
{
  "message": "File uploaded and processing started",
  "job_id": "uuid-string",
  "filename": "example.mp4",
  "email": "user@example.com"
}
```

**Webhook Payload (on completion):**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "is_repurpose": false,
  "email": "user@example.com",
  "result": {
    "message": "File processed, transcribed, and analyzed successfully",
    "original_filename": "example.mp4",
    "original_file_url": "https://...",
    "transcription_filename": "example_transcription.txt",
    "transcription_url": "https://...",
    "srt_filename": "example.srt",
    "srt_url": "https://...",
    "report_filename": "example_report.txt",
    "report_url": "https://...",
    "plain_text_filename": "example_plain.txt",
    "plain_text_url": "https://..."
  }
}
```

---

### 5. Repurpose Content (NEW)
**POST** `/repurpose` (Requires Authentication)

Upload a file, transcribe it, and mark it for AI-powered content repurposing.

**Request:**
- Content-Type: `multipart/form-data`
- Fields:
  - `file` (required): The audio/video file to process
  - `email` (required): Email address for notifications
  - `repurpose_message` (required): Instructions for content transformation

**Example Request:**
```bash
curl -X POST "http://localhost:5000/repurpose" \
  -H "X-API-Key: your_api_key" \
  -F "file=@audio.mp3" \
  -F "email=user@example.com" \
  -F "repurpose_message=Convert this content for software developers"
```

**Response (202):**
```json
{
  "message": "File uploaded and repurpose processing started",
  "job_id": "uuid-string",
  "filename": "audio.mp3",
  "email": "user@example.com",
  "repurpose_message": "Convert this content for software developers"
}
```

**Webhook Payload (on completion):**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "is_repurpose": true,
  "repurpose_message": "Convert this content for software developers",
  "email": "user@example.com",
  "result": {
    "message": "File processed, transcribed, and analyzed successfully",
    "original_filename": "audio.mp3",
    "original_file_url": "https://...",
    "transcription_filename": "audio_transcription.txt",
    "transcription_url": "https://...",
    "srt_filename": "audio.srt",
    "srt_url": "https://...",
    "report_filename": "audio_report.txt",
    "report_url": "https://...",
    "plain_text_filename": "audio_plain.txt",
    "plain_text_url": "https://..."
  }
}
```

---

### 6. Convert and Transcribe
**POST** `/convert_and_transcribe` (Requires Authentication)

Start transcription for an already uploaded file.

**Request Body:**
```json
{
  "filename": "example.mp4",
  "file_source": "S3"  // Optional, defaults to "S3"
}
```

**Response (202):**
```json
{
  "message": "Task started successfully",
  "job_id": "uuid-string"
}
```

---

### 7. Get Job Status
**GET** `/job_status/<job_id>`

Check the status of a processing job.

**Response:**
- Processing (202):
```json
{
  "status": "processing"
}
```
- Completed (200):
```json
{
  "message": "File processed, transcribed, and analyzed successfully",
  "original_filename": "...",
  "transcription_url": "...",
  // ... other URLs
}
```
- Failed (500):
```json
{
  "error": "Error message"
}
```
- Not Found (404):
```json
{
  "error": "Job not found"
}
```

---

### 8. Get Active Jobs
**GET** `/jobs/active`

List all active and queued jobs.

**Response (200):**
```json
{
  "active_jobs": [
    {
      "job_id": "uuid-string",
      "status": "processing",
      "filename": "example.mp4",
      "email": "user@example.com"
    }
  ],
  "count": 1
}
```

---

## Webhook Integration

The application sends webhook notifications when jobs complete or fail. Configure the webhook URL in your `.env` file:

```
WEBHOOK_URL=https://your-webhook-endpoint.com/webhook
```

### Webhook Payload Fields

All webhook payloads include:
- `job_id`: Unique identifier for the job
- `status`: "completed" or "failed"
- `is_repurpose`: Boolean indicating if this is a repurpose job
- `email`: Email address associated with the job (may be null)

For repurpose jobs, additionally:
- `repurpose_message`: The transformation instructions

For completed jobs:
- `result`: Object containing all generated file URLs

### n8n Integration

The webhook payloads are designed for easy integration with n8n workflows:

1. **Regular transcription jobs**: `is_repurpose = false`
2. **Repurpose jobs**: `is_repurpose = true` with `repurpose_message`

Your n8n workflow can check the `is_repurpose` field to determine whether to:
- Process normally (regular transcription)
- Fetch transcription and apply AI transformation (repurpose job)

## File Types Supported

- Video: mp4, mov, avi, wmv, flv, webm
- Audio: mp3, wav, m4a

## File Size Limits

- Maximum file size: 25 MB (configurable via MAX_FILE_SIZE in settings)
- Files larger than 25 MB will be automatically scaled down

## Error Codes

- 200: Success
- 202: Accepted (processing started)
- 400: Bad Request (missing or invalid parameters)
- 401: Unauthorized (invalid API key)
- 404: Not Found
- 500: Internal Server Error