<div align="center">
  <a href="https://cloud-station.io">
    <img src="https://server.cloud-station.io/cloudstation/cs_icon.png" alt="CloudStation Logo" width="50">
  </a>
  <h3 align="center">CloudStation Flask Audio Processing API</h3>
  <p align="center">
    Deploy your Flask application for audio processing and transcription seamlessly with CloudStation.
    <br />
    <a href="https://cloud-station.io">Visit CloudStation</a> ·
    <a href="https://documentation.cloud-station.io/s/ce6e8846-8aec-4337-a850-5188b6dc6d6e">Documentation</a> ·
    <a href="https://blog.cloud-station.io">Blog</a>
  </p>
</div>

## Overview

This repository showcases a Flask application for audio file processing, transcription, and analysis. It leverages FFmpeg for video scaling and audio extraction, and uses the Whisper model for speech recognition. The application can be effortlessly deployed on CloudStation, allowing you to focus on writing code without worrying about infrastructure.

## Features

- Video scaling using FFmpeg
- Audio extraction from video files
- Speech recognition using OpenAI's Whisper model
- Silence detection and analysis
- Transcription with timestamps
- SRT file generation
- Automated deployment using Docker
- Integration with MinIO or S3 for file storage

## Technologies Used

- Flask: Web framework for building the API
- FFmpeg: Multimedia framework for handling video and audio files
- Whisper: State-of-the-art speech recognition model by OpenAI
- PyDub: Audio processing library for silence detection
- Boto3: AWS SDK for Python, used for S3 interactions
- Docker: Containerization for easy deployment and scaling
- MinIO: High-performance object storage compatible with Amazon S3 API

## GPU Support

This application is designed to take advantage of GPU acceleration if available. When a GPU is detected, the Whisper model will automatically use it for faster processing. If no GPU is available, the application will fallback to CPU processing.

## Getting Started

Follow these steps to get your Flask Audio Processing API running on CloudStation.

### Prerequisites

1. Ensure you have a CloudStation account. If you don't have one yet, sign up [here](https://www.cloud-station.io/signup).

2. Deploy a MinIO instance:
   This application requires a MinIO instance or S3-compatible storage for file operations. CloudStation offers a one-click deployment for MinIO:
   
   a. Go to the [CloudStation Template Store](https://app.cloud-station.io/template-store/minio)
   b. Click on "Deploy" for the MinIO template
   c. Follow the prompts to complete the deployment

  After the deployment, go to the Variables tab of your deployed MinIO instance and note down the following credentials:

- MINIO_ROOT_USER
- MINIO_ROOT_PASSWORD
- MINIO_SERVER_URL

You'll need these for configuring the Flask application.

### Quick Deploy

To deploy this application instantly, click the button below:

<p align="center">
  <a href="https://app.cloud-station.io/template-store/minio">
    <img src="https://server.cloud-station.io/cloudstation/Deploy_TO_CS.gif" alt="Deploy to CloudStation" width="250"">
  </a>
</p>

### Deployment Process

This application uses a Dockerfile for automated builds and deployments. When you deploy to CloudStation:

1. The Dockerfile in the repository is used to build the container image.
2. The multi-stage build process optimizes the final image size.
3. All necessary dependencies, including FFmpeg, are installed in the container.
4. The application is set up to run using Gunicorn for improved performance.

You don't need to manually build or push any images - CloudStation handles the entire process automatically based on the Dockerfile.

### Environment Variables

The following environment variables are required:

```
API_KEY=somekey
S3_ACCESS_KEY=MINIO_ROOT_USER
S3_BUCKET=MINIO_BUCKET
S3_ENDPOINT=MINIO_SERVER_URL
S3_SECRET_KEY=MINIO_ROOT_PASSWORD
WEBHOOK_URL=https://webhook
MAX_FILE_SIZE=26214400
```

Replace the S3_* variables with the values from your deployed MinIO instance. The `WEBHOOK_URL` is used to send notifications once the conversion is complete. `MAX_FILE_SIZE` is set to 25MB (26,214,400 bytes) by default.

### Authentication

All endpoints (except `/` and `/health`) require authentication using an API key. Include the API key in the `Authorization` header of your requests.

## API Endpoints

- `GET /`: Returns the index page (upload.html)
- `POST /check_api_key`: Validates the provided API key
- `POST /upload`: Handles file uploads
- `POST /convert_and_transcribe`: Initiates audio conversion and transcription
- `GET /job_status/<job_id>`: Retrieves the status of a conversion job
- `GET /health`: Health check endpoint

## Usage

1. **Check API Key**
   ```
   POST /check_api_key
   Content-Type: application/json
   
   {
     "api_key": "your_api_key_here"
   }
   ```

2. **Upload File**
   ```
   POST /upload
   Authorization: your_api_key_here
   Content-Type: multipart/form-data
   
   file: [your_audio_or_video_file]
   ```

3. **Convert and Transcribe**
   ```
   POST /convert_and_transcribe
   Authorization: your_api_key_here
   Content-Type: application/json
   
   {
     "filename": "your_uploaded_file.mp4",
   }
   ```

4. **Check Job Status**
   ```
   GET /job_status/<job_id>
   Authorization: your_api_key_here
   ```

5. **Health Check**
   ```
   GET /health
   ```

## Processing Pipeline

1. The uploaded file is scaled down using FFmpeg if it exceeds the maximum file size.
2. Audio is extracted from the video file.
3. The Whisper model transcribes the audio.
4. Silence analysis is performed on the audio.
5. An SRT file is generated from the transcription.
6. An analysis report is created, including information about silent parts, repeated sentences, and segments flagged for review.
7. All generated files (SRT, transcription, and report) are uploaded to MinIO/S3.
8. A webhook notification is sent upon job completion.

## Customization

To modify the application, make changes in your forked repository and push them. CloudStation will automatically rebuild and redeploy your application using the Dockerfile.

## Contributing

We welcome contributions to enhance this example application. Feel free to fork the repository, create a feature branch, and submit a pull request.

## Support

For support, visit our [Help Center](https://documentation.cloud-station.io/s/ce6e8846-8aec-4337-a850-5188b6dc6d6e) or reach out via [Slack](https://join.slack.com/t/cloudstationio/shared_invite/zt-20kougo40-Kd1196QzZ7bwUA0oPfZORA).

## Connect with Us

<p align="center">
  <a href="https://www.cloud-station.io/">Website</a> · 
  <a href="https://twitter.com/CloudStation_io">Twitter</a> · 
  <a href="https://join.slack.com/t/cloudstationio/shared_invite/zt-20kougo40-Kd1196QzZ7bwUA0oPfZORA">Slack</a>
</p>
