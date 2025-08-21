# Deploying VoiceScribe to VPS with Coolify

This guide walks you through deploying VoiceScribe to your VPS using Coolify.

## Prerequisites

- VPS with Coolify installed
- Domain name (optional, but recommended)
- Git repository access (fork this repo first)

## Deployment Methods

### Method 1: Using Docker Compose (Recommended)

This method includes MinIO for storage, making it a complete solution.

#### Step 1: Fork and Prepare Repository

1. Fork this repository to your GitHub account
2. Clone to your local machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/VoiceScribe.git
   cd VoiceScribe
   ```

#### Step 2: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your values:
   ```bash
   # IMPORTANT: Change these values!
   API_KEY=generate_a_secure_random_key_here
   MINIO_ROOT_USER=your_minio_admin
   MINIO_ROOT_PASSWORD=secure_password_min_8_chars
   ```

3. Commit your docker-compose.yml (but NOT the .env file):
   ```bash
   git add docker-compose.yml .env.example DEPLOYMENT_COOLIFY.md
   git commit -m "Add Coolify deployment configuration"
   git push origin main
   ```

#### Step 3: Deploy with Coolify

1. **Log into Coolify** on your VPS

2. **Create New Project**:
   - Click "New Project"
   - Give it a name like "VoiceScribe"

3. **Add New Resource**:
   - Select "Docker Compose"
   - Choose your server

4. **Configure Source**:
   - Select "GitHub" (or your git provider)
   - Repository: `YOUR_USERNAME/VoiceScribe`
   - Branch: `main`
   - Build Path: `/` (root)

5. **Environment Variables**:
   In Coolify's environment variables section, add:
   ```
   API_KEY=your_secure_api_key
   S3_ACCESS_KEY=minioadmin
   S3_SECRET_KEY=your_secure_password
   S3_BUCKET=voicescribe
   S3_ENDPOINT=http://minio:9000
   WEBHOOK_URL=https://your-webhook.com (optional)
   MAX_FILE_SIZE=26214400
   MINIO_ROOT_USER=minioadmin
   MINIO_ROOT_PASSWORD=your_secure_password
   ```

6. **Network Configuration**:
   - Enable "Expose to Internet"
   - Add domain if you have one
   - Coolify will handle SSL automatically with Let's Encrypt

7. **Deploy**:
   - Click "Deploy"
   - Wait for build and deployment to complete

### Method 2: Using Dockerfile Only (External Storage)

If you already have MinIO or S3 storage set up:

#### Step 1: Deploy in Coolify

1. **Create New Resource**:
   - Select "Dockerfile"
   - Choose your server

2. **Configure Source**:
   - Repository: `YOUR_USERNAME/VoiceScribe`
   - Branch: `main`
   - Dockerfile path: `/Dockerfile`

3. **Environment Variables**:
   ```
   API_KEY=your_secure_api_key
   S3_ACCESS_KEY=your_external_minio_access_key
   S3_SECRET_KEY=your_external_minio_secret_key
   S3_BUCKET=voicescribe
   S3_ENDPOINT=https://your-minio-instance.com
   WEBHOOK_URL=https://your-webhook.com (optional)
   MAX_FILE_SIZE=26214400
   ```

4. **Port Configuration**:
   - Set port to `5000`

5. **Deploy**

### Method 3: Using Coolify's Git Integration

1. **In Coolify**:
   - New Resource → GitHub App
   - Connect your GitHub account
   - Select your forked VoiceScribe repo

2. **Build Configuration**:
   - Build Pack: `Dockerfile`
   - Port: `5000`

3. **Add environment variables** as shown above

4. **Deploy**

## Post-Deployment Setup

### 1. Create MinIO Bucket (if using bundled MinIO)

Access MinIO console:
- URL: `http://your-domain:9001` or use Coolify's proxy
- Login with MINIO_ROOT_USER and MINIO_ROOT_PASSWORD

Create bucket:
1. Click "Buckets" → "Create Bucket"
2. Name: `voicescribe`
3. Click "Create"

### 2. Test the Deployment

1. **Health Check**:
   ```bash
   curl https://your-domain/health
   ```

2. **Web Interface**:
   - Navigate to `https://your-domain/`
   - You should see the upload interface

3. **API Test**:
   ```bash
   curl -X POST https://your-domain/check_api_key \
     -H "Content-Type: application/json" \
     -d '{"api_key":"your_api_key"}'
   ```

## Coolify-Specific Settings

### Resource Limits
In Coolify's resource settings:
- Memory: 4GB minimum (for Whisper model)
- CPU: 2 cores minimum

### Health Checks
Coolify automatically uses the health check defined in docker-compose.yml:
- Endpoint: `/health`
- Interval: 30 seconds

### Persistent Storage
For MinIO data persistence:
- Coolify automatically manages volumes
- Data persists across redeployments

### SSL/TLS
- Coolify automatically provisions Let's Encrypt certificates
- Just add your domain in the settings

## Troubleshooting

### Container Won't Start
Check logs in Coolify:
1. Go to your application
2. Click "Logs"
3. Look for error messages

### Out of Memory
- Increase memory limits in Coolify resource settings
- Minimum 4GB recommended for Whisper model

### MinIO Connection Issues
- Ensure MinIO container is running
- Check S3_ENDPOINT is correct (`http://minio:9000` for docker-compose)
- Verify credentials match

### File Upload Failures
- Check MAX_FILE_SIZE environment variable
- Ensure MinIO bucket exists
- Verify S3 credentials

## Monitoring

### In Coolify
- View real-time logs
- Monitor resource usage
- Set up alerts

### Application Logs
- Application logs are available in Coolify's log viewer
- Filter by container name for specific services

## Updating

To update your deployment:
1. Push changes to your GitHub repository
2. In Coolify, click "Redeploy"
3. Coolify will pull latest changes and redeploy

## Security Recommendations

1. **API Key**: Generate a strong, random API key
2. **MinIO Credentials**: Use strong passwords (min 8 characters)
3. **Network**: Consider using Coolify's private network for MinIO
4. **Firewall**: Configure VPS firewall to only allow necessary ports
5. **Updates**: Regularly update the application and dependencies

## Support

- Check Coolify documentation: https://coolify.io/docs
- VoiceScribe issues: Create an issue in the GitHub repository
- Coolify Discord: https://discord.gg/coolify