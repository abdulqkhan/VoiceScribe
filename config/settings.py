import os
from dotenv import load_dotenv

load_dotenv()

S3_ENDPOINT = os.getenv('S3_ENDPOINT')
S3_BUCKET = os.getenv('S3_BUCKET')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
WEBHOOK_URL = os.getenv('WEBHOOK_URL')
API_KEY = os.getenv('API_KEY') 
ALLOWED_FILE_SOURCES = os.getenv('ALLOWED_FILE_SOURCES')
MAX_FILE_SIZE= 25 * 1024 * 1024  # 25 MiB
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'wmv', 'flv', 'webm'}