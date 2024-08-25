from app.routes import app
from config.settings import API_KEY


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
if not API_KEY:
    raise ValueError("API_KEY must be set in config/settings.py")