# Kids Writing Assistant API

A simple Flask API server for the Kids Writing Assistant application.

## Local Development

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the server:
```
python app.py
```

## Deployment to Render

### Deploy to Render Free Tier

1. Create a new Web Service on Render
2. Connect this GitHub repository
3. Use the following settings:
   - **Name**: kids-writing-api (or your preferred name)
   - **Environment**: Python 3
   - **Region**: Select the region closest to your users
   - **Branch**: main
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free

4. Add the following environment variable:
   - Key: `PYTHON_VERSION`, Value: `3.10.7`

5. Click "Create Web Service"

After deployment, your API will be available at the URL Render provides (e.g., `https://kids-writing-api.onrender.com`).

## API Endpoints

- `GET /api/test` - Test if the API is running
- `POST /api/analyze` - Analyze text and return emotions
- `GET /api/images/<filename>` - Get an image by filename 