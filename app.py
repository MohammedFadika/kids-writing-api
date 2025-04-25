import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import json
import time
from PIL import Image
import traceback

app = Flask(__name__)

# Enable CORS for all routes with additional options
CORS(app, resources={r"/api/*": {"origins": "*", "allow_headers": ["Content-Type"], "methods": ["GET", "POST"]}})

# Base directory for images
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print(f"Base directory: {BASE_DIR}")

# Create sample content
SAMPLE_IMAGES = {}
SAMPLE_RESPONSES = {
    "happy": {
        "emotions": {
            "joy": True,
            "sadness": False,
            "anger": False, 
            "fear": False
        },
        "feedback": ("I love how you expressed happiness in your writing! To make your story even more engaging, try describing what made you happy and how that happiness felt in your body. Did you smile, laugh, or jump with excitement?", True),
        "imagePath": "happy_image.txt"
    },
    "sad": {
        "emotions": {
            "joy": False,
            "sadness": True,
            "anger": False,
            "fear": False
        },
        "feedback": ("I can feel the sadness in your writing. You've done a good job expressing this emotion. To make your writing even more powerful, try describing how sadness feels in your body. Did your shoulders slump? Did you feel heavy?", True),
        "imagePath": "sad_image.txt"
    }
}

# Create a simple text file to serve as our test image
def create_test_file():
    with open(os.path.join(BASE_DIR, "test_image.txt"), "w") as f:
        f.write("This is a test image placeholder")
    print(f"Created test file at {os.path.join(BASE_DIR, 'test_image.txt')}")
    
    # Create sample image files
    with open(os.path.join(BASE_DIR, "happy_image.txt"), "w") as f:
        f.write("This is a happy image placeholder")
    
    with open(os.path.join(BASE_DIR, "sad_image.txt"), "w") as f:
        f.write("This is a sad image placeholder")

@app.route('/api/test', methods=['GET'])
def test_api():
    return jsonify({'status': 'API is working', 'version': '1.0'}), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    try:
        print("Received API request:", request.json)
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Check if we're in production environment (Render)
        is_production = os.environ.get('RENDER_ENVIRONMENT') == 'true'
        
        # Simple text-based emotion detection
        is_happy = any(word in text.lower() for word in ["happy", "joy", "glad", "excited", "fun", "smile"])
        is_sad = any(word in text.lower() for word in ["sad", "unhappy", "cry", "lost", "miss", "alone"])
        
        # Create emotion object
        emotions = {
            'joy': is_happy,
            'sadness': is_sad,
            'anger': False,
            'fear': False
        }
        
        # Generate feedback based on emotions
        if is_happy:
            feedback = ("I love how you expressed happiness in your writing! To make your story even more engaging, try describing what made you happy and how that happiness felt in your body. Did you smile, laugh, or jump with excitement?", True)
        elif is_sad:
            feedback = ("I can feel the sadness in your writing. You've done a good job expressing this emotion. To make your writing even more powerful, try describing how sadness feels in your body. Did your shoulders slump? Did you feel heavy?", True)
        else:
            feedback = ("You've written a nice piece! To make it more engaging, try adding details about how you felt. What emotions were you experiencing? How did those emotions feel in your body?", True)
        
        # For production environment, use placeholder image instead of trying to generate one
        if is_production:
            # Create a simple colored rectangle as placeholder
            image_filename = f"placeholder_image_{int(time.time())}.png"
            image_path = os.path.join(BASE_DIR, image_filename)
            
            # Create a simple image (blue for sad, yellow for happy, green for neutral)
            color = (255, 255, 0) if is_happy else (0, 0, 255) if is_sad else (0, 255, 0)
            img = Image.new('RGB', (500, 300), color=color)
            img.save(image_path)
        else:
            # Try to use Stable Diffusion locally if available
            try:
                # Code to generate image with Stable Diffusion would go here
                # For simplicity, we'll use the same placeholder approach
                image_filename = f"placeholder_image_{int(time.time())}.png"
                image_path = os.path.join(BASE_DIR, image_filename)
                
                color = (255, 255, 0) if is_happy else (0, 0, 255) if is_sad else (0, 255, 0)
                img = Image.new('RGB', (500, 300), color=color)
                img.save(image_path)
            except Exception as e:
                print(f"Local image generation error: {e}")
                # Fallback to placeholder
                image_filename = f"placeholder_image_{int(time.time())}.png"
                image_path = os.path.join(BASE_DIR, image_filename)
                
                color = (255, 255, 0) if is_happy else (0, 0, 255) if is_sad else (0, 255, 0)
                img = Image.new('RGB', (500, 300), color=color)
                img.save(image_path)
        
        response = {
            'emotions': emotions,
            'feedback': feedback,
            'imagePath': image_filename
        }
        
        print("Sending response:", response)
        return jsonify(response)
    except Exception as e:
        print("Error processing request:", str(e))
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/<filename>')
def serve_image(filename):
    """Serve images from the server's filesystem"""
    print(f"Serving image: {filename}")
    print(f"Looking for image in: {BASE_DIR}")
    print(f"Full path: {os.path.join(BASE_DIR, filename)}")
    print(f"File exists: {os.path.exists(os.path.join(BASE_DIR, filename))}")
    return send_from_directory(BASE_DIR, filename)

if __name__ == '__main__':
    # Create our test files and sample responses
    create_test_file()
    
    # Get port from environment variable or default to 5001
    port = int(os.environ.get('PORT', 5001))
    
    # In production, the host should be '0.0.0.0' to listen on all interfaces
    host = '0.0.0.0'
    
    print(f"Starting Flask server at http://{host}:{port}")
    print(f"Base directory: {BASE_DIR}")
    
    # In production (Render), Flask's debug mode should be off
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(debug=debug, port=port, host=host) 