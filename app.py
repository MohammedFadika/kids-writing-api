import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import json

app = Flask(__name__)

# Enable CORS for all routes with additional options
CORS(app, resources={r"/api/*": {"origins": "*", "allow_headers": ["Content-Type"], "methods": ["GET", "POST"]}})

# Get the absolute path of the current directory for image storage
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Create sample content
SAMPLE_IMAGES = {}
SAMPLE_RESPONSES = {}

# Create a simple text file to serve as our test image
def create_test_file():
    with open(os.path.join(BASE_DIR, "test_image.txt"), "w") as f:
        f.write("This is a test image placeholder")
    print(f"Created test file at {os.path.join(BASE_DIR, 'test_image.txt')}")
    
    # Create a simple happy image response
    SAMPLE_RESPONSES["happy"] = {
        'emotions': {'joy': True, 'sadness': False, 'anger': False, 'fear': False},
        'feedback': "I like how you shared your happiness! To make your joy more vivid, try describing how your happiness felt in your body - maybe you jumped up and down or had a huge smile?",
        'imagePath': 'happy_image.txt'
    }
    
    # Create a simple sad image response
    SAMPLE_RESPONSES["sad"] = {
        'emotions': {'joy': False, 'sadness': True, 'anger': False, 'fear': False},
        'feedback': "I notice the sadness in your writing. Try describing how it feels physically, such as 'chest feeling heavy'.",
        'imagePath': 'sad_image.txt'
    }
    
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
        
        # Simple text-based emotion detection
        is_happy = any(word in text.lower() for word in ["happy", "joy", "glad", "excited", "fun", "smile"])
        is_sad = any(word in text.lower() for word in ["sad", "unhappy", "cry", "lost", "miss", "alone"])
        
        if is_happy:
            response = SAMPLE_RESPONSES["happy"]
        elif is_sad:
            response = SAMPLE_RESPONSES["sad"]
        else:
            # Default to happy for simplicity
            response = SAMPLE_RESPONSES["happy"]
        
        print("Sending response:", response)
        return jsonify(response)
    except Exception as e:
        print("Error processing request:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/<filename>', methods=['GET'])
def serve_image(filename):
    # Return the test file
    if filename.endswith('.txt'):
        return send_from_directory(BASE_DIR, filename)
    return jsonify({'error': 'Image not found'}), 404

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