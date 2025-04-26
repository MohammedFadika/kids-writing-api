import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import json
import time
from PIL import Image
import traceback
import requests
from io import BytesIO
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

app = Flask(__name__)

# Enable CORS for all routes with additional options
CORS(app, resources={r"/api/*": {"origins": "*", "allow_headers": ["Content-Type"], "methods": ["GET", "POST"]}})

# Base directory for images
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print(f"Base directory: {BASE_DIR}")

# Hugging Face model ID for the emotion detection model
EMOTION_MODEL_ID = "Mohvmmed111/kidswrittingemodetect"
print(f"Using emotion model from Hugging Face Hub: {EMOTION_MODEL_ID}")

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

# Load the RoBERTa model for emotion detection
try:
    print("Loading RoBERTa model for emotion detection from Hugging Face Hub...")
    emotion_model = RobertaForSequenceClassification.from_pretrained(EMOTION_MODEL_ID)
    emotion_tokenizer = RobertaTokenizer.from_pretrained(EMOTION_MODEL_ID)
    print("RoBERTa model loaded successfully!")
    
    def detect_emotions(text):
        """Detect emotions using the fine-tuned RoBERTa model"""
        encoding = emotion_tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            output = emotion_model(**encoding)
        predictions = torch.sigmoid(output.logits)
        predictions = predictions > 0.5  # Threshold to get 0 or 1 for each emotion
        emotion_columns = ["anger", "fear", "sadness", "joy"]  # The emotions our model was trained on
        predicted_emotions = dict(zip(emotion_columns, predictions.squeeze().tolist()))
        return predicted_emotions
        
except Exception as e:
    print(f"Error loading RoBERTa model from Hugging Face Hub: {e}")
    print("Falling back to simple keyword-based emotion detection")
    
    def detect_emotions(text):
        """Simple keyword-based emotion detection as fallback"""
        text = text.lower()
        emotions = {
            "anger": any(word in text for word in ["angry", "mad", "furious", "rage", "broke", "hate"]),
            "fear": any(word in text for word in ["scared", "afraid", "fear", "worried", "nervous", "dark"]),
            "sadness": any(word in text for word in ["sad", "unhappy", "cry", "lost", "miss", "alone"]),
            "joy": any(word in text for word in ["happy", "joy", "glad", "excited", "fun", "smile"])
        }
        return emotions

# Function to generate image with Replicate API
def generate_image_with_replicate(prompt, negative_prompt="low quality, blurry, distorted", timeout=90):
    """Generate an image using Replicate's API for Stable Diffusion"""
    
    # Get token from environment
    api_token = os.environ.get('REPLICATE_API_TOKEN')
    if not api_token:
        print("No Replicate API token found. Falling back to placeholder image.")
        return None
    
    # Set up headers
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }
    
    # Stable Diffusion endpoint
    api_url = "https://api.replicate.com/v1/predictions"
    
    # Using Stable Diffusion v1.5
    model_version = "db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"
    
    print(f"Using Replicate API for image generation with prompt: {prompt}")
    
    # Prepare payload for Replicate
    payload = {
        "version": model_version,
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512
        }
    }
    
    try:
        # Submit the request
        print("Submitting request to Replicate API...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        # Check response
        if response.status_code in [200, 201]:
            print("✅ Request successful! Prediction started.")
            prediction = response.json()
            prediction_id = prediction.get("id")
            
            if not prediction_id:
                print("No prediction ID returned")
                return None
            
            # Poll for the result
            print(f"Prediction ID: {prediction_id}")
            print("Waiting for image generation to complete...")
            get_url = f"{api_url}/{prediction_id}"
            
            # Wait for the prediction to complete with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                time.sleep(2)
                poll_response = requests.get(get_url, headers=headers)
                
                if poll_response.status_code != 200:
                    print(f"Error polling prediction: {poll_response.status_code}")
                    print(f"Response: {poll_response.text}")
                    break
                    
                prediction_status = poll_response.json()
                status = prediction_status.get("status")
                
                print(f"Status: {status}")
                
                if status == "succeeded":
                    print("✅ Image generation complete!")
                    # Get the image URL
                    output_url = prediction_status.get("output")
                    
                    if output_url:
                        # If output is a list, take the first item
                        if isinstance(output_url, list) and len(output_url) > 0:
                            output_url = output_url[0]
                            
                        print(f"Image URL: {output_url}")
                        
                        # Download the image
                        image_response = requests.get(output_url)
                        if image_response.status_code == 200:
                            return image_response.content
                    break
                elif status == "failed":
                    print(f"❌ Image generation failed: {prediction_status.get('error')}")
                    break
            
            # If we get here, the prediction timed out or failed
            print("Image generation did not complete in the expected time")
            return None
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None

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
        
        # Use RoBERTa model to detect emotions
        emotions = detect_emotions(text)
        print(f"Detected emotions: {emotions}")
        
        # Get the dominant emotions
        is_happy = emotions.get('joy', False)
        is_sad = emotions.get('sadness', False)
        is_angry = emotions.get('anger', False)
        is_fearful = emotions.get('fear', False)
        
        # Generate feedback based on emotions
        if is_happy:
            feedback = ("I love how you expressed happiness in your writing! To make your story even more engaging, try describing what made you happy and how that happiness felt in your body. Did you smile, laugh, or jump with excitement?", True)
            prompt = "a happy child playing in a sunny park, joyful colors, bright atmosphere, children's book illustration style"
        elif is_sad:
            feedback = ("I can feel the sadness in your writing. You've done a good job expressing this emotion. To make your writing even more powerful, try describing how sadness feels in your body. Did your shoulders slump? Did you feel heavy?", True)
            prompt = "a child looking out a rainy window, blue tones, gentle rain, melancholic mood, children's book illustration style"
        elif is_angry:
            feedback = ("I can sense the anger in your writing. You've done a good job expressing this emotion. To make your writing even more impactful, try describing how anger feels in your body. Did your face get hot? Did your heart beat faster?", True)
            prompt = "a child with furrowed brows and crossed arms, warm orange and red tones, expressive posture, children's book illustration style"
        elif is_fearful:
            feedback = ("I can feel the fear in your writing. You've done a good job expressing this emotion. To make your writing even more compelling, try describing how fear feels in your body. Did you tremble? Did you feel cold?", True)
            prompt = "a child hiding under blankets with a flashlight, dark blue shadows, mysterious atmosphere, children's book illustration style"
        else:
            feedback = ("You've written a nice piece! To make it more engaging, try adding details about how you felt. What emotions were you experiencing? How did those emotions feel in your body?", True)
            prompt = "a thoughtful child writing in a journal, calm scene, neutral colors, children's book illustration style"
        
        # Generate image with Replicate
        image_data = generate_image_with_replicate(prompt)
        
        if image_data:
            # Save the image from binary data
            image_filename = f"generated_image_{int(time.time())}.png"
            image_path = os.path.join(BASE_DIR, image_filename)
            
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            print(f"Saved generated image to {image_path}")
        else:
            # Fallback to placeholder if generation failed
            print("Image generation failed, using placeholder")
            image_filename = f"placeholder_image_{int(time.time())}.png"
            image_path = os.path.join(BASE_DIR, image_filename)
            
            # Create a simple image with color based on emotion
            if is_happy:
                color = (255, 255, 0)  # Yellow for happy
            elif is_sad:
                color = (0, 0, 255)    # Blue for sad
            elif is_angry:
                color = (255, 0, 0)    # Red for angry
            elif is_fearful:
                color = (128, 0, 128)  # Purple for fearful
            else:
                color = (0, 255, 0)    # Green for neutral
                
            img = Image.new('RGB', (512, 512), color=color)
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
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    # Run the app
    app.run(host="0.0.0.0", port=port, debug=debug) 