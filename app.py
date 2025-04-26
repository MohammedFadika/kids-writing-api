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
import re
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

# Enhanced emotion keyword dictionaries with weighted scoring
EMOTION_KEYWORDS = {
    "joy": {
        "high": ["thrilled", "ecstatic", "overjoyed", "elated", "delighted", "exciting", "wonderful", "amazing", "fantastic", "awesome", "incredible", "love"],
        "medium": ["happy", "glad", "cheerful", "pleased", "content", "fun", "enjoyed", "joy", "smile", "laughed", "great", "good", "positive", "lucky"],
        "low": ["nice", "fine", "okay", "alright", "satisfied", "playful", "friendly", "peaceful"]
    },
    "sadness": {
        "high": ["devastated", "heartbroken", "miserable", "depressed", "grief", "tragic", "awful", "terrible", "horrible", "crying", "sobbing", "despair"],
        "medium": ["sad", "unhappy", "disappointed", "upset", "hurt", "lonely", "missing", "sorry", "regret", "gloomy", "blue", "lost"],
        "low": ["down", "troubled", "bothered", "uncomfortable", "tired", "confused", "uncertain", "meh", "sigh"]
    },
    "anger": {
        "high": ["furious", "enraged", "outraged", "livid", "hate", "despise", "disgusted", "horrified", "violent", "exploded", "screaming", "yelling"],
        "medium": ["angry", "mad", "annoyed", "frustrated", "irritated", "bothered", "complained", "unfair", "broke", "ruined", "destroyed"],
        "low": ["bothered", "displeased", "grumpy", "bothered", "unimpressed", "impatient", "argued", "disagreed"]
    },
    "fear": {
        "high": ["terrified", "panicked", "horrified", "petrified", "nightmare", "dreadful", "scared", "paranoid", "traumatic", "paralyzed", "frozen"],
        "medium": ["afraid", "frightened", "anxious", "worried", "nervous", "fearful", "uneasy", "timid", "shaking", "trembling", "alarmed"],
        "low": ["concerned", "unsure", "doubtful", "shy", "hesitant", "careful", "uncomfortable", "suspicious", "strange", "dark"]
    }
}

# Negation words that can reverse emotion meaning
NEGATION_WORDS = ["not", "no", "never", "don't", "can't", "couldn't", "wouldn't", "didn't", "isn't", 
                  "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "shouldn't", "won't"]

# Intensifiers that can strengthen emotions
INTENSIFIERS = ["very", "really", "extremely", "incredibly", "absolutely", "completely", "totally", 
                "deeply", "terribly", "awfully", "super", "so", "too", "quite", "especially"]

# Amplifying punctuation patterns
EMPHASIS_PATTERNS = [r'!+', r'\?!+', r'\?{2,}', r'\.{3,}', r'[A-Z]{2,}']

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
use_keyword_fallback = False
try:
    print("Loading RoBERTa model for emotion detection from Hugging Face Hub...")
    # Load with memory-efficient settings for deployment environments
    config = {'torchscript': True, 'low_cpu_mem_usage': True}
    
    # Attempt to load with memory optimizations
    emotion_model = RobertaForSequenceClassification.from_pretrained(
        EMOTION_MODEL_ID, 
        torchscript=True,
        low_cpu_mem_usage=True
    )
    emotion_tokenizer = RobertaTokenizer.from_pretrained(EMOTION_MODEL_ID)
    print("RoBERTa model loaded successfully!")
    use_keyword_fallback = False
    
except Exception as e:
    print(f"Error loading RoBERTa model from Hugging Face Hub: {e}")
    print("Falling back to enhanced linguistic analysis for emotion detection")
    use_keyword_fallback = True

def detect_emotions(text):
    """Detect emotions from text"""
    if not use_keyword_fallback:
        try:
            # Try using the fine-tuned model
            encoding = emotion_tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                output = emotion_model(**encoding)
            predictions = torch.sigmoid(output.logits)
            predictions = predictions > 0.5  # Threshold to get 0 or 1 for each emotion
            emotion_columns = ["anger", "fear", "sadness", "joy"]  # The emotions our model was trained on
            predicted_emotions = dict(zip(emotion_columns, predictions.squeeze().tolist()))
            return predicted_emotions
        except Exception as e:
            print(f"Error using RoBERTa model for prediction: {e}")
            print("Falling back to enhanced linguistic analysis for this request")
    
    # Enhanced fallback detection using linguistic analysis
    return advanced_emotion_detection(text)

def advanced_emotion_detection(text):
    """
    Advanced emotion detection using linguistic features:
    - Expanded emotional vocabulary with intensity levels
    - Negation handling
    - Contextual analysis
    - Emphasis detection (punctuation, capitalization)
    """
    # Normalize text
    text = text.lower()
    
    # Detect emphasis patterns (exclamation marks, question marks, etc.)
    emphasis_multiplier = 1.0
    for pattern in EMPHASIS_PATTERNS:
        if re.search(pattern, text):
            emphasis_multiplier = 1.2
            break
    
    # Initialize emotion scores
    emotion_scores = {
        "anger": 0.0,
        "fear": 0.0,
        "sadness": 0.0,
        "joy": 0.0
    }
    
    # Split into sentences to better handle negation and context
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        # Split into words
        words = re.findall(r'\b\w+\b', sentence)
        
        # Check for negation words
        has_negation = any(neg in words for neg in NEGATION_WORDS)
        
        # Check for intensifiers
        intensifier_present = any(intensifier in words for intensifier in INTENSIFIERS)
        intensity_multiplier = 1.3 if intensifier_present else 1.0
        
        # Analyze each emotion
        for emotion, levels in EMOTION_KEYWORDS.items():
            local_score = 0.0
            
            # Check high intensity words
            for word in levels["high"]:
                if word in sentence:
                    local_score += 3.0
            
            # Check medium intensity words
            for word in levels["medium"]:
                if word in sentence:
                    local_score += 2.0
            
            # Check low intensity words
            for word in levels["low"]:
                if word in sentence:
                    local_score += 1.0
            
            # Apply modifiers (negation, intensity)
            if has_negation:
                # If negation is present, reverse the emotion (e.g., "not happy" -> sadness)
                if emotion == "joy":
                    emotion_scores["sadness"] += local_score * 0.7  # Add to opposite emotion
                    local_score *= 0.2  # Reduce original emotion
                elif emotion == "sadness":
                    emotion_scores["joy"] += local_score * 0.5  # Add to opposite emotion
                    local_score *= 0.2  # Reduce original emotion
                else:
                    local_score *= 0.3  # Just reduce for anger/fear
            
            # Apply intensity multiplier from any intensifiers
            local_score *= intensity_multiplier
            
            # Add the score to the emotion
            emotion_scores[emotion] += local_score
    
    # Apply emphasis multiplier (for exclamations, etc)
    for emotion in emotion_scores:
        emotion_scores[emotion] *= emphasis_multiplier
    
    # Normalize scores to a 0-1 range if any emotions were detected
    max_score = max(emotion_scores.values())
    if max_score > 0:
        threshold = max_score * 0.3  # Dynamic threshold as 30% of max score
        
        # Convert to binary output based on threshold
        binary_emotions = {
            emotion: score >= threshold
            for emotion, score in emotion_scores.items()
        }
        
        # If no emotions reached threshold, ensure at least the max emotion is True
        if not any(binary_emotions.values()) and max_score > 0:
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            binary_emotions[max_emotion] = True
        
        return binary_emotions
    else:
        # Default to all false if no emotions were detected
        return {emotion: False for emotion in emotion_scores}

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
        
        # Detect emotions
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