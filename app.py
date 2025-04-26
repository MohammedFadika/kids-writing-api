import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import json
import time
from PIL import Image
import traceback
import requests
import random
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

# Story elements to extract for personalized feedback
STORY_ELEMENTS = {
    "characters": [
        r'\b(?:my|with|and|to|from)\s+([a-z]+\s+)?friend(?:s)?\b',
        r'\b(?:my|with)\s+(?:mom|mother|dad|father|brother|sister|cousin|aunt|uncle|grandma|grandpa|grandmother|grandfather|family|teacher|classmate|pet|dog|cat)\b',
        r'\bwe\b'
    ],
    "activities": [
        r'\b(?:play(?:ing|ed)?|danc(?:ing|ed)?|sing(?:ing)?|read(?:ing)?|watch(?:ing|ed)?|saw|see(?:ing)?|went|go(?:ing)?|visit(?:ing|ed)?)\b',
        r'\b(?:the\s+)?(?:park|school|beach|zoo|museum|game|party|birthday|holiday|vacation|trip|movie|show|concert)\b'
    ],
    "objects": [
        r'\b(?:my|the|a|an)\s+(?:toy|book|game|ball|bike|computer|phone|tablet|present|gift)\b'
    ],
    "feelings": EMOTION_KEYWORDS
}

# Feedback templates for different emotion combinations
FEEDBACK_TEMPLATES = {
    # Single emotions
    "joy": [
        "I love how you expressed happiness in your story! To make your writing even more engaging, try describing what made you happy and how that happiness felt in your body. Did you smile, laugh, or jump with excitement?",
        "Your joy really shines through in this piece! To make it even stronger, consider adding more sensory details - what did you see, hear, or feel during this happy moment?",
        "What a wonderful expression of happiness! You could make your story even more vivid by describing the physical sensations of joy - did your heart beat faster? Did you have a warm feeling inside?"
    ],
    "sadness": [
        "I can feel the sadness in your writing. You've done a good job expressing this emotion. To make your writing even more powerful, try describing how sadness feels in your body. Did your shoulders slump? Did you feel heavy?",
        "You've captured a sense of sadness well. To deepen the emotional impact, you might add more details about how this feeling affected you physically - maybe tears, a lump in your throat, or feeling tired?",
        "The melancholy comes through in your story. To connect even more with your reader, try adding how your surroundings looked or sounded different when you were feeling sad."
    ],
    "anger": [
        "I can sense the anger in your writing. You've done a good job expressing this emotion. To make your writing even more impactful, try describing how anger feels in your body. Did your face get hot? Did your heart beat faster?",
        "Your frustration comes through clearly! To make this even more powerful, you could describe the physical sensations of anger - clenched fists, gritted teeth, or feeling like you might explode.",
        "The anger in your story is well-expressed. To make it even more vivid, try adding details about your voice - did it get louder? Did your words come out faster when you were angry?"
    ],
    "fear": [
        "I can feel the fear in your writing. You've done a good job expressing this emotion. To make your writing even more compelling, try describing how fear feels in your body. Did you tremble? Did you feel cold?",
        "You've captured a sense of fear well in your story. To make it even more gripping, consider adding details about your physical reactions - maybe a racing heart, goosebumps, or wanting to hide?",
        "The feeling of being scared comes through in your writing. To make it even more powerful, try describing how fear affected your breathing or how sounds seemed louder when you were afraid."
    ],
    "neutral": [
        "You've written a nice piece! To make it more engaging, try adding details about how you felt. What emotions were you experiencing? How did those emotions feel in your body?",
        "Your writing has a clear narrative. To make it even more compelling, consider adding more emotional color - how did these events make you feel inside?",
        "You've told an interesting story! To make it even better, try adding some emotional details - how did you feel during these moments, and how did those feelings affect you?"
    ],
    
    # Mixed emotions (joy + another emotion)
    "joy+sadness": [
        "I love how you've expressed both happiness and sadness in your story! This mix of emotions makes your writing very realistic. You could make it even more powerful by describing how these contrasting emotions felt in your body at different moments.",
        "Your story beautifully captures the complexity of feeling both joy and sadness. To make it even more nuanced, try describing how quickly your emotions changed from one to the other.",
        "The blend of happiness and sadness in your writing creates emotional depth. To enhance this further, you might describe how these mixed feelings affected your energy levels or interactions with others."
    ],
    "joy+anger": [
        "Your story shows an interesting mix of joy and frustration! This emotional complexity makes your writing feel authentic. You could make it even more powerful by describing how these contrasting feelings existed together.",
        "I'm impressed by how you've captured both happiness and anger in your writing. To make this even more compelling, try showing how one emotion transformed into the other.",
        "The combination of joy and anger in your story creates an engaging emotional journey. To make it even stronger, consider describing how these mixed feelings influenced your decisions or actions."
    ],
    "joy+fear": [
        "Your writing beautifully balances excitement and nervousness! This emotional mix makes your story very relatable. You could make it even more vivid by describing how these different feelings affected your body simultaneously.",
        "I like how your story captures both happiness and anxiety. To make it even more powerful, try describing moments where these emotions competed with each other.",
        "The blend of joy and fear in your writing creates wonderful tension. To enhance this further, consider describing how these conflicting emotions influenced your thoughts throughout the experience."
    ],
    
    # Mixed emotions (sadness + another emotion)
    "sadness+anger": [
        "Your writing powerfully expresses both sadness and anger. This emotional complexity makes your story very impactful. You could make it even stronger by describing how these emotions might have fueled each other.",
        "I'm impressed by how you've captured feelings of both sadness and frustration. To make this even more compelling, try describing how these emotions felt physically different in your body.",
        "The combination of sadness and anger in your story creates a strong emotional impact. To enhance this further, consider showing how these feelings influenced how you saw the world around you."
    ],
    "sadness+fear": [
        "Your story effectively communicates both sadness and fear. This emotional depth makes your writing very moving. You could make it even more powerful by describing how these feelings reinforced each other.",
        "I appreciate how your writing captures both sadness and anxiety. To make this even more compelling, try describing how these emotions affected your ability to think clearly or make decisions.",
        "The blend of sadness and fear in your writing creates a vulnerable and authentic narrative. To enhance this further, consider describing moments where one emotion provided a brief respite from the other."
    ],
    
    # Mixed emotions (anger + fear)
    "anger+fear": [
        "Your writing powerfully expresses both anger and fear. This combination creates a very intense emotional landscape. You could make it even more impactful by describing how these strong emotions competed for your attention.",
        "I'm impressed by how you've captured the volatile mix of frustration and anxiety. To make this even more compelling, try describing how these emotions might have affected your breathing or heart rate.",
        "The combination of anger and fear in your story creates powerful tension. To enhance this further, consider showing which emotion felt more dominant at different points in your experience."
    ],
    
    # Multiple emotions (3+)
    "multiple_emotions": [
        "Wow! Your story contains a rich tapestry of emotions. This complexity makes your writing very sophisticated. You could make it even more powerful by showing how these different feelings ebbed and flowed throughout your experience.",
        "I'm impressed by the emotional depth in your writing. To make this even more compelling, try describing how managing these various feelings affected your energy or concentration.",
        "Your ability to express multiple emotions creates a very authentic and relatable story. To enhance this further, consider describing which emotion felt strongest at different key moments."
    ]
}

# Prompt templates for image generation based on emotions
IMAGE_PROMPTS = {
    "joy": [
        "a happy child playing in a sunny park, joyful colors, bright atmosphere, children's book illustration style",
        "a smiling child with friends celebrating, confetti, bright sunshine, children's book illustration style",
        "a child jumping with excitement, arms raised, golden sunlight, children's book illustration style"
    ],
    "sadness": [
        "a child looking out a rainy window, blue tones, gentle rain, melancholic mood, children's book illustration style",
        "a thoughtful child sitting alone, autumn leaves falling, soft blue colors, children's book illustration style",
        "a child with a wistful expression, cloudy sky, muted colors, children's book illustration style"
    ],
    "anger": [
        "a child with furrowed brows and crossed arms, warm orange and red tones, expressive posture, children's book illustration style",
        "a child with a frustrated expression, scattered toys, bold colors, children's book illustration style",
        "a determined child resolving a conflict, dynamic composition, vibrant red accents, children's book illustration style"
    ],
    "fear": [
        "a child hiding under blankets with a flashlight, dark blue shadows, mysterious atmosphere, children's book illustration style",
        "a child looking cautiously around a corner, soft purple shadows, gentle lighting, children's book illustration style",
        "a child bravely facing shadows, twilight colors, mystical atmosphere, children's book illustration style"
    ],
    "neutral": [
        "a thoughtful child writing in a journal, calm scene, neutral colors, children's book illustration style",
        "a child exploring nature with curiosity, balanced composition, natural colors, children's book illustration style",
        "a contemplative child looking at the sky, serene scene, gentle colors, children's book illustration style"
    ],
    "joy+sadness": [
        "a child smiling despite rain, rainbow appearing, mixture of bright and blue tones, children's book illustration style",
        "a child with a bittersweet expression holding a memory item, sunshine through clouds, children's book illustration style",
        "a child saying goodbye but looking forward, mixed warm and cool colors, children's book illustration style"
    ],
    "joy+anger": [
        "a child with mixed expressions winning a difficult game, dynamic composition, red and yellow colors, children's book illustration style",
        "a child overcoming frustration to achieve something, contrasting colors, energetic scene, children's book illustration style",
        "a child letting go of anger to enjoy a moment, transformative colors from red to yellow, children's book illustration style"
    ],
    "joy+fear": [
        "a child on an exciting adventure looking both thrilled and nervous, contrasting light and shadow, children's book illustration style",
        "a child on a rollercoaster with an expression mixing excitement and fear, dynamic scene, children's book illustration style",
        "a child discovering something new with wonder and caution, magical lighting, children's book illustration style"
    ],
    "sadness+anger": [
        "a child with tears but determined expression, stormy scene, powerful colors, children's book illustration style",
        "a child feeling hurt but standing strong, rain and fire elements, emotional scene, children's book illustration style",
        "a child processing difficult emotions, moody landscape, deep colors, children's book illustration style"
    ],
    "sadness+fear": [
        "a child seeking comfort in a dark room, gentle moonlight, protective blanket, children's book illustration style",
        "a child facing uncertainty with tears but courage, misty scene, soft protective light, children's book illustration style",
        "a child with a worried expression during a storm, finding a safe place, children's book illustration style"
    ],
    "anger+fear": [
        "a child showing defiance despite being scared, dramatic lighting, powerful stance, children's book illustration style",
        "a child confronting shadows with a mixture of fear and determination, dynamic composition, children's book illustration style",
        "a child with complex emotions facing a challenge, intense colors, atmospheric scene, children's book illustration style"
    ],
    "multiple_emotions": [
        "a child experiencing a life-changing moment with complex emotions, rich color palette, expressive scene, children's book illustration style",
        "a child on an emotional journey through different landscapes, magical realism, diverse colors, children's book illustration style",
        "a child with a face showing layered emotions, surrounded by symbolic elements, artistic style, children's book illustration style"
    ]
}

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
        "feedback": (FEEDBACK_TEMPLATES["joy"][0], True),
        "imagePath": "happy_image.txt"
    },
    "sad": {
        "emotions": {
            "joy": False,
            "sadness": True,
            "anger": False,
            "fear": False
        },
        "feedback": (FEEDBACK_TEMPLATES["sadness"][0], True),
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

def extract_story_elements(text):
    """Extract key elements from the story for personalized feedback"""
    elements = {
        "characters": set(),
        "activities": set(),
        "objects": set(),
        "emotion_words": set()
    }
    
    # Normalize text
    text_lower = text.lower()
    
    # Extract characters
    for pattern in STORY_ELEMENTS["characters"]:
        matches = re.findall(pattern, text_lower)
        if matches:
            elements["characters"].update(matches)
    
    # Extract activities
    for pattern in STORY_ELEMENTS["activities"]:
        matches = re.findall(pattern, text_lower)
        if matches:
            elements["activities"].update(matches)
    
    # Extract objects
    for pattern in STORY_ELEMENTS["objects"]:
        matches = re.findall(pattern, text_lower)
        if matches:
            elements["objects"].update(matches)
    
    # Extract emotion words
    for emotion, levels in STORY_ELEMENTS["feelings"].items():
        for intensity, words in levels.items():
            for word in words:
                if word in text_lower:
                    elements["emotion_words"].add(word)
    
    return elements

def get_emotion_combination_key(emotions):
    """Determine the appropriate emotion combination key for feedback and image prompts"""
    active_emotions = [emotion for emotion, is_active in emotions.items() if is_active]
    
    if len(active_emotions) == 0:
        return "neutral"
    elif len(active_emotions) == 1:
        return active_emotions[0]
    elif len(active_emotions) == 2:
        # Sort to ensure consistent ordering (e.g., joy+sadness instead of sadness+joy)
        sorted_emotions = sorted(active_emotions)
        return f"{sorted_emotions[0]}+{sorted_emotions[1]}"
    else:
        return "multiple_emotions"

def generate_personalized_feedback(text, emotions, story_elements):
    """Generate personalized feedback based on emotions and story elements"""
    # Determine the emotion combination key
    emotion_key = get_emotion_combination_key(emotions)
    
    # Get appropriate feedback templates
    templates = FEEDBACK_TEMPLATES.get(emotion_key, FEEDBACK_TEMPLATES["neutral"])
    
    # Select a random template to avoid repetition
    base_feedback = random.choice(templates)
    
    # Add personalization if we have story elements
    if story_elements["characters"] or story_elements["activities"]:
        personalized_prefix = ""
        
        # Mention characters if available
        if story_elements["characters"]:
            # Clean up character mentions
            characters = [c.strip() for c in story_elements["characters"] if len(c.strip()) > 0]
            if characters:
                character_text = random.choice(characters)
                personalized_prefix = f"I noticed you wrote about {character_text}. "
        
        # Mention activities if available and not already mentioned characters
        elif story_elements["activities"]:
            # Clean up activity mentions
            activities = [a.strip() for a in story_elements["activities"] if len(a.strip()) > 0]
            if activities:
                activity_text = random.choice(activities)
                personalized_prefix = f"I see you described {activity_text}. "
        
        # Add the personalized prefix if we created one
        if personalized_prefix:
            base_feedback = personalized_prefix + base_feedback
    
    return (base_feedback, True)

def generate_image_prompt(emotions, story_elements):
    """Generate an appropriate image prompt based on emotions and story elements"""
    # Determine the emotion combination key
    emotion_key = get_emotion_combination_key(emotions)
    
    # Get appropriate image prompts
    prompts = IMAGE_PROMPTS.get(emotion_key, IMAGE_PROMPTS["neutral"])
    
    # Select a random prompt to avoid repetition
    base_prompt = random.choice(prompts)
    
    # Enhance with story elements if available (future enhancement)
    # This could incorporate specific characters or settings from the story
    
    return base_prompt

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
        
        # Extract story elements for personalization
        story_elements = extract_story_elements(text)
        print(f"Extracted story elements: {story_elements}")
        
        # Generate personalized feedback
        feedback = generate_personalized_feedback(text, emotions, story_elements)
        
        # Generate appropriate image prompt
        prompt = generate_image_prompt(emotions, story_elements)
        
        # Get dominant emotions for placeholder image if needed
        is_happy = emotions.get('joy', False)
        is_sad = emotions.get('sadness', False)
        is_angry = emotions.get('anger', False)
        is_fearful = emotions.get('fear', False)
        
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