import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import base64
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Import segmentation models
try:
    import segmentation_models_pytorch as smp
    logger.info("Segmentation models loaded successfully")
except ImportError as e:
    logger.error(f"Error importing segmentation models: {e}")

# Global models
classification_model = None
segmentation_model = None
CLASS_NAMES = ['Splicing', 'Copy-Move', 'Inpainting', 'Face Manipulation']

def load_models():
    """Load pre-trained models"""
    global classification_model, segmentation_model
    
    try:
        logger.info("Loading ResNet50 classification model...")
        # For classification - using pretrained ResNet50
        classification_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        # Modify final layer for 4 classes
        num_features = classification_model.fc.in_features
        classification_model.fc = nn.Linear(num_features, 4)
        classification_model = classification_model.to(device)
        classification_model.eval()
        logger.info("Classification model loaded")
        
        logger.info("Loading DeepLabV3+ segmentation model...")
        # For segmentation - using DeepLabV3+ with ResNet50 backbone
        segmentation_model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
        segmentation_model = segmentation_model.to(device)
        segmentation_model.eval()
        logger.info("Segmentation model loaded")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def preprocess_image(image, size=256):
    """Preprocess image for model input"""
    if isinstance(image, str):
        # Base64 string
        image_data = base64.b64decode(image)
        image = Image.open(io.BytesIO(image_data))
    
    image = image.convert('RGB')
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    
    # Normalize
    img_array = np.array(image) / 255.0
    img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    
    return img_tensor.to(device)

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    # Normalize to 0-255 range
    if image_array.max() <= 1:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)
    
    # Convert to PIL Image
    if len(image_array.shape) == 2:
        pil_image = Image.fromarray(image_array, mode='L')
    else:
        pil_image = Image.fromarray(image_array, mode='RGB')
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return img_base64

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'models_loaded': classification_model is not None and segmentation_model is not None
    })

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Classify image manipulation type"""
    try:
        if 'image' not in request.files and 'image_base64' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get image
        if 'image' in request.files:
            image = Image.open(request.files['image'].stream)
        else:
            image = request.json['image_base64']
        
        # Preprocess
        img_tensor = preprocess_image(image)
        
        # Classify
        with torch.no_grad():
            output = classification_model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return jsonify({
            'class': CLASS_NAMES[predicted_class],
            'class_id': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                CLASS_NAMES[i]: float(probabilities[0, i].item())
                for i in range(len(CLASS_NAMES))
            }
        })
        
    except Exception as e:
        logger.error(f"Error in classification: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/segment', methods=['POST'])
def segment_image():
    """Generate segmentation mask for manipulated regions"""
    try:
        if 'image' not in request.files and 'image_base64' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get image
        if 'image' in request.files:
            image = Image.open(request.files['image'].stream)
        else:
            image = request.json['image_base64']
        
        original_image = image if isinstance(image, Image.Image) else Image.open(io.BytesIO(base64.b64decode(image)))
        original_size = original_image.size
        
        # Preprocess
        img_tensor = preprocess_image(image)
        
        # Segment
        with torch.no_grad():
            output = segmentation_model(img_tensor)
            mask = torch.sigmoid(output[0, 0].cpu())
            mask_numpy = mask.numpy()
            
            # Threshold
            binary_mask = (mask_numpy > 0.5).astype(np.uint8) * 255
        
        # Resize mask to original size
        mask_pil = Image.fromarray(binary_mask)
        mask_pil = mask_pil.resize(original_size, Image.Resampling.BILINEAR)
        
        # Convert to base64
        mask_base64 = image_to_base64(np.array(mask_pil))
        
        return jsonify({
            'mask': mask_base64,
            'confidence': float(mask_numpy.mean()),
            'size': list(original_size)
        })
        
    except Exception as e:
        logger.error(f"Error in segmentation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Complete analysis: classification + segmentation"""
    try:
        if 'image' not in request.files and 'image_base64' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get image
        if 'image' in request.files:
            image = Image.open(request.files['image'].stream)
        else:
            image = request.json['image_base64']
        
        # Preprocess
        img_tensor = preprocess_image(image)
        
        # Classify
        with torch.no_grad():
            # Classification
            class_output = classification_model(img_tensor)
            class_probs = torch.softmax(class_output, dim=1)
            predicted_class = class_probs.argmax(dim=1).item()
            class_confidence = class_probs[0, predicted_class].item()
            
            # Segmentation
            seg_output = segmentation_model(img_tensor)
            mask = torch.sigmoid(seg_output[0, 0].cpu())
            mask_numpy = mask.numpy()
            binary_mask = (mask_numpy > 0.5).astype(np.uint8) * 255
        
        # Convert mask to base64
        mask_base64 = image_to_base64(binary_mask)
        
        return jsonify({
            'classification': {
                'class': CLASS_NAMES[predicted_class],
                'class_id': predicted_class,
                'confidence': float(class_confidence),
                'probabilities': {
                    CLASS_NAMES[i]: float(class_probs[0, i].item())
                    for i in range(len(CLASS_NAMES))
                }
            },
            'segmentation': {
                'mask': mask_base64,
                'confidence': float(mask_numpy.mean())
            }
        })
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """API Documentation"""
    return jsonify({
        'name': 'Image Manipulation Detection API',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/classify': 'Classify image manipulation type',
            'POST /api/segment': 'Generate segmentation mask',
            'POST /api/analyze': 'Complete analysis (classification + segmentation)',
            'GET /health': 'Health check'
        },
        'supported_classes': CLASS_NAMES
    })

if __name__ == '__main__':
    try:
        logger.info("Starting Image Manipulation Detection API...")
        load_models()
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
