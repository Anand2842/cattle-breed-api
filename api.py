"""
Cattle Breed Classifier - Production API
50 Indian Cattle Breeds Classification
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import time

app = Flask(__name__)
CORS(app)

MODEL_PATH = "cattle_50breeds_model.pth"
# Use CPU on Render (no GPU on free tier)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    breeds = checkpoint['breeds']
    num_classes = checkpoint['num_classes']
    
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
    return model, breeds

print("Loading model...")
MODEL, BREEDS = load_model()
print(f"✅ Model loaded! {len(BREEDS)} breeds on {DEVICE}")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    img_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        start = time.time()
        outputs = MODEL(img_tensor)
        inference_time = (time.time() - start) * 1000
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        top5_probs, top5_indices = torch.topk(probs, 5)
    
    return {
        'breed': BREEDS[pred.item()],
        'confidence': round(conf.item() * 100, 2),
        'top5': [
            {'breed': BREEDS[idx], 'confidence': round(prob * 100, 2)}
            for idx, prob in zip(top5_indices[0].tolist(), top5_probs[0].tolist())
        ],
        'inference_time_ms': round(inference_time, 2)
    }

@app.route('/')
def home():
    return jsonify({
        'name': 'Indian Cattle Breed Classifier API',
        'version': '1.0.0',
        'breeds': len(BREEDS),
        'endpoints': {
            'health': 'GET /api/health',
            'breeds': 'GET /api/breeds',
            'predict_file': 'POST /api/predict',
            'predict_base64': 'POST /api/predict/base64'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'cattle_50breeds', 'breeds_count': len(BREEDS), 'device': str(DEVICE)})

@app.route('/api/breeds', methods=['GET'])
def list_breeds():
    return jsonify({'breeds': BREEDS, 'count': len(BREEDS)})

@app.route('/api/predict', methods=['POST'])
def predict_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        image = Image.open(file.stream).convert('RGB')
        result = predict_image(image)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/base64', methods=['POST'])
def predict_base64():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    try:
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        result = predict_image(image)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
