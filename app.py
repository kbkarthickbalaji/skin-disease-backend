from flask import Flask, render_template, request, jsonify
import onnxruntime as ort
import json
import numpy as np
import os
import cv2
import threading
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load ONNX model instead of H5
ort_session = ort.InferenceSession('models/skin_model.onnx')

_config_path = os.path.join('models', 'model_config.json')
if os.path.isfile(_config_path):
    with open(_config_path, encoding='utf-8') as f:
        model_config = json.load(f)
else:
    model_config = {'preprocess': 'scale', 'chart_labels': ['AK', 'BCC', 'DF', 'BKL', 'MEL', 'NV', 'VASC']}

INPUT_SIZE = model_config.get('input_size', 224)
CHART_LABELS = model_config.get('chart_labels', ['AK', 'BCC', 'DF', 'BKL', 'MEL', 'NV', 'VASC'])

# 2. Detailed Medical Data
recommendations = {
    'Actinic keratoses': {
        'remedy': 'Avoid sun exposure and use high-SPF sunscreen.',
        'doctor': 'Consult a dermatologist for possible cryotherapy or topical creams.'
    },
    'Basal cell carcinoma': {
        'remedy': 'Protect the area from further irritation or injury.',
        'doctor': 'Urgent: This is a form of skin cancer. Professional surgical removal is usually required.'
    },
    'Benign keratosis': {
        'remedy': 'No active treatment needed as these are non-cancerous.',
        'doctor': 'Consult if the lesion becomes itchy, inflamed, or bleeds.'
    },
    'Dermatofibroma': {
        'remedy': 'Usually harmless. Avoid picking at the bump.',
        'doctor': 'Can be removed for cosmetic reasons if it causes discomfort.'
    },
    'Melanocytic nevi': {
        'remedy': 'Monitor for any changes in size, shape, or color (ABCDE rule).',
        'doctor': 'Regular skin checks are recommended for individuals with many moles.'
    },
    'Melanoma': {
        'remedy': 'Immediate protection from UV radiation.',
        'doctor': 'CRITICAL: High risk of spreading. Immediate biopsy and specialist consultation required.'
    },
    'Vascular lesions': {
        'remedy': 'Gentle skincare; avoid harsh chemicals on the affected area.',
        'doctor': 'Laser therapy is often used if removal is desired.'
    }
}

# Class order must match training (see models/class_labels.json)
_labels_path = os.path.join('models', 'class_labels.json')
if os.path.isfile(_labels_path):
    with open(_labels_path, encoding='utf-8') as f:
        classes = json.load(f)
else:
    classes = list(recommendations.keys())

# --- NEW: Delayed Delete Function ---
def delayed_delete(file_path, delay):
    """Wait for delay (seconds) and delete the file in background."""
    time.sleep(delay)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🔒 Privacy Success: {file_path} deleted after delay.")
    except Exception as e:
        print(f"Error in delayed delete: {e}")

# Preprocessing must match training (no tf dependencies)
def process_image_opencv(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    if model_config.get('preprocess') == 'efficientnet':
        img_array = img.astype(np.float32)
    else:
        img_array = img.astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/metrics')
def metrics():
    # Performance page-la metrics show aaga intha route mukkiyam
    return render_template('performance.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save image temporarily
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    try:
        # 3. Preprocessing and Prediction
        processed_img = process_image_opencv(img_path)
        
        # Prepare normalization inputs for ONNX model
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
        variance = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
        sqrt_variance = np.sqrt(variance)
        
        inputs = {
            'input_layer:0': processed_img.astype(np.float32),
            'skin_classifier_1/normalization_1/Sub/y:0': mean,
            'skin_classifier_1/normalization_1/Sqrt/x:0': sqrt_variance
        }
        
        preds = ort_session.run(None, inputs)[0]
        pred_idx = np.argmax(preds)
        confidence = float(np.max(preds)) * 100
        
        pred_class = classes[pred_idx]
        info = recommendations.get(pred_class)

        result_data = {
            'class': pred_class,
            'confidence': f"{confidence:.2f}%",
            'remedy': info['remedy'],
            'doctor': info['doctor'],
            'image_url': img_path.replace('\\', '/'),
            'chart_data': preds.tolist()[0],
            'chart_labels': CHART_LABELS,
        }

        # --- Confidence Threshold Logic ---
        if confidence < 40.0:
            result_data.update({
                'class': 'Low Confidence Result',
                'remedy': 'The image is too blurry or unclear.',
                'doctor': 'Please retake in better light.'
            })

        # --- DELAYED PRIVACY DELETE (5 Minutes = 300 Seconds) ---
        # Image website-la theriyanum, aana folder-layum delete aaganum
        # 300 seconds background-la wait panni delete pannum
        threading.Thread(target=delayed_delete, args=(img_path, 300)).start()

        return jsonify(result_data)

    except Exception as e:
        # Error vanthaalum safety-kaaga udane delete panna try panrom
        if os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Use 5050 — port 5000 is often taken by other local dev servers on Windows
    port = int(os.environ.get('PORT', 5050))
    print(f'DermAI running at http://127.0.0.1:{port}')
    app.run(debug=True, host='127.0.0.1', port=port, use_reloader=False)