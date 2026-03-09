import cv2
import json
import numpy as np
import os
from flask import Flask, render_template, Response, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load model and labels
MODEL_PATH = "models/sign_model.h5"
SIMPLE_MODEL_PATH = "models/simple_sign_model.h5"
MAP_PATH = "models/class_indices.json"
SIMPLE_MAP_PATH = "models/simple_class_indices.json"

# Global variables for model and video capture
model = None
idx2label = {}
cap = None

# Define preprocessing function for MobileNetV2
def preprocess_input(x):
    """Manual preprocessing for MobileNetV2: normalize to [-1, 1]"""
    return (x / 127.5) - 1.0

# Load TensorFlow and model with compatibility handling
try:
    import tensorflow as tf
    
    # Try to load the original model first
    print("Attempting to load original model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        # Load original class indices
        with open(MAP_PATH, "r") as f:
            idx2label = json.load(f)
        print("Original model loaded successfully")
    except Exception as e:
        print(f"Failed to load original model: {e}")
        print("Attempting to load simple fallback model...")
        try:
            # Load the simple model as fallback
            model = tf.keras.models.load_model(SIMPLE_MODEL_PATH)
            # Load simple class indices
            with open(SIMPLE_MAP_PATH, "r") as f:
                idx2label = json.load(f)
            print("Simple fallback model loaded successfully")
        except Exception as e2:
            print(f"Failed to load simple model: {e2}")
            raise e  # Re-raise the original exception
    
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Labels: {idx2label}")
    
except Exception as e:
    print(f"Warning: Could not load model or labels: {e}")
    import traceback
    traceback.print_exc()
    print("The application will run but prediction features will be disabled.")

def generate_frames():
    """Generate video frames with sign language prediction overlay"""
    global cap, model, idx2label
    
    # Check if model loaded successfully
    if model is None or not idx2label:
        # Return frames without prediction if model not loaded
        if cap is None:
            cap = cv2.VideoCapture(0)
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                # Add text indicating model not loaded
                cv2.putText(frame, "Model not loaded - Compatibility issue", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "See console for details", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                
                # Yield frame in multipart format for streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    # Initialize camera
    if cap is None:
        cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process frame for prediction
            processed_frame = process_frame(frame)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            
            # Yield frame in multipart format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_frame(frame):
    """Process a frame to add sign language prediction overlay"""
    global model, idx2label
    
    # Check if model and labels loaded successfully
    if model is None or not idx2label:
        return frame
    
    # Crop square ROI from center
    h, w = frame.shape[:2]
    size = min(h, w) // 2
    cx, cy = w // 2, h // 2
    x1, y1 = cx - size//2, cy - size//2
    x2, y2 = x1+size, y1+size
    roi = frame[y1:y2, x1:x2]
    
    # Preprocess for model
    img = cv2.resize(roi, (124,124))
    x = np.expand_dims(img, 0).astype("float32")
    
    # Preprocess input for MobileNetV2
    x = preprocess_input(x)
    
    # Make prediction
    try:
        preds = model.predict(x, verbose=0)
        idx = int(np.argmax(preds[0]))
        label = idx2label.get(str(idx), "Unknown")
        prob = preds[0][idx]
    except Exception as e:
        print(f"Prediction error: {e}")
        label = "Error"
        prob = 0.0
    
    # Draw rectangle around ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display prediction and probability
    display_text = f"{label} ({prob:.2f})"
    cv2.putText(frame, display_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Change background color based on label
    color = (255, 255, 255)  # White default
    if label == "Yes":
        color = (0, 255, 0)   # Green
    elif label == "No":
        color = (0, 0, 255)   # Red
    elif label == "Hello":
        color = (255, 0, 0)   # Blue
    elif label == "Thank you":
        color = (255, 255, 0) # Cyan
    elif label == "Please":
        color = (255, 0, 255) # Magenta
    elif label == "Sorry":
        color = (0, 255, 255) # Yellow
    
    # Apply colored overlay
    overlay = frame.copy()
    overlay[:, :] = color
    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    
    return frame

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    """API endpoint for getting current prediction"""
    global cap, model, idx2label
    
    # Check if model and labels loaded successfully
    if model is None or not idx2label:
        return jsonify({"error": "Model or labels not loaded - Compatibility issue"})
    
    if cap is None:
        cap = cv2.VideoCapture(0)
    
    success, frame = cap.read()
    if not success:
        return jsonify({"error": "Failed to capture frame"})
    
    # Process frame for prediction
    h, w = frame.shape[:2]
    size = min(h, w) // 2
    cx, cy = w // 2, h // 2
    x1, y1 = cx - size//2, cy - size//2
    x2, y2 = x1+size, y1+size
    roi = frame[y1:y2, x1:x2]
    
    # Preprocess for model
    img = cv2.resize(roi, (124,124))
    x = np.expand_dims(img, 0).astype("float32")
    
    # Preprocess input for MobileNetV2
    x = preprocess_input(x)
    
    # Make prediction
    try:
        preds = model.predict(x, verbose=0)
        idx = int(np.argmax(preds[0]))
        label = idx2label.get(str(idx), "Unknown")
        prob = preds[0][idx]
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"})
    
    return jsonify({
        "label": label,
        "probability": float(prob),
        "class_index": idx
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)