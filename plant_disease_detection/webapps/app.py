from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os
from scipy.stats import entropy

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Load temperature scaler if available (from the improved training code)
temperature_scaler = None
if os.path.exists('temperature_scaler.pkl'):
    with open('temperature_scaler.pkl', 'rb') as f:
        temperature_scaler = pickle.load(f)
    print("Temperature scaler loaded for calibrated predictions")
else:
    print("No temperature scaler found - using raw model predictions")

# Class names from your dataset
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy", 
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_YellowLeaf_Curl_Virus",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two_spotted_spider_mite"
]

def apply_temperature_scaling(logits, temperature=1.0):
    """Apply temperature scaling to logits"""
    scaled_logits = logits / temperature
    return tf.nn.softmax(scaled_logits).numpy()

def get_calibrated_predictions(img_array):
    """Get calibrated predictions with uncertainty measures"""
    # Get raw predictions
    raw_predictions = model.predict(img_array, verbose=0)[0]
    
    # Apply temperature scaling if available
    if temperature_scaler:
        # Get logits (we'll approximate from softmax output)
        # Note: This is approximate - ideally save logits from training
        logits = np.log(np.clip(raw_predictions, 1e-15, 1-1e-15))
        calibrated_predictions = apply_temperature_scaling(
            logits.reshape(1, -1), 
            temperature_scaler.temperature
        )[0]
    else:
        calibrated_predictions = raw_predictions
    
    return raw_predictions, calibrated_predictions

def calculate_uncertainty_metrics(predictions):
    """Calculate various uncertainty metrics"""
    # Entropy (higher = more uncertain)
    pred_entropy = entropy(predictions)
    
    # Max probability (confidence)
    max_prob = np.max(predictions)
    
    # Standard deviation
    std_dev = np.std(predictions)
    
    # Top-2 difference (difference between top 2 predictions)
    sorted_preds = np.sort(predictions)[::-1]
    top2_diff = sorted_preds[0] - sorted_preds[1]
    
    # Effective number of classes (lower = more concentrated)
    effective_classes = np.exp(pred_entropy)
    
    return {
        'entropy': pred_entropy,
        'max_prob': max_prob,
        'std_dev': std_dev,
        'top2_diff': top2_diff,
        'effective_classes': effective_classes
    }

def is_likely_plant_leaf(img_array):
    """Simple heuristic to check if image might be a plant leaf"""
    # Check if image has some green content
    img_rgb = img_array[0]  # Remove batch dimension
    
    # Calculate green channel dominance
    green_dominance = np.mean(img_rgb[:,:,1]) / (np.mean(img_rgb) + 1e-8)
    
    # Check color variance (leaves usually have some color variation)
    color_variance = np.var(img_rgb)
    
    # Check if image is not mostly uniform (like solid color backgrounds)
    brightness_variance = np.var(np.mean(img_rgb, axis=2))
    
    return {
        'green_dominance': green_dominance,
        'color_variance': color_variance,
        'brightness_variance': brightness_variance,
        'likely_leaf': green_dominance > 0.8 and color_variance > 0.01 and brightness_variance > 0.001
    }

def make_robust_prediction(img_array):
    """Make a robust prediction with uncertainty quantification"""
    # Get predictions
    raw_preds, calibrated_preds = get_calibrated_predictions(img_array)
    
    # Calculate uncertainty metrics
    uncertainty = calculate_uncertainty_metrics(calibrated_preds)
    
    # Check if image looks like a plant leaf
    leaf_check = is_likely_plant_leaf(img_array)
    
    # Get top prediction
    predicted_index = np.argmax(calibrated_preds)
    predicted_class = class_names[predicted_index]
    confidence = calibrated_preds[predicted_index]
    
    # Decision logic with multiple criteria
    decision_factors = {
        'confidence': confidence,
        'entropy': uncertainty['entropy'],
        'top2_diff': uncertainty['top2_diff'],
        'effective_classes': uncertainty['effective_classes'],
        'likely_leaf': leaf_check['likely_leaf']
    }
    
    # Determine prediction reliability
    if not leaf_check['likely_leaf']:
        status = "REJECTED"
        reason = "Image doesn't appear to be a plant leaf"
        result_text = f"❌ {reason}"
        
    elif confidence < 0.5:  # Lower threshold but with other checks
        status = "LOW_CONFIDENCE"
        reason = "Model is uncertain about this prediction"
        result_text = f"🤔 {predicted_class} - {reason} (Confidence: {confidence:.2f})"
        
    elif uncertainty['entropy'] > 2.0:  # High entropy = high uncertainty
        status = "HIGH_UNCERTAINTY"
        reason = "High uncertainty across multiple classes"
        result_text = f"❓ {predicted_class} - {reason} (Confidence: {confidence:.2f})"
        
    elif uncertainty['top2_diff'] < 0.1:  # Top 2 predictions are very close
        status = "AMBIGUOUS"
        reason = "Multiple possible diagnoses"
        # Show top 2 predictions
        top2_indices = np.argsort(calibrated_preds)[-2:][::-1]
        top2_classes = [class_names[i] for i in top2_indices]
        top2_probs = [calibrated_preds[i] for i in top2_indices]
        result_text = f"⚠️ Ambiguous: {top2_classes[0]} ({top2_probs[0]:.2f}) or {top2_classes[1]} ({top2_probs[1]:.2f})"
        
    elif uncertainty['effective_classes'] > 5:  # Too many classes have significant probability
        status = "SCATTERED"
        reason = "Prediction scattered across many classes"
        result_text = f"🔄 {predicted_class} - {reason} (Confidence: {confidence:.2f})"
        
    else:
        status = "CONFIDENT"
        reason = "High confidence prediction"
        # Format class name for better readability
        formatted_class = predicted_class.replace('_', ' ').replace('__', ' - ')
        result_text = f"✅ {formatted_class} (Confidence: {confidence:.2f})"
    
    return {
        'status': status,
        'prediction': predicted_class,
        'formatted_prediction': result_text,
        'confidence': confidence,
        'raw_confidence': raw_preds[predicted_index],
        'uncertainty_metrics': uncertainty,
        'leaf_check': leaf_check,
        'decision_factors': decision_factors,
        'reason': reason
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    
    if not file:
        return render_template('index.html', 
                             prediction="❌ No file uploaded",
                             details="Please select an image file to upload.")
    
    try:
        # Load and preprocess image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((150, 150))  # Match model input size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make robust prediction
        result = make_robust_prediction(img_array)
        
        # Prepare detailed information
        details = []
        details.append(f"Status: {result['status']}")
        details.append(f"Reason: {result['reason']}")
        
        if temperature_scaler:
            details.append(f"Raw confidence: {result['raw_confidence']:.3f}")
            details.append(f"Calibrated confidence: {result['confidence']:.3f}")
        else:
            details.append(f"Confidence: {result['confidence']:.3f}")
            
        details.append(f"Uncertainty (entropy): {result['uncertainty_metrics']['entropy']:.3f}")
        details.append(f"Top-2 difference: {result['uncertainty_metrics']['top2_diff']:.3f}")
        
        # Add leaf check info
        if not result['leaf_check']['likely_leaf']:
            details.append("⚠️ Image may not be a plant leaf")
            details.append(f"Green dominance: {result['leaf_check']['green_dominance']:.3f}")
        
        details_text = " | ".join(details)
        
        return render_template('index.html', 
                             prediction=result['formatted_prediction'],
                             details=details_text)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction=f"❌ Error processing image",
                             details=f"Error details: {str(e)}")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """API endpoint that returns JSON response"""
    file = request.files.get('file')
    
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    
    try:
        # Load and preprocess image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make robust prediction
        result = make_robust_prediction(img_array)
        
        # Return detailed JSON response
        return jsonify({
            'success': True,
            'prediction': {
                'class': result['prediction'],
                'formatted': result['formatted_prediction'],
                'confidence': float(result['confidence']),
                'status': result['status'],
                'reason': result['reason']
            },
            'uncertainty': {
                'entropy': float(result['uncertainty_metrics']['entropy']),
                'top2_difference': float(result['uncertainty_metrics']['top2_diff']),
                'effective_classes': float(result['uncertainty_metrics']['effective_classes'])
            },
            'leaf_check': result['leaf_check'],
            'calibrated': temperature_scaler is not None
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'calibrated': temperature_scaler is not None,
        'classes': len(class_names)
    })

if __name__ == '__main__':
    print("🌱 Plant Disease Detection App Starting...")
    print(f"📊 Model loaded with {len(class_names)} classes")
    print(f"🎯 Calibration: {'Enabled' if temperature_scaler else 'Disabled'}")
    print("🚀 Server starting on http://localhost:5000")
    app.run(debug=True)