from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("ml_webapp/crop_recommendation_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['nitrogen']),
            float(request.form['phosphorus']),
            float(request.form['potassium']),
            float(request.form['moisture']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            int(request.form['farming_method'])  
        ]

        print("Input Data:", data)  # ✅ Debugging Step

        data = np.array(data).reshape(1, -1)

        prediction = model.predict(data)
        crop = "Rice" if prediction[0] == 0 else "Maize"

        print("Predicted Crop:", crop)  # ✅ Debugging Step

        return jsonify({'crop': crop})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
