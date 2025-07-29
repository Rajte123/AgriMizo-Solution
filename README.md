# AgriMizo-Solution
🌾 Agri ML Projects – Crop Recommendation, Disease Detection & Weather Forecasting
This repository contains machine learning-based solutions for enhancing agricultural productivity, specifically focused on the state of Mizoram, India.
It includes:

✅ Crop Recommendation System for Mizoram

✅ Plant Disease Detection System for Mizoram

✅ Weather Forecasting Web App for Mizoram

📁 Project Structure
bash
Copy
Edit

agri_ml_projects/
├── crop_recommendation/      # ML model to recommend best crops
├── plant_disease_detection/  # Image-based disease classifier
├── weather_prediction/       # Weather forecast using API
└── README.md                 # Project overview
🔍 1. Crop Recommendation System
📌 Description
This model recommends the best crop to grow based on soil nutrients (N, P, K), temperature, humidity, pH, and rainfall.

🛠️ Tech Stack
Python, Pandas, scikit-learn

Jupyter Notebook

🚀 How to Run
bash
Copy
Edit
cd crop_recommendation
jupyter notebook crop_recommendation_model.ipynb
🌿 2. Plant Disease Detection System
📌 Description
A deep learning model that classifies plant diseases from leaf images.

🛠️ Tech Stack
TensorFlow / Keras, CNN, OpenCV

Python

🚀 How to Run
bash
Copy
Edit
cd plant_disease_detection
python train_model.py
python predict_disease.py
🌦️ 3. Weather Prediction Web App (Mizoram)
📌 Description
A simple weather forecast web app that provides weather for districts like Aizawl, Lunglei, Champhai, etc. using OpenWeatherMap API.

🛠️ Tech Stack
HTML, CSS, JavaScript

Weather API (OpenWeatherMap)

🚀 How to Run
bash
Copy
Edit
cd weather_prediction
open index.html
🔑 Get Free API Key
Go to OpenWeatherMap

Create a free account

Get your API key from the dashboard

Add it to script.js:

js
Copy
Edit
const API_KEY = "your_api_key_here";
