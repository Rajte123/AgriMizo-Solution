import os
import requests
from flask import Flask, render_template, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
API_KEY = os.getenv("WEATHER_API_KEY")

@app.route('/', methods=['GET', 'POST'])
def index():
    weather_data = None
    if request.method == 'POST':
        location = request.form['location']
        url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={location}&days=3"
        response = requests.get(url)
        if response.status_code == 200:
            weather_data = response.json()
        else:
            weather_data = {"error": "Location not found"}
    return render_template('index.html', weather=weather_data)
    
if __name__ == '__main__':
    app.run(debug=True)
