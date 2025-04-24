from flask import Flask, request, render_template
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('weather_model.pkl')
scaler = joblib.load('scaler.pkl')
weather_classes = np.load('weather_classes.npy', allow_pickle=True)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    temperature = float(request.form['temperature_C'])
    pressure = float(request.form['pressure_kpa'])
    humidity = float(request.form['relative_humidity'])
    wind_speed = float(request.form['wind_speed_kmph'])
    visibility = float(request.form['visibility_km'])
    hour = float(request.form['hour'])

    # Create feature array (match train_model.py order)
    features = np.array([[temperature, humidity, wind_speed, visibility, pressure, hour]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)
    predicted_weather = weather_classes[prediction[0]]

    return render_template('index.html', prediction_text=f'Predicted Weather: {predicted_weather}')

if __name__ == '__main__':
    app.run(debug=True)