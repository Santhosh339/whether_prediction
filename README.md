Weather Prediction App 🌦️
A sleek and modern Weather Prediction App that uses machine learning to forecast weather conditions (Clear, Cloudy, Fog, Rain, Snow) based on user inputs like temperature, pressure, humidity, wind speed, visibility, and hour. Built with a Flask backend, a Random Forest model, and a responsive Tailwind CSS front-end, this project delivers accurate predictions with a professional, user-friendly interface.
🚀 Features

        Accurate Predictions: Leverages a Random Forest classifier trained on real-world weather data.
        Intuitive UI: Modern, card-based interface styled with Tailwind CSS for a seamless experience on all devices.
        Real-Time Interaction: Input weather parameters and get instant predictions via a Flask-powered backend.
        Data Preprocessing: Handles missing values, feature scaling, and weather category simplification.
        Easy Setup: Clear instructions for local deployment and testing.
📋 Prerequisites
        Python 3.9.12 or higher
        Git (for cloning the repository)
        Basic knowledge of Flask and machine learning
🛠️ Installation
        Clone the Repository:
        git clone https://github.com/your-username/weather-prediction-app.git
        cd weather-prediction-app


Create a Virtual Environment:
python -m venv venv


Windows:
venv\Scripts\activate


macOS/Linux:
source venv/bin/activate




Install Dependencies:
pip install pandas numpy scikit-learn flask joblib imblearn


Download the Dataset:

Place Weather Data.csv in the project root (weather-prediction-app/).Note: The dataset should have columns: Date/Time, Temp_C, Rel Hum_%, Wind Speed_km/h, Visibility_km, Press_kPa, Weather.



🚴‍♂️ Running the App

Train the Model:
python train_model.py


This generates weather_model.pkl, scaler.pkl, and weather_classes.npy for predictions.
Check the console for accuracy and classification report.


Start the Flask App:
python app.py


Open http://127.0.0.1:5000 in your browser.


Test Predictions:

Enter values in the form (e.g., Temperature: 4.6°C, Pressure: 99.26 kPa, Humidity: 72%, Wind Speed: 39.0 km/h, Visibility: 25.0 km, Hour: 1).
Click Predict Weather to see the result (e.g., Predicted Weather: Cloudy).



🎨 Example Inputs for Predictions



Weather
Temperature (°C)
Pressure (kPa)
Humidity (%)
Wind Speed (km/h)
Visibility (km)
Hour



Clear
-9.0
100.83
63
13.0
25.0
21


Cloudy
4.6
99.26
72
39.0
25.0
1


Fog
-1.8
101.24
86
4.0
8.0
0


Rain
3.1
99.68
88
15.0
12.9
19


Snow
-8.8
100.32
79
4.0
9.7
0


🧠 How It Works

    Data Preprocessing (train_model.py):
    
    Extracts Hour from Date/Time.
    Simplifies weather categories (e.g., “Rain Showers” → “Rain”).
    Scales features and encodes labels.
    Uses SMOTE to handle class imbalance for better Rain predictions.


Model Training:

    Trains a Random Forest classifier with 200 estimators.
    Saves the model, scaler, and class labels.


Web App (app.py, index.html):

      Flask serves a Tailwind-styled form.
      User inputs are scaled and fed to the model for real-time predictions.



📈 Performance

        Accuracy: ~73% (varies with dataset).
        Strengths: High recall for Fog (0.94) and Clear (0.78).
        Improvements: Enhanced Rain recall using SMOTE for balanced predictions.

🐛 Troubleshooting

        FileNotFoundError:
        Ensure Weather Data.csv, weather_model.pkl, scaler.pkl, and weather_classes.npy are in the project root.
        Update paths in train_model.py or app.py if needed.
        

Prediction Issues:
      Check the classification report from train_model.py for class-specific performance.
      Rerun train_model.py if predictions are inconsistent.


Form Not Working:
Verify templates/index.html exists and app.py is running.



🌟 Future Enhancements

        Add real-time weather API integration.
        Improve model accuracy with hyperparameter tuning.
        Deploy to Heroku or Vercel for public access.
        Enhance UI with weather icons and animations.

📝 License
      This project is licensed under the MIT License.
🙌 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the app.

        Fork the repository.
        Create a new branch (git checkout -b feature/your-feature).
        Commit changes (git commit -m 'Add your feature').
        Push to the branch (git push origin feature/your-feature).
        Open a Pull Request.

📬 Contact
      For questions or feedback, reach out via GitHub Issues or email at santhoshalakunta333@gmail.com.

⭐ Star this repo if you find it useful! Happy predicting! ☀️🌧️❄️
