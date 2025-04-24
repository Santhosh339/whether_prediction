# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np
# import joblib

# # Load dataset
# df = pd.read_csv('C:\\Users\\venusraj\\Desktop\\whether_prediction\\Weather_Data.csv')

# # Extract Hour
# df['Hour'] = pd.to_datetime(df['Date/Time']).dt.hour

# # Simplify Weather
# def simplify_weather(weather):
#     weather = weather.lower()
#     if 'fog' in weather or 'mist' in weather:
#         return 'Fog'
#     elif 'snow' in weather or 'ice pellets' in weather:
#         return 'Snow'
#     elif 'rain' in weather or 'drizzle' in weather:
#         return 'Rain'
#     elif 'cloudy' in weather:
#         return 'Cloudy'
#     elif 'clear' in weather:
#         return 'Clear'
#     else:
#         return 'Other'

# df['Weather'] = df['Weather'].apply(simplify_weather)

# # Features
# features = ['Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa', 'Hour']
# X = df[features]
# y = df['Weather']

# # Check missing values
# print("Missing Values:\n", X.isnull().sum())

# # Encode target
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)

# # Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# # Train model with class weights
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# rf_model.fit(X_train, y_train)

# # Evaluate
# y_pred = rf_model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# # Save model and scaler
# joblib.dump(rf_model, 'weather_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# np.save('weather_classes.npy', le.classes_)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('C:\\Users\\venusraj\\Desktop\\whether_prediction\\Weather_Data.csv')

# Extract Hour
df['Hour'] = pd.to_datetime(df['Date/Time']).dt.hour

# Simplify Weather
def simplify_weather(weather):
    weather = weather.lower()
    if 'fog' in weather or 'mist' in weather:
        return 'Fog'
    elif 'snow' in weather or 'ice pellets' in weather:
        return 'Snow'
    elif 'rain' in weather or 'drizzle' in weather:
        return 'Rain'
    elif 'cloudy' in weather:
        return 'Cloudy'
    elif 'clear' in weather:
        return 'Clear'
    else:
        return 'Other'

df['Weather'] = df['Weather'].apply(simplify_weather)

# Features
features = ['Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa', 'Hour']
X = df[features]
y = df['Weather']

# Check missing values
print("Missing Values:\n", X.isnull().sum())

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and scaler
joblib.dump(rf_model, 'weather_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
np.save('weather_classes.npy', le.classes_)