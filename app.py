from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model("wind_speed_lstm_model.h5")

app = Flask(__name__)

@app.route('/')
def home():
    return "Wind Speed Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = np.array(data['features']).reshape(1, 5, 56)
    prediction = model.predict(input_features)
    return jsonify({'wind_speed': prediction[0][0]})

if __name__ == '__main__':
    app.run(debug=True)
