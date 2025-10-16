from flask import Flask, request, jsonify
import pickle
import numpy as np
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

@app.route("/health")
def health():
    return "Service is running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


