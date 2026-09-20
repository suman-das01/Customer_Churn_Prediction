from flask import Flask, request, jsonify, render_template
import json
import math

app = Flask(__name__)

# Load deployment artifact
with open("churn_artifact.json", "r") as f:
    artifact = json.load(f)


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# Prepare input features
def prepare_features(data):

    features = []

    # Numerical features
    for i, col in enumerate(artifact["numerical_cols"]):

        value = float(data[col])

        mean = artifact["scaler_mean"][i]
        scale = artifact["scaler_scale"][i]

        scaled_value = (value - mean) / scale

        features.append(scaled_value)

    # Categorical features
    for i, col in enumerate(artifact["categorical_cols"]):

        user_value = str(data[col])

        categories = artifact["categories"][i]

        for category in categories:

            if user_value == category:
                features.append(1)
            else:
                features.append(0)

    return features


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = prepare_features(data)

    coefficients = artifact["coefficients"]
    intercept = artifact["intercept"]

    # Logistic Regression calculation
    z = intercept

    for feature, coefficient in zip(features, coefficients):
        z += feature * coefficient

    probability = sigmoid(z)

    prediction = 1 if probability >= 0.5 else 0

    result = {
        "prediction": prediction,
        "churn_probability": probability
    }

    return jsonify(result)


if __name__ == "__main__":

    print("Starting Customer Churn Prediction App...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )