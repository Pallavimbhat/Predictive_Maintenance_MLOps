from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

model = joblib.load("rf_model.pkl")

labels = {
    0: "Healthy",
    1: "TWF",
    2: "HDF",
    3: "PWF",
    4: "OSF",
    5: "RNF"
}

@app.route("/")
def home():
    return "CI/CD Pipeline Successfully Updated"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["features"]

    prediction = model.predict([data])[0]

    return jsonify({
        "prediction": labels[int(prediction)]
    })



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)