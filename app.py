from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

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
    return render_template("index.html")

@app.route("/predict-ui", methods=["POST"])
def predict_ui():

    data = [
        float(request.form["f1"]),
        float(request.form["f2"]),
        float(request.form["f3"]),
        float(request.form["f4"]),
        float(request.form["f5"]),
        float(request.form["f6"])
    ]

    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)[0]

    return render_template(
        "index.html",
        prediction=labels[int(prediction)]
    )



if __name__ == "__main__":
    port = 5001
    app.run(host="0.0.0.0", port=port)