from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Get correct path to dataset (safer than ../)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "test.csv")

# Load dataset once when server starts
data = pd.read_csv(DATA_PATH)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Pick a random row (makes it feel dynamic)
        row = data.sample(1).iloc[0]

        # Convert row to dictionary
        result = row.to_dict()

        return jsonify({
            "status": "success",
            "sample": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
