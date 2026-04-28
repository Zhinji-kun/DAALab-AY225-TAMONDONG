from flask import Flask, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "test.csv")

try:
    data = pd.read_csv(DATA_PATH)
    print("CSV loaded successfully")
except Exception as e:
    print("CSV loading error:", e)


@app.route("/dataset/test.csv", methods=["GET"])
def dataset_csv():
    # Serve the raw CSV so the frontend can fetch it reliably.
    try:
        return send_file(DATA_PATH, mimetype="text/csv", as_attachment=False)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict", methods=["GET"])
def predict():
    try:
        row = data.sample(1).iloc[0]
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
