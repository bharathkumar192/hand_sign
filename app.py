from ultralytics import YOLO
from flask import Flask, request, jsonify
import cv2
import numpy as np
import gdown
import os
# from flask_ngrok import run_with_ngrok
# from pyngrok import ngrok

app = Flask(__name__)
# run_with_ngrok(app)  

MODEL_ID = "1eO5f_WpQsYbcCsB26F0hSLO-OrZ2EsGT"
MODEL_NAME = "best_3.pt"

if not os.path.exists(MODEL_NAME):
    print(f"Downloading {MODEL_NAME} from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, output=MODEL_NAME, quiet=False)
    print("Download complete.")
else:
    print(f"{MODEL_NAME} already exists in the current directory.")

# Load the model
model = YOLO("best_3.pt")

@app.route('/')
def home():
    return "Flask App for Tamil Hand Sign"

@app.route('/predictions', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = request.files['image'].read()  # Changed to request.files
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Make predictions
    results = model.predict(source=image)
    
    # Extract predictions
    predictions = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            predictions.append(result.names[cls])
            print(predictions)
            print(predictions)
    print(predictions)
    # Return the predictions as a JSON response
    return jsonify({'predictions': predictions}), 200, {'Content-Type': 'application/json'}


if __name__ == '__main__':
    # ngrok.set_auth_token("2gtde6QPcIZPmHSSq0ZhCfW3jli_3ANYfarF8RhuJQbFB7ym1")
    # public_url = ngrok.connect(5000, hostname="monarch-modest-cod.ngrok-free.app")
    # print(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000/\"")
    # app.config["BASE_URL"] = public_url
    app.run(debug=True)
