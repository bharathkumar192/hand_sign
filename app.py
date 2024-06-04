from ultralytics import YOLO
from flask import Flask, Response, request
import cv2
import numpy as np
import threading
from flask_ngrok import run_with_ngrok
import threading
from pyngrok import ngrok
from flask_cors import CORS
import os
import gdown

app = Flask(__name__)
cors = CORS(app)
run_with_ngrok(app)

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

# Global variables for video stream and processing
video_stream = None
predictions_queue = []
lock = threading.Lock()
cv = threading.Condition()

def process_frames():
    global video_stream, predictions_queue, cv

    while True:
        with cv:  # Acquire the condition variable lock
            cv.wait_for(lambda: video_stream is not None)  # Wait for video_stream to be available

            ret, frame = video_stream.read()
            if not ret:
                with cv:
                    video_stream = None  # Reset when the stream ends
                    cv.notify_all()  # Notify other threads
                break

            results = model.predict(source=frame, stream=True)
            predictions = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls)
                    prediction_value = result.names[cls]
                    print("====",prediction_value)
                    predictions.append(prediction_value)

            with cv:  # Use lock to avoid race conditions on predictions_queue
                predictions_queue.append(predictions)
                cv.notify_all()  # Notify the generator thread



@app.route('/')
def home():
    return "Flask App for Tamil Hand Sign"

@app.route('/predictions', methods=['POST'])
def predictions_feed():
    global video_stream, cv

    video_data = request.get_data()
    video_stream = cv2.imdecode(np.frombuffer(video_data, np.uint8), cv2.IMREAD_COLOR)

    with cv:  # Notify the processing thread that video_stream is available
        cv.notify_all()

    return Response(gen_predictions(), mimetype='text/event-stream')


def gen_predictions():
    global predictions_queue, cv
    while True:
        with cv:
            cv.wait_for(lambda: predictions_queue or video_stream is None)  # Wait for predictions or stream end
            if not predictions_queue and video_stream is None:
                break  # Stream ended, stop the generator
            if predictions_queue:
                predictions = predictions_queue.pop(0)
                print(predictions)
                yield f'data: {predictions}\n\n'

if __name__ == '__main__':
    # app.run(debug=True)
    ngrok.set_auth_token("2gtde6QPcIZPmHSSq0ZhCfW3jli_3ANYfarF8RhuJQbFB7ym1")
    public_url = ngrok.connect(5000, hostname="monarch-modest-cod.ngrok-free.app")
    print(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000/\"")
    app.config["BASE_URL"] = public_url
    app.run()
