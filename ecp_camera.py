from flask import Flask, render_template, Response
import cv2
import torch
import os

app = Flask(__name__)

# Load the YOLOv5 model (ensure correct path to the .pt file)
model_path = '/home/cyberwolf/Desktop/uploads/yolov5/ecp.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load YOLOv5 model from the provided path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Function to generate video frames with YOLOv5 detection
def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        # Read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform object detection using YOLOv5
            results = model(frame)

            # Render results on the frame (bounding boxes, labels, etc.)
            result_img = results.render()[0]

            # Encode the result image as a JPEG format
            ret, buffer = cv2.imencode('.jpg', result_img)
            frame = buffer.tobytes()

            # Yield the frame in the required format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Render the main page template
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Route to serve the video feed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
