from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import pickle

app = Flask(__name__)

model_dict = pickle.load(open('ML/model.pickle', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

@app.route('/')
def home():
    return render_template('index.html')

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process the frame and send it to the client
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    frame_data = request.data  # Receive frame data from the client

    # Your machine learning prediction code here
    # Example: Replace this with your actual machine learning prediction logic
    prediction = model.predict([np.frombuffer(frame_data, dtype=np.uint8)])

    predicted_character = chr(ord('A') + int(prediction[0]))

    return jsonify({'prediction': predicted_character})

if __name__ == "__main__":
    app.run(debug=True)
