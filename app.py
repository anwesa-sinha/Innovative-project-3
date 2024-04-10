from flask import Flask, render_template, Response, jsonify
import time
import cv2
import numpy as np
import pickle
import mediapipe as mp

#to install library - pip3 install pyttsx3
#to install library - pip3 install speechrecognition 
#to install library - pip3 install pyaudio
import speech_recognition
import pyttsx3

app = Flask(__name__)

model_dict = pickle.load(open('./ML/model.pickle', 'rb'))
model = model_dict['model']

# global variable that will save the predicted texts
predicted_character = ''

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
               8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
               16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', }


def detect_hand_sign(frame):
    global predicted_character

    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        # character generated (output of gesture recognition)
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    return frame


def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_hand_sign(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
def text_speech():
    #initialize the library
    text_speech = pyttsx3.init()

    # print("Enter your text to be converted to speech: ")
    # text = input()

    text = predicted_character

    #get the speech rate property
    rate = text_speech.getProperty('rate')

    # Set a slower speech rate
    text_speech.setProperty('rate', rate - 70)  # adjust the number to increase or decrease speech rate

    text_speech.say(text) #voice out the text
    text_speech.runAndWait() #Waits for speech to finish before continuing

    #if continuous running of the program needed then put the code block in a while true loop

def speech_text():
    recognizer = speech_recognition.Recognizer()

    while True:
        try:
            with speech_recognition.Microphone() as mic:
                print("Speak please: \n")

                #accessing microphone
                recognizer.adjust_for_ambient_noise(mic, duration = 0.1)  #Duration for the amount of time it will take to recognize the speech
                audio = recognizer.listen(mic, timeout=3)  #if the mic does not hear any voices ... it times out in 3 secs (subject to change)

                #google supported english recognition
                text = recognizer.recognize_google(audio)
                text = text.lower()  #make all the text lowercase to handle grammatical errors.

                print(f"Recognized: {text}")

        except speech_recognition.UnknownValueError():
            #if we get some error we will make the recognizer object again and proceed as required
            recognizer = speech_recognition.Recognizer()
            continue


@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/get_text')
def get_text():
    # text_speech = pyttsx3.init()
    # rate = text_speech.getProperty('rate')
    # text_speech.setProperty('rate', rate - 70) 
    # text_speech.say(predicted_character)
    # text_speech.runAndWait()
    # Function to dynamically generate or fetch the text
    return jsonify({'text': predicted_character})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/text_to_speech')
def text_to_speech_route():
    text_speech()
    return 'Text to speech initiated'

# @app.route('/display',methods=['POST'])
# def display_text():
#     text = "Hello, world!"  # Replace this with the text you want to display
#     return render_template('camera.html', text=text)


if __name__ == '__main__':
    app.run(debug=True)