from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import pickle
import mediapipe as mp
import statistics
from google import genai
import threading
import pyttsx3

app = Flask(__name__)

def sentence_formation(words):
#    client = genai.Client(api_key="AIzaSyBmMv7minae9QdQg7QjwEQ490CLdVzo2uE")
#    response = client.models.generate_content(
#    model="gemini-2.0-flash", 
#    contents=f"A mute person is showing me hand signs {{{', '.join(words)}}}. Give a simplest sentence or context he is trying to say.max token 10"
#    )
#    return(response.text)
   return("sentence formed")


# Load the model
try:
    model_dict = pickle.load(open('./ML2/model.pickle', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Ensure './ML2/model.pickle' exists.")


# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict =    {0: 'i',1: 'you',2: 'call',3: 'name',4: 'fine',
                  5: 'help',6: 'hi',7: 'bad', 8: 'eat',9: 'going',
                  10: "worry",11: "what",12:"enjoy",13: "Thank you",14:"0",
                  15:"1",16:"2",17:"3",18:"4",}
# global variable that will save the predicted texts
predicted_character = ''
no_of_frames = 0
each_frame_output = []
words = []
sentence = ""
def detect_hand_sign(frame):
    """Detect hand signs in the given frame."""
    global predicted_character, no_of_frames,sentence
    if(no_of_frames == 15):                
        if each_frame_output:  # This checks if the list is not empty
            mode_val = statistics.mode(each_frame_output)
            words.append(mode_val)
        each_frame_output.clear()
        no_of_frames = 0

    data_aux = []
    # position of hand on screen
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        if ( len(results.multi_hand_landmarks) > 1):
            if(words):
                sentence= sentence_formation(words)
                print(sentence)
                # text_speech()
            cv2.putText(frame, sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            words.clear()   
        
        else:
            sentence = ""
            no_of_frames = no_of_frames + 1
            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(
                #     frame,  # image to draw
                #     hand_landmarks,  # model output
                #     mp_hands.HAND_CONNECTIONS,  # hand connections
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()
                # )

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

            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_word = labels_dict[int(prediction[0])]
                each_frame_output.append(predicted_word)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, str(no_of_frames), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Prediction error: {e}")

    elif sentence:
        cv2.putText(frame, sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
      
    return frame


def generate_frames():
    """Generate video frames from the webcam."""
    cap = cv2.VideoCapture(0)

    while True:
        try:
            success, frame = cap.read()
            if not success:
                break

            frame = detect_hand_sign(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in video feed: {e}")
            break
    cap.release()
        
def text_speech():
    """Convert the predicted text to speech."""
    if not sentence:
        print("No sentence to convert to speech.")
        return
    def speak ():
        try:
            #initialize the library
            text_speech = pyttsx3.init()

            text = sentence

            #get the speech rate property
            rate = text_speech.getProperty('rate')

            # Set a slower speech rate
            text_speech.setProperty('rate', rate - 70)  # adjust the number to increase or decrease speech rate

            text_speech.say(text) #voice out the text
            text_speech.runAndWait() #Waits for speech to finish before continuing
        except Exception as e:
            print(f"Text to speech error: {e}")

    #if continuous running of the program needed then put the code block in a while true loop
    speech_thread = threading.Thread(target=speak)
    speech_thread.start()

@app.route('/camera')
def camera():
    return render_template('new_camera.html') 

translation_history = [] 

@app.route('/get_text')
def get_text():
    global sentence
    return jsonify({'text': sentence})

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


if __name__ == '__main__':
    app.run(debug=True)