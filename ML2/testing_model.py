# set TF_ENABLE_ONEDNN_OPTS=0

import pickle
import cv2
import mediapipe as mp
import numpy as np
import statistics
from google import genai


def sentence_formation(words):
#    client = genai.Client(api_key="AIzaSyBmMv7minae9QdQg7QjwEQ490CLdVzo2uE")
#    response = client.models.generate_content(
#    model="gemini-2.0-flash", 
#    contents=f"A mute person is showing me hand signs {{{', '.join(words)}}}. Give a simplest sentence or context he is trying to say.max token 10"
#    )
#    return(response.text)
   return("sentence formed")

model_dict = pickle.load(open('./model.pickle', 'rb'))
model = model_dict['model']
print("Model loaded")
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict =    {0: 'i',1: 'you',2: 'call',3: 'name',4: 'fine',
                  5: 'help',6: 'hi',7: 'bad', 8: 'eat',9: 'going',
                  10: "worry",11: "what",12:"enjoy",13: "Thank you",14:"0",
                  15:"1",16:"2",17:"3",18:"4",}
print("Label assigned")
no_of_frames = 0
each_frame_output = []
words = []
sentence = ""
while True:
    if(no_of_frames == 15):        
        
        if each_frame_output:  # This checks if the list is not empty
            mode_val = statistics.mode(each_frame_output)
            print(mode_val)
            words.append(mode_val)
        each_frame_output.clear()
        no_of_frames = 0

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        if ( len(results.multi_hand_landmarks) >1):            
            if(words):
                print(words)
                sentence= sentence_formation(words)
                print(sentence)
            
            cv2.putText(frame, sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            words.clear()                
            cv2.waitKey(1)
            continue
        sentence = ""
        no_of_frames = no_of_frames + 1
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

        predicted_character = labels_dict[int(prediction[0])]
        each_frame_output.append(predicted_character)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA)
        cv2.putText(frame, str(no_of_frames), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    elif sentence:
        cv2.putText(frame, sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()