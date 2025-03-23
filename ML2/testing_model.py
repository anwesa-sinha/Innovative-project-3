import pickle
import cv2
import mediapipe as mp
import numpy as np
import statistics

model_dict = pickle.load(open('./model.pickle', 'rb'))
model = model_dict['model']
print("Model loaded")
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict =    {0: 'A',1: 'B',2: 'C',3: 'D',4: 'E',5: 'F',6: 'G',7: 'H',
                  8: 'I',9: 'J',}
print("Label assigned")
no_of_frames = 0
each_frame_output = []
words = []
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
            cv2.imshow('frame', frame)
            if(words):
                print(words)
            words.clear()
            cv2.waitKey(1)
            continue
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

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()