#to install library - pip3 install pyttsx3
#to install library - pip3 install speechrecognition 
#to install library - pip3 install pyaudio

import speech_recognition
import pyttsx3

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

