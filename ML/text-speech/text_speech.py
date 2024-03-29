#to install the library - pip install pyttsx3

import pyttsx3

#initialize the library
text_speech = pyttsx3.init()

print("Enter your text to be converted to speech: ")
text = input()

#get the speech rate property
rate = text_speech.getProperty('rate')

# Set a slower speech rate
text_speech.setProperty('rate', rate - 70)  # adjust the number to increase or decrease speech rate

text_speech.say(text) #voice out the text
text_speech.runAndWait() #Waits for speech to finish before continuing

#if continuous running of the program needed then put the code block in a while true loop
