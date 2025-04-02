# SignSage
![](https://github.com/anwesa-sinha/Innovative-project-3/assets/94695669/dd6520fd-3c8e-4f62-95a0-2b1872462e69)
## 📌 About SignSage

SignSage is an innovative real-time sign language interpretation system designed to bridge the communication gap between individuals using sign language and those unfamiliar with it. Utilizing cutting-edge technologies, including OpenCV, MediaPipe, and machine learning algorithms, SignSage recognizes and translates sign language gestures into spoken or written language, enhancing accessibility and inclusivity.

## 🛠️ Features

- **Real-Time Gesture Recognition**: Uses computer vision to identify and interpret sign language gestures.
- **Flask-Based Web Application**: Provides an intuitive and user-friendly interface.
- **Machine Learning Integration**: Employs trained models for accurate recognition and translation.
- **Text-to-Speech Support**: Converts recognized gestures into speech for seamless communication.
- **Cross-Platform Compatibility**: Works on various operating systems with Python support.

---

## 📋 System Requirements

Before installing SignSage, ensure that your system meets the following requirements:

### ✅ Prerequisites

- **Python** (Download: [Python Official Website](https://www.python.org/downloads))
- **Flask 3.0.1** <i>Framework for backend development<i>
- **OpenCV 4.7.0.68** <i>Computer vision library for image processing<i>
- **MediaPipe 0.9.0.1** <i>Framework for hand tracking and gesture recognition<i>
- **scikit-learn 1.2.1** <i>Machine learning library for model training<i>
- **pyttsx3** <i>Text-to-speech conversion library<i>
- **pyaudio** <i>Audio processing library<i>
- **Google Generative AI SDK** <i>`google-genai` for advanced AI-based translation<i>

---

## 🚀 Installation Guide

To install and set up SignSage on your local machine, follow these steps:

1. Clone the repository to your local system:

  ```bash
   git clone https://github.com/anwesa-sinha/Innovative-project-3.git
   cd Innovative-project-3
  ```

2. Install the following dependencies

```bash
  pip install Flask
  pip install opencv-python==4.7.0.68
  pip install mediapipe==0.9.0.1
  pip install scikit-learn==1.2.1
  pip3 install pyttsx3
  pip3 install pyaudio
  pip install -q -U google-genai
  
```
 🔗 Link  (if pip fails to work)
 Flask 3.0.1 : [Click Here](https://pypi.org/project/Flask/)<br>
 opencv-python 4.7.0.68 : [Click Here](https://pypi.org/project/opencv-python/4.7.0.68/)<br>
 mediapipe 0.9.0.1 : [Click Here](https://pypi.org/project/mediapipe/0.9.0.1/)<br>
 scikit-learn 1.2.1 : [Click Here](https://pypi.org/project/scikit-learn/1.2.1/)<br>


3. Run the project
```bash
   set TF_ENABLE_ONEDNN_OPTS=0
   python app.py
```
4. Open your web browser and access the application at:
```bash
   http://127.0.0.1:5000/
```



 
