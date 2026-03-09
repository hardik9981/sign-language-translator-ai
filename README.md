# 🤟 Sign Language Translator AI

A real-time **Sign Language Recognition System** that detects hand gestures using a webcam and converts them into **text and speech**.

This project uses **Computer Vision and Deep Learning** to help bridge communication gaps for people with hearing or speech impairments.

---

# 🚀 Features

* Real-time gesture recognition using webcam
* Converts hand gestures to readable text
* Text-to-Speech output for detected gestures
* Deep Learning model trained on custom dataset
* Built using TensorFlow, OpenCV, and Python
* Supports multiple hand gestures

---

# 🧠 Supported Gestures

The model can currently recognize the following gestures:

* Hello
* Help
* Thank You
* Yes
* No
* Stop
* Please
* Drink
* Eat
* Love

---

# 🛠 Tech Stack

**Programming Language**

* Python

**Libraries & Frameworks**

* TensorFlow / Keras
* OpenCV
* NumPy
* pyttsx3 (Text-to-Speech)

**Tools**

* VS Code
* Git & GitHub

---

# 📂 Project Structure

```
SignProject
│
├ images              # Demo screenshots
├ models              # Trained model files
│   ├ sign_model.h5
│   └ class_indices.json
│
├ collect_data.py     # Script to collect gesture images
├ organize_dataset.py # Dataset organization
├ train_model.py      # Model training
├ predict_webcam.py   # Real-time prediction
│
├ requirements.txt
├ README.md
└ .gitignore
```

---

# ⚙️ Installation

Clone the repository

```
git clone https://github.com/yourusername/sign-language-translator-ai.git
cd sign-language-translator-ai
```

Install dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Start the real-time gesture detection system:

```
python predict_webcam.py
```

A webcam window will open and the model will start detecting hand gestures.

images/demo1.png
images/demo2.png
images/demo3.png
images/demo4.png
images/demo5.png
images/demo6.png
images/demo7.png

# 📸 Demo

Below are some example predictions from the system.

---

# 📊 Model Training

The deep learning model was trained on a **custom dataset of hand gestures** using TensorFlow.

* Image size: 224x224
* Training epochs: 15
* Validation accuracy: ~93%

---

# 🔮 Future Improvements

* More gesture classes
* Sentence generation from gestures
* Web deployment
* Mobile application version

---

# 👨‍💻 Author

**Hardik Kumar **

Cybersecurity & AI Enthusiast
Open Source Learner
