import cv2
import numpy as np
import tensorflow as tf
import json
import pyttsx3
import time

# load model
model = tf.keras.models.load_model("models/sign_model.h5")

# load classes
with open("models/class_indices.json") as f:
    classes = json.load(f)

IMG_SIZE = 224
CONF_THRESHOLD = 0.80

# voice
engine = pyttsx3.init()
engine.setProperty("rate",150)

last_spoken = ""
last_time = 0

cap = cv2.VideoCapture(0)

print("Press q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame,1)

    h,w,_ = frame.shape

    # center box
    x1 = int(w*0.3)
    x2 = int(w*0.7)
    y1 = int(h*0.2)
    y2 = int(h*0.8)

    roi = frame[y1:y2,x1:x2]

    img = cv2.resize(roi,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    preds = model.predict(img,verbose=0)

    class_id = int(np.argmax(preds))
    confidence = float(preds[0][class_id])

    label = "Show your hand"

    if confidence > CONF_THRESHOLD:
        label = classes.get(str(class_id),"Unknown")

        current_time = time.time()

        if label != last_spoken or current_time-last_time > 2:
            engine.say(label)
            engine.runAndWait()

            last_spoken = label
            last_time = current_time

    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (x1,y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Prediction",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()