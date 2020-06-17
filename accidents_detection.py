
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import os
import pyrebase
from datetime import datetime



options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 234100,
    'threshold': 0.6,
    'gpu': 0.7
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

config = {

    "apiKey": "AIzaSyCxyEHca1eSWz8JT9NiLGJTEApG94moaBU",
    "authDomain": "smarttraffic-2a170.firebaseapp.com",
    "databaseURL": "https://smarttraffic-2a170.firebaseio.com",
    "projectId": "smarttraffic-2a170",
    "storageBucket": "smarttraffic-2a170.appspot.com",
    "messagingSenderId": "383660536137"

}

firebase = pyrebase.initialize_app(config)

db = firebase.database()


#capture = cv2.VideoCapture(r"C:\Users\ASUS\Desktop\accident1.mp4")
capture = cv2.VideoCapture("accidents.mp4")
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

d = ["false", "percent", "time"]

current_time = time.time()
current_time1 = current_time


while True:
    stime = time.time()
    ret, frame = capture.read()
    time1 = datetime.time(datetime.now())
    time_now = str(time1)
    print(time.time() - current_time1)
    if time.time() - current_time1 >= 5:
        current_time1 = time.time()
        if (d[0] == "accidents"):
            db.child("mydata").update({"accidents": "true"})
            db.child("mydata").update({"percent": d[1]})
            db.child("mydata").update({"time": d[2]})
        else:
            db.child("mydata").update({"accidents": "false"})
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            print(label)
            confidence = result['confidence']
            percent = str('{:.0f}%'.format(confidence * 100))
            if label == "accidents" and confidence >= 0.8:
                d[0] = label
                d[1] = percent
                d[2] = int(time.time())
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 10)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('Object_Detection', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
