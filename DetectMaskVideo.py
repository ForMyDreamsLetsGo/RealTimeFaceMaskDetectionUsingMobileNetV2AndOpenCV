from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from keras.preprocessing.image import img_to_array
#from keras.models import load_model
import tensorflow as tf
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os
import time

IMG_SIZE = 256
LABELS = {0: 'with_mask', 1: 'without_mask', 2: 'mask_weared_incorrect'}
COLORS = {'with_mask': (0, 255, 0), 'without_mask': (0, 0, 255), 'mask_weared_incorrect': (0, 255, 255)}

FACE_MODEL_DIR = 'face_detector'
MASK_MODEL_PATH = 'best_modelV2d.keras'
CONFIDENCE_THRESH = 0.5

prototxtPath = os.path.join(FACE_MODEL_DIR, 'deploy.prototxt')
weightsPath = os.path.join(FACE_MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
face_net = cv2.dnn.readNet(prototxtPath, weightsPath)
mask_net = tf.keras.models.load_model(MASK_MODEL_PATH,compile=False)

face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#for p in (prototxtPath,weightsPath,MASK_MODEL_PATH):
 #   if not os.path.isfile(p):
  #      raise FileNotFoundError(f"Required file not found: {p}")

def detect_and_predict_mask(frame):
    (h, w) = frame.shape[:2]
    blob= cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections=face_net.forward()

    faces, locs=[],[]
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < CONFIDENCE_THRESH:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w-1, endX), min(h-1, endY)
        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = preprocess_input(face)
        faces.append(face)
        locs.append((startX, startY, endX, endY))

    preds=[]
    if faces:
      preds=mask_net.predict(np.stack(faces),batch_size=8)
    return locs,preds

print('[INFO] Starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

try:
  while True:
    frame=vs.read()
    if frame is None:
      continue
    frame=cv2.resize(frame,(400,int(frame.shape[0]*400/frame.shape[1])))
    locs,preds=detect_and_predict_mask(frame)
    for (box, pred) in zip(locs, preds):
      (startX, startY, endX, endY) = box
      idx=np.argmax(pred)
      label=LABELS[idx]
      color=COLORS[label]
      confidence=pred[idx]*100
      text=f'{label}: {confidence:.2f}%'
      cv2.putText(frame, text, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
      cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)

    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF
    if key == ord('q'):
      break
except KeyboardInterrupt:
    pass
finally:
    print('[INFO] Cleaning up...')
    cv2.destroyAllWindows()
    vs.stop()
