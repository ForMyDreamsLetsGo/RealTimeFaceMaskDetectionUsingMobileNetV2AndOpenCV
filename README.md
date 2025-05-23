# RealTimeFaceMaskDetectionUsingMobileNetV2AndOpenCV

facemask.py :- Used to train the face mask detection model on Google Colab Free tier
detectMaskVideo.py:- used to run real time face mask detection using OpenCV on laptop webacm
Enviornment:
  Tensorflow-cpu 2.18.0
  Python 2.10.0

About the model:
  :- Multi class Face Mask detection: Mask, NO Mask, Mask Worn Incorrect
  :- Training Data: Publicly available andrewmd dataset from Kaggle
  :- Augmentation:Random flip, resize, Random lightning , blur , Random Contrast, Random Hue
  :- Custom F1EarlyStopping callback to focus more on F1 score during Training instead of Validation Accuracy.

  ...Under Maintenance . more details will be added later


  Contact Dhruv.correspond@gmail.com if help needed with project implementation, project file or presentation
