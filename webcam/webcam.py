#Dependcies

import numpy as np
import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#webcam cepture
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
   min_detection_confidence=0.5,
   min_tracking_confidence=0.5) as holistic:
   
   while True:
    ret, frame = cap.read()
    #convrt BGR to RGB
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #Make Detetction
    results = holistic.process(image)
    print(results.face_landmarks)

    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    #Draw face landmark
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)


    cv2.imshow('frame', image)

    if cv2.waitKey(1) == ord ("q"):
        break


cap.release()
cv2.destroyAllWindows()
