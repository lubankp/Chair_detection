import cv2
import tensorflow as tf
import numpy as np


def live_detection(tracker):

    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        _, frame = cap.read()
        frame = frame[50:500, 50:500, :]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))
        yhat = tracker.predict(np.expand_dims(resized / 255, 0))
        sample_coords = yhat[1][0]

        if yhat[0] > 0.5:
            cv2.rectangle(frame,
                          tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                          tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                          (255, 0, 0), 2)
            cv2.rectangle(frame,
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                       [0, -30])),
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                       [80, 0])),
                                (255, 0, 0), -1)
            cv2.putText(frame, 'chair', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                        [0,-5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()