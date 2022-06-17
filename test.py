# import các thư viện cần thiết
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

#Xây dựng kiến trúc mạng tích chập 2D (CNN)
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Lưu lại các trọng số của mô hình
emotion_model.load_weights('model.h5')
#Sử dụng openCV haarcascade xml phát hiện các khuôn mặt trong webcam và dự đoán cảm xúc
cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: "   Angry(tuc gian)   ", 1: "Disgusted(ghe tom)", 2: "  Fearful(so hai)  ", 3: "   Happy(hanh phuc)   ", 4: "  Neutral(tu nhien)  ",
                5: "    Sad(buon)    ", 6: "Surprised(bat ngo)"}
emoji_dist = {0: "./emojis/angry.png", 2: "./emojis/disgusted.png", 2: "./emojis/fearful.png", 3: "./emojis/happy.png",
              4: "./emojis/neutral.png", 5: "./emojis/sad.png", 6: "./emojis/surpriced.png"}
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
#Sau khi train xong ta dùng opencv với file haarcascadefrontalfacedefault.xml để get ra vị trí khuôn mặt, rồi load_model ta vừa train bên trên để dự đoán xem cảm xúc của khuôn mặt là gì:
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces: # tạo khung hình
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0) # tỉ lệ khung
        emotion_prediction = emotion_model.predict(cropped_img) 
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(0)
cap.release()
cv2.destroyAllWindows()
