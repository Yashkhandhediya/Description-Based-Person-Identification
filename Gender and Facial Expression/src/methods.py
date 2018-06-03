import cv2
import numpy as np
from time import sleep
import tensorflow as tf
from data_bridge import *
import sys
import os
import time
import sys
from statistics import mode
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input



class Raw_file:
    def __init__(self, root):
        self.data_bridge = Singleton(Data_bridge)
        self.gui_root = root

    def main_thread(self):
        if self.data_bridge.processing_chosen_by_radio_butten=='vid':
            self.cap = cv2.VideoCapture(self.data_bridge.selected_video_file_path)
            while self.data_bridge.start_process_manager:
                ret, frame = self.cap.read()
                cv2.imshow('Selected Video', frame)
                cv2.waitKey(1)
                self.gui_root.update()
            cv2.destroyAllWindows()
            self.cap.release()
        if self.data_bridge.processing_chosen_by_radio_butten=='web':
            self.cap = cv2.VideoCapture(0)
            while self.data_bridge.start_process_manager:
                ret, frame = self.cap.read()
                cv2.imshow('Camera window', frame)
                cv2.waitKey(1)
                self.gui_root.update()
            cv2.destroyAllWindows()
            self.cap.release()
        if self.data_bridge.processing_chosen_by_radio_butten=='img':
            img1 = cv2.imread(self.data_bridge.selected_video_file_path,1)
            img1=cv2.resize(img1,(720,480))
            while self.data_bridge.start_process_manager:
                cv2.imshow("Image window",img1)
                self.gui_root.update()
            cv2.destroyAllWindows()


class Face_detection:
    def __init__(self, root):
        self.data_bridge = Singleton(Data_bridge)
        self.gui_root = root

    def main_thread(self):

        if self.data_bridge.processing_chosen_by_radio_butten == 'img':
            flag=0
            detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
            emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
            gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
            emotion_labels = get_labels('fer2013')
            gender_labels = get_labels('imdb')
            face_detection = load_detection_model(detection_model_path)
            emotion_classifier = load_model(emotion_model_path, compile=False)
            gender_classifier = load_model(gender_model_path, compile=False)
            emotion_target_size = emotion_classifier.input_shape[1:3]
            gender_target_size = gender_classifier.input_shape[1:3]

            while self.data_bridge.start_process_manager and flag==0:
                flag=1
                image_path = self.data_bridge.selected_video_file_path
                font = cv2.FONT_HERSHEY_SIMPLEX

                # hyper-parameters for bounding boxes shape
                gender_offsets = (30, 60)
                gender_offsets = (10, 10)
                emotion_offsets = (20, 40)
                emotion_offsets = (0, 0)

                rgb_image = load_image(image_path, grayscale=False)
                gray_image = load_image(image_path, grayscale=True)
                gray_image = np.squeeze(gray_image)
                gray_image = gray_image.astype('uint8')

                faces = detect_faces(face_detection, gray_image)
                for face_coordinates in faces:
                    x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
                    rgb_face = rgb_image[y1:y2, x1:x2]
                    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                    gray_face = gray_image[y1:y2, x1:x2]
                    try:
                        rgb_face = cv2.resize(rgb_face, (gender_target_size))
                        gray_face = cv2.resize(gray_face, (emotion_target_size))
                    except:
                        continue
                    rgb_face = preprocess_input(rgb_face, False)
                    rgb_face = np.expand_dims(rgb_face, 0)
                    gender_prediction = gender_classifier.predict(rgb_face)
                    gender_label_arg = np.argmax(gender_prediction)
                    gender_text = gender_labels[gender_label_arg]
                    gray_face = preprocess_input(gray_face, True)
                    gray_face = np.expand_dims(gray_face, 0)
                    gray_face = np.expand_dims(gray_face, -1)
                    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
                    emotion_text = emotion_labels[emotion_label_arg]
                    if gender_text == gender_labels[0]:
                        color = (0, 0, 255)
                    else:
                        color = (255, 0, 0)
                    draw_bounding_box(face_coordinates, rgb_image, color)
                    draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
                    draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)


                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite('../images/predicted_test_image.png', bgr_image)

                print("File has been stored in Images folder")
                print("Press stop processing to exit")

            while self.data_bridge.start_process_manager:
                self.gui_root.update()

        if( (self.data_bridge.processing_chosen_by_radio_butten == 'vid') or (self.data_bridge.processing_chosen_by_radio_butten=='web')):
            detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
            emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
            gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
            emotion_labels = get_labels('fer2013')
            gender_labels = get_labels('imdb')
            # Models
            face_detection = load_detection_model(detection_model_path)
            emotion_classifier = load_model(emotion_model_path, compile=False)
            gender_classifier = load_model(gender_model_path, compile=False)
            emotion_target_size = emotion_classifier.input_shape[1:3]
            gender_target_size = gender_classifier.input_shape[1:3]

            while self.data_bridge.start_process_manager:
                font = cv2.FONT_HERSHEY_SIMPLEX
                frame_window = 10
                gender_offsets = (30, 60)
                emotion_offsets = (20, 40)
                gender_window = []
                emotion_window = []
                # starting video streaming
                cv2.namedWindow('Window_frame')
                if self.data_bridge.processing_chosen_by_radio_butten=='vid':
                    self.cap=cv2.VideoCapture(self.data_bridge.selected_video_file_path)
                else:
                    self.cap = cv2.VideoCapture(0)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('Save.avi', fourcc, 20.0, (720, 480))
                while self.data_bridge.start_process_manager:
                    ret, bgr_image = self.cap.read()
                    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                    faces = detect_faces(face_detection, gray_image)

                    for face_coordinates in faces:

                        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
                        rgb_face = rgb_image[y1:y2, x1:x2]

                        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                        gray_face = gray_image[y1:y2, x1:x2]
                        try:
                            rgb_face = cv2.resize(rgb_face, (gender_target_size))
                            gray_face = cv2.resize(gray_face, (emotion_target_size))
                        except:
                            continue
                        gray_face = preprocess_input(gray_face, False)
                        gray_face = np.expand_dims(gray_face, 0)
                        gray_face = np.expand_dims(gray_face, -1)
                        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
                        emotion_text = emotion_labels[emotion_label_arg]
                        emotion_window.append(emotion_text)

                        rgb_face = np.expand_dims(rgb_face, 0)
                        rgb_face = preprocess_input(rgb_face, False)
                        gender_prediction = gender_classifier.predict(rgb_face)
                        gender_label_arg = np.argmax(gender_prediction)
                        gender_text = gender_labels[gender_label_arg]
                        gender_window.append(gender_text)

                        if len(gender_window) > frame_window:
                            emotion_window.pop(0)
                            gender_window.pop(0)
                        try:
                            emotion_mode = mode(emotion_window)
                            gender_mode = mode(gender_window)
                        except:
                            continue

                        if gender_text == gender_labels[0]:
                            color = (0, 0, 0)
                        else:
                            color = (0, 0, 0)

                        draw_bounding_box(face_coordinates, rgb_image, color)
                        draw_text(face_coordinates, rgb_image, gender_mode,
                                  color, 0, -20, 1, 1)
                        draw_text(face_coordinates, rgb_image, emotion_mode,
                                  color, 0, -45, 1, 1)

                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Window_frame', bgr_image)
                    self.gui_root.update()
                self.cap.release()
                cv2.destroyAllWindows()

