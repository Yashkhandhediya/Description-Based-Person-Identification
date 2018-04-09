import cv2
import numpy as np
from time import sleep
import tensorflow as tf
from data_bridge import *
import yolo_model
import yolo_settings

# from utils import VOC
import os
import time
# import matplotlib.pyplot as plt TODO: check what error comes when using matplotlib


class Raw_video:
    """
    Whenever we want to just watch raw video this  class will be used for method.
    """

    def __init__(self, root):
        self.data_bridge = Singleton(Data_bridge)
        self.gui_root = root

    def main_thread(self):
        self.cap = cv2.VideoCapture(self.data_bridge.selected_video_file_path)
        while self.data_bridge.start_process_manager:
            ret, frame = self.cap.read()
            cv2.imshow('window', frame)
            cv2.waitKey(1)
            self.gui_root.update()
        cv2.destroyAllWindows()
        self.cap.release()


class YOLO_person_detection:
    """
    YOLO person detection
    """

    def __init__(self, root):
        self.sess = tf.InteractiveSession()
        self.model = yolo_model.Model(training=False)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        self.boundary1 = yolo_settings.cell_size * yolo_settings.cell_size * yolo_settings.num_class
        self.boundary2 = self.boundary1 + yolo_settings.cell_size * yolo_settings.cell_size * yolo_settings.box_per_cell
        try:
            self.saver.restore(self.sess, os.getcwd() + '/YOLO_small.ckpt')
            print 'load from past checkpoint'
        except:
            try:
                self.saver.restore(self.sess, os.getcwd() + '/YOLO_small.ckpt')
                print ('load from YOLO small pretrained')
            except:
                print ('you must train first, exiting..')
                exit(0)
        self.data_bridge = Singleton(Data_bridge)
        self.gui_root = root
        self.skip_frames = 50

    def main_thread(self):
        self.cap = cv2.VideoCapture(self.data_bridge.selected_video_file_path)
        num = 0
        while self.data_bridge.start_process_manager:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (720, 480))
            if num < self.skip_frames:
                print(num)
                num += 1
                cv2.imshow('Camera', frame)
                cv2.waitKey(10)
                continue
            result = self.detect(frame)
            # print(np.shape(result), "result shape", result[:][0])
            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(1)
            num += 1
            self.gui_root.update()
        cv2.destroyAllWindows()
        self.cap.release()

    def detect(self,img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (yolo_settings.image_size, yolo_settings.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, yolo_settings.image_size, yolo_settings.image_size, 3))
        result = self.detect_from_cvmat(inputs)[0]
        print (result)
        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / yolo_settings.image_size)
            result[i][2] *= (1.0 * img_h / yolo_settings.image_size)
            result[i][3] *= (1.0 * img_w / yolo_settings.image_size)
            result[i][4] *= (1.0 * img_h / yolo_settings.image_size)
        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.model.logits, feed_dict={self.model.images: inputs})
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))
        return results

    def interpret_output(self, output):
        probs = np.zeros((yolo_settings.cell_size, yolo_settings.cell_size, yolo_settings.box_per_cell, len(yolo_settings.classes_name)))
        class_probs = np.reshape(output[0: self.boundary1], (yolo_settings.cell_size, yolo_settings.cell_size, yolo_settings.num_class))
        scales = np.reshape(output[self.boundary1: self.boundary2],
                            (yolo_settings.cell_size, yolo_settings.cell_size, yolo_settings.box_per_cell))
        boxes = np.reshape(output[self.boundary2:], (yolo_settings.cell_size, yolo_settings.cell_size, yolo_settings.box_per_cell, 4))
        offset = np.transpose(
            np.reshape(np.array([np.arange(yolo_settings.cell_size)] * yolo_settings.cell_size * yolo_settings.box_per_cell),
                       [yolo_settings.box_per_cell, yolo_settings.cell_size, yolo_settings.cell_size]), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / yolo_settings.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= yolo_settings.image_size

        for i in range(yolo_settings.box_per_cell):
            for j in range(yolo_settings.num_class):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= yolo_settings.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > yolo_settings.IOU_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([yolo_settings.classes_name[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1],
                           boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
            # cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.CV_AA)

    def iou(self,box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)
