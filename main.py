import read_data_path as rp
import numpy as np
import cv2
import train as tr
from train import SequenceData
import yolo_loss
import tiny_yolov1_model
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class_dictionary = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
                    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
                    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
                    'sofa': 17, 'train': 18, 'tvmonitor': 19}
class_list = list(class_dictionary.keys())

if __name__ == "__main__":

    train_x, train_y, val_x, val_y, test_x, test_y = rp.make_data()
    train_generator = SequenceData(train_x, train_y, 32)
    validation_generator = SequenceData(val_x, val_y, 32)

    # tr.train_network(train_generator, validation_generator, epoch=100)
    # tr.load_network_then_train(train_generator, validation_generator, epoch=50,
    #                            input_name='best_val_67.1054.h5', output_name='second_weights.hdf5')

