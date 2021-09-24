import numpy as np
from keras.models import load_model
import cv2
import tiny_yolov1_model
from yolo_loss import yolo_loss
from keras.utils import Sequence
import math
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_addons as tfa


class SequenceData(Sequence):

    def __init__(self, data_x, data_y, batch_size):
        self.batch_size = batch_size
        self.data_x = data_x
        self.data_y = data_y
        self.indexes = np.arange(len(self.data_x))

    def __len__(self):
        return math.floor(len(self.data_x) / float(self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, idx):

        batch_index = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.data_x[k] for k in batch_index]
        batch_y = [self.data_y[k] for k in batch_index]

        x = np.zeros((len(batch_x), 448, 448, 3))
        y = np.zeros((len(batch_y), 7, 7, 25))

        for i in range(self.batch_size):

            img = cv2.imread(batch_x[i])
            obj_all = batch_y[i].strip().split()
            size = img.shape

            img1 = img / 255
            resize_img = cv2.resize(img1, (448, 448), interpolation=cv2.INTER_AREA)
            x[i, :, :, :] = resize_img

            for j in range(len(obj_all)):
                obj = obj_all[j].split(',')
                x1, y1, x2, y2 = [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])]
                category = int(obj[4])

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                grid_x = int(7 * center_x / size[1])
                grid_y = int(7 * center_y / size[0])

                center_x_ratio = center_x * 7 / size[1] - grid_x
                center_y_ratio = center_y * 7 / size[0] - grid_y
                w_ratio = w / size[1]
                h_ratio = h / size[0]

                y[i, grid_y, grid_x, category] = 1
                y[i, grid_y, grid_x, 20] = 1
                y[i, grid_y, grid_x, 21:25] = np.array([center_x_ratio, center_y_ratio, w_ratio, h_ratio])

        return x, y


def train_network(train_generator, validation_generator, epoch):

    model = tiny_yolov1_model.create_network()
    model.load_weights('raw_weights.hdf5', by_name=True, skip_mismatch=True)

    for i in range(31):
        model.layers[i].trainable = False
        print(model.layers[i])

    model.summary()
    adam = Adam(lr=1e-4, amsgrad=True)

    model.compile(loss=yolo_loss, optimizer=adam)

    log_dir = "Logs/"
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}_{val_loss:.4f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights('first_weights.hdf5')


# Load the partially trained model and continue training and save
def load_network_then_train(train_generator, validation_generator, epoch, input_name, output_name):

    model = tiny_yolov1_model.create_network()
    model.load_weights(input_name)

    for i in range(31):
        model.layers[i].trainable = False
        print(model.layers[i])

    model.summary()

    sgd = optimizers.SGD(lr=1e-5, momentum=0.9)

    model.compile(loss=yolo_loss, optimizer=sgd)
    log_dir = "Logs/"
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}_{val_loss:.4f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights(output_name)
