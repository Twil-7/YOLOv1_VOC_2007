import numpy as np
from keras.models import Sequential, Model, Input
from keras.layers import Dense, LeakyReLU, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import layers
from keras.engine.topology import Layer
import keras.backend as k
from keras.regularizers import l2


class my_reshape(Layer):
    def __init__(self, target_shape, **kwargs):
        super(my_reshape, self).__init__(**kwargs)
        self.target_shape = target_shape    # (7, 7, 30)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], ) + self.target_shape    # (batch, 7, 7, 30)

    def call(self, inputs, **kwargs):
        s = [self.target_shape[0], self.target_shape[1]]    # [7, 7]
        c = 20
        b = 2

        idx1 = s[0] * s[1] * c    # 7 * 7 * 20 = 980
        idx2 = idx1 + s[0] * s[1] * b    # 980 + 7 * 7 * 2 = 1078

        class_tensor = k.reshape(inputs[:, :idx1], (k.shape(inputs)[0],) + tuple([s[0], s[1], c]))
        class_tensor = k.softmax(class_tensor)    # shape = (batch, 7, 7, 20)

        confidence_tensor = k.reshape(inputs[:, idx1:idx2], (k.shape(inputs)[0],) + tuple([s[0], s[1], b]))
        confidence_tensor = k.sigmoid(confidence_tensor)    # shape = (batch, 7, 7, 2)

        location_tensor = k.reshape(inputs[:, idx2:], (k.shape(inputs)[0],) + tuple([s[0], s[1], b * 4]))
        location_tensor = k.sigmoid(location_tensor)    # shape = (batch, 7, 7, 8)

        outputs = k.concatenate([class_tensor, confidence_tensor, location_tensor])    # shape = (batch, 7, 7, 30)

        return outputs


def create_network():

    inputs = Input((448, 448, 3))

    x = Conv2D(16, (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4))(inputs)
    x = BatchNormalization(name='bnconvolutional_0')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), padding='same', name='convolutional_1', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same', name='convolutional_2', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same', name='convolutional_3', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_3')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), padding='same', name='convolutional_4', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_4')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(512, (3, 3), padding='same', name='convolutional_5', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(1024, (3, 3), padding='same', name='convolutional_6', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, (3, 3), padding='same', name='convolutional_7', use_bias=False,
               kernel_regularizer=l2(5e-3))(x)
    x = BatchNormalization(name='bnconvolutional_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)

    x = Dense(1470, activation='relu', name='connected_0')(x)
    x = Dropout(0.5)(x)

    x = Dense(1470, activation='linear', name='connected_1')(x)
    outputs = my_reshape((7, 7, 30))(x)
    model = Model(inputs=inputs, outputs=outputs)

    return model



