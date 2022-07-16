import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import MaxPooling1D


def Lenet(inputShape, embeddingDim=64):
    """
    :param inputShape: 6 * 20
    :param embeddingDim: 48
    :return: model
    """
    inputs = Input(inputShape)
    x = Conv2D(6, (5, 5), padding='valid', strides=(1, 1), activation='relu', kernel_initializer='uniform')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(16, (5, 5), padding='valid', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(embeddingDim, activation='relu')(x)

    tag_name = 'encoder'
    model = Model(inputs, outputs, name=tag_name)

    return model


def vgg_seen(inputShape, embeddingDim=1000):
    """
    :param inputShape: 6 * 20
    :param embeddingDim: 48
    :return: model
    """
    inputs = Input(inputShape)
    x = Conv1D(64, 3, padding='same', strides=1, activation='relu')(inputs)
    x = Conv1D(64, 3, padding='same', strides=1, activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    x = Conv1D(128, 3, padding='same', strides=1, activation='relu')(x)
    x = Conv1D(128, 3, padding='same', strides=1, activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    x = Conv1D(256, 3, padding='same', strides=1, activation='relu')(x)
    x = Conv1D(256, 3, padding='same', strides=1, activation='relu')(x)
    x = Conv1D(256, 3, padding='same', strides=1, activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    x = Conv1D(512, 3, padding='same', strides=1, activation='relu')(x)
    x = Conv1D(512, 3, padding='same', strides=1, activation='relu')(x)
    x = Conv1D(512, 3, padding='same', strides=1, activation='relu')(x)

    x = Flatten()(x)
    x = Dense(4096)(x)
    x = Dense(4096)(x)
    outputs = Dense(embeddingDim)(x)
    model = Model(inputs, outputs)

    return model
