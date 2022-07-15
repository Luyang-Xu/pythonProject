import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization


def vgg16_model(inputShape, embeddingDim=48):
    inputs = Input(inputShape)
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    print(x.shape)
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    print(x.shape)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    print(x.shape)
    x = Dropout(0.3)(x)
    print(x.shape)
    x = Conv2D(128, (2, 2), padding="same", activation="relu")(x)
    print(x.shape)
    x = Conv2D(128, (2, 2), padding="same", activation="relu")(x)
    print(x.shape)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    print(x.shape)
    x = Dropout(0.3)(x)
    print(x.shape)
    x = Flatten()(x)
    print(x.shape)
    x = Dense(100)(x)
    print(x.shape)
    outputs = Dense(embeddingDim)(x)
    print(outputs.shape)

    model = Model(inputs, outputs)
    return model


def Lenet(inputShape, embeddingDim=48):
    """
    :param inputShape: 6 * 20
    :param embeddingDim: 48
    :return: model
    """
    # (None, 6, 20, 6)
    # (None, 6, 20, 6)
    # (None, 3, 10, 6)
    # (None, 3, 10, 16)
    # (None, 3, 10, 16)
    # (None, 1, 5, 16)
    inputs = Input(inputShape)
    x = Conv2D(6, (5, 5), padding='same', strides=(1, 1), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(16, (5, 5), padding='same', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(84, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(embeddingDim, activation='relu')(x)

    tag_name = 'encoder'
    model = Model(inputs, outputs, name=tag_name)

    return model





