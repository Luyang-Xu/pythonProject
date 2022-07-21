from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LayerNormalization
import config


def Lenet(inputShape, embeddingDim=config.Embedding_Dim):
    inputs = Input(inputShape)
    x = Conv2D(6, (5, 5), padding='same', strides=(1, 1), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(16, (5, 5), padding='same', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(embeddingDim, activation=None)(x)
    outputs = LayerNormalization(axis=1)(outputs)

    tag_name = 'encoder'
    model = Model(inputs, outputs, name=tag_name)

    return model
