import json
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import config
from tensorflow.keras.callbacks import ReduceLROnPlateau


def split_unknown(dataset, labels, unknown_category):
    """
    set services from the parameter as the unknown categories
    :param dataset:
    :param unknown_category: the categories treated as the unknown
    """
    known_class = []
    known_label = []
    unknown_class = []
    unknown_label = []

    total_num = len(dataset)
    for i in range(total_num):
        if labels[i] in unknown_category:
            unknown_class.append(dataset[i])
            unknown_label.append(labels[i])
        else:
            known_class.append(dataset[i])
            known_label.append(labels[i])

    known_class = np.array(known_class)
    known_label = np.array(known_label)
    unknown_class = np.array(unknown_class)
    unknown_label = np.array(unknown_label)

    # known_class, known_label = shuffle(known_class, known_label, random_state=0)
    # unknown_class, unknown_label = shuffle(unknown_class, unknown_label, random_state=0)

    return (known_class, known_label), (unknown_class, unknown_label)


def read_data_from_disk(path, file):
    data = []
    label = []
    with open(path + file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            length = sample[0:40]
            payload = sample[40:120]
            replace = ['', "", 0]
            payload = ['00' if i in replace else i for i in payload]
            payload = list(map(lambda x: int(x, 16), payload))
            length.extend(payload)
            feature = length

            data.append(np.array(feature).reshape(6, 20))
            label.append(sample[-1])
    data = np.array(data)
    label = np.array(label)
    data, label = shuffle(data, label, random_state=0)
    return (data, label)


def split_train_val_test(dataset, label, train_ratio=0.8, val_ratio=0.1):
    total_samples = len(dataset)
    train_len = int(total_samples * train_ratio)
    val_len = int(total_samples * val_ratio)

    # generate the train dataset
    train = dataset[0:train_len, :, :]
    train_label = label[0:train_len]
    # generate the val dataset
    val = dataset[train_len:train_len + val_len, :, :]
    val_label = label[train_len:train_len + val_len]
    # generate the test dataset
    test = dataset[train_len + val_len:, :, :]
    test_label = label[train_len + val_len:]
    return (train, train_label), (val, val_label), (test, test_label)


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                       keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def euclidean_distance_triplet(vectorA, vectorB):
    sumSquared = K.sum(K.square(vectorA - vectorB), axis=0,
                       keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def mining_triplets_distance(arrA, arrB):
    distance = np.sqrt(np.sum(np.square(arrA - arrB)))
    return distance

def distance_pairs(vectors):
    (anchor, pos, neg) = vectors
    sum_ap = K.sum(K.square(anchor - pos), axis=1,
                   keepdims=True)
    d_ap = K.sqrt(K.maximum(sum_ap, K.epsilon()))

    sum_an = K.sum(K.square(anchor - neg), axis=1,
                   keepdims=True)
    d_an = K.sqrt(K.maximum(sum_an, K.epsilon()))
    # add a filter
    return (d_ap, d_an)


    # positive_dist = tf.math.reduce_mean(tf.square(anchor - pos), axis=1, keepdims = True)
    # negative_dist = tf.math.reduce_mean(tf.square(anchor - neg), axis=1, keepdims = True)
    #
    # return (positive_dist, negative_dist)


def contrastive_plot(H, plotPath):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)


def dynamic_LR():
    return ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
