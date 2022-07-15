import json
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import Counter
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


def split_train_val_test(dataset, label, train_ratio=0.9, val_ratio=0.05):
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


def make_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels))


def make_pairs_balance(images, labels, total_labels, maximum=50):
    pairImages = []
    pairLabels = []
    # numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, total_labels)]
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(idx[label], min(len(idx[label]), maximum))
        for pos in idxB:
            posImage = images[pos]
            # prepare a positive pair and update the images and labels
            # lists, respectively
            pairImages.append([currentImage, posImage])
            pairLabels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        neg_images_index = np.random.choice(negIdx, min(len(negIdx), maximum))
        for neg in neg_images_index:
            negImage = images[neg]
            # prepare a negative pair of images and update our lists
            pairImages.append([currentImage, negImage])
            pairLabels.append([0])
    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels))


def make_pairs_balance_record(images, labels, total_labels, maximum=50):
    pairImages = []
    pairLabels = []
    record_map = {}
    for lb in range(0, total_labels):
        record_map[lb] = [0]

    # numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, total_labels)]
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(idx[label], min(len(idx[label]), maximum))
        # 添加训练记录样本
        record_map[label].extend(idxB)

        for pos in idxB:
            posImage = images[pos]
            # prepare a positive pair and update the images and labels
            # lists, respectively
            pairImages.append([currentImage, posImage])
            pairLabels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        neg_images_index = np.random.choice(negIdx, min(len(negIdx), maximum))
        for neg in neg_images_index:
            negImage = images[neg]
            # prepare a negative pair of images and update our lists
            pairImages.append([currentImage, negImage])
            pairLabels.append([0])
    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels), record_map)


def giant_make_pairs(images, labels, current_index, maximum=20):
    label_counter = Counter(list(labels))
    num_categories = len(np.unique(labels))
    pairImages = []
    pairLabels = []
    idx = np.where(labels == current_index)[0]
    negIdx = np.where(labels != current_index)[0]

    for i in range(len(idx)):
        currentImage = images[idx[i]]
        # in case some categories are less than maximum
        option = np.random.choice(idx, max(len(idx), maximum))
        for j in option:
            posImage = images[j]
            pairImages.append([currentImage, posImage])
            pairLabels.append([1])

    for i in range(len(idx)):
        currentImage = images[idx[i]]
        option = np.random.choice(negIdx, max(len(negIdx), maximum * (num_categories - 1)))
        for neg in option:
            negImage = images[neg]
            pairImages.append([currentImage, negImage])
            pairLabels.append([0])

    return (np.array(pairImages), np.array(pairLabels))


def generate_triplets_with_records(images, labels, total_labels_num, maximum=50):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    triple_Images = []
    triple_labels = []
    record_map = {}

    for lb in range(0, total_labels_num):
        record_map[lb] = [0]

    # numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, total_labels_num)]
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(idx[label], min(len(idx[label]), maximum))
        record_map[label].extend(idxB)

        negIdx = np.where(labels != label)[0]
        neg_images_idx = np.random.choice(negIdx, min(len(negIdx), maximum))

        flag = min(len(idxB), len(neg_images_idx))

        for pos in range(flag):
            posImage = images[idxB[pos]]
            negImage = images[neg_images_idx[pos]]

            triple_Images.append([currentImage, posImage, negImage])
            triple_labels.append([1, 1, 0])

        # prepare a negative pair of images and update our lists
    # return a 2-tuple of our image pairs and labels
    return (np.array(triple_Images), np.array(triple_labels), record_map)


def distance_pairs(vectors):
    (anchor, pos, neg) = vectors
    # print(anchor.shape, pos.shape, neg.shape)

    d_ap = tf.math.reduce_mean(tf.math.square(anchor - pos))
    d_an = tf.math.reduce_mean(tf.math.square(anchor - neg))

    # d_ap = tf.math.sqrt(tf.math.maximum(d_ap, K.epsilon()))
    # d_an = tf.math.sqrt(tf.math.maximum(d_an, K.epsilon()))
    # print(d_ap.shape, d_an.shape)

    # sum_ap = K.sum(K.square(anchor - pos), axis=1,
    #                keepdims=True)
    # d_ap = K.sqrt(K.maximum(sum_ap, K.epsilon()))
    #
    # sum_an = K.sum(K.square(anchor - neg), axis=1,
    #                keepdims=True)
    # d_an = K.sqrt(K.maximum(sum_an, K.epsilon()))

    return (d_ap, d_an)


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                       keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def euclidean_distance_test(vectorA, vectorB):
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(vectorA - vectorB), axis=0,
                       keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


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
