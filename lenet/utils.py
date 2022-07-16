import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
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

    return (known_class, known_label), (unknown_class, unknown_label)


def split_train_val_test(dataset, label, train_ratio=0.9, val_ratio=0.05):
    total_samples = len(dataset)
    train_len = int(total_samples * train_ratio)
    val_len = int(total_samples * val_ratio)

    # generate the train dataset
    train = dataset[0:train_len, :]
    train_label = label[0:train_len]
    # generate the val dataset
    val = dataset[train_len:(train_len + val_len), :]
    val_label = label[train_len:(train_len + val_len)]
    # generate the test dataset
    test = dataset[(train_len + val_len):, :]
    test_label = label[(train_len + val_len):]
    return (train, train_label), (val, val_label), (test, test_label)


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
