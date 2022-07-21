# import the necessary packages
import tensorflow.keras.backend as K
import tensorflow as tf
import config


# https://blog.csdn.net/autocyz/article/details/53149760

def contrastive_loss(y, preds, margin=config.MARGIN):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, preds.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    # return the computed contrastive loss to the calling function
    return loss



def triplet_loss(ap, an, margin=config.MARGIN):
    ap = tf.cast(ap, tf.float32)
    an = tf.cast(an, tf.float32)
    loss = tf.maximum(0.0, margin + ap - an)
    # loss = tf.reduce_mean(loss)
    return loss


def cal_four_metrics(KP, KN, KU, UP, UN):
    pr = mec_purity_rate(KP, KN, KU, UP, UN)

    acc = mec_accuracy(KP, KN, KU)

    fdr = mec_false_detection_rate(KU, KP, KN)

    tdr = mec_true_detection_rate(UP, UN)

    return (pr, acc, fdr, tdr)


def mec_purity_rate(KP, KN, KU, UP, UN):
    val = (KP + UP) / (KP + KN + KU + UP + UN)
    return round(val, 2)


def mec_accuracy(KP, KN, KU):
    val = KP / (KP + KN + KU)
    return round(val, 2)


def mec_false_detection_rate(KU, KP, KN):
    val = KU / (KP + KN + KU)
    return round(val, 2)


def mec_true_detection_rate(UP, UN):
    val = UP / (UP + UN)
    return round(val, 2)


