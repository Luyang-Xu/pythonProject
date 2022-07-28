# import the necessary packages
import tensorflow.keras.backend as K
import tensorflow as tf
import config
import numpy as np


# https://blog.csdn.net/autocyz/article/details/53149760
def triplet_loss(ap, an, margin=config.MARGIN):
    ap = tf.cast(ap, tf.float32)
    an = tf.cast(an, tf.float32)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    loss = K.maximum(0.0, margin + ap - an)
    return loss


def tripletLoss(y_true, y_pred, margin=config.MARGIN, embeddingDim=config.Embedding_Dim):
    anchor, positive, negative = y_pred[:, :embeddingDim], y_pred[:, embeddingDim:2 * embeddingDim], y_pred[:,
                                                                                                     2 * embeddingDim:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    loss = tf.math.log1p(tf.math.exp(positive_dist - negative_dist + margin))
    # loss = tf.maximum(positive_dist - negative_dist + margin, 0.)
    return loss


def lossless_triplet_loss(y_true, y_pred, N=config.Embedding_Dim, beta=config.Embedding_Dim, epsilon=K.epsilon()):
    anchor, positive, negative = y_pred[:, :N], y_pred[:, N:2 * N], y_pred[:, 2 * N:]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=1)

    # non linear values
    # -ln(-x/N + 1)
    pos_dist = -tf.math.log(-tf.math.divide((pos_dist), beta) + 1 + epsilon)
    neg_dist = -tf.math.log(-tf.math.divide((N - neg_dist), beta) + 1 + epsilon)

    loss = pos_dist + neg_dist

    return loss


def improved_triplet_loss(y_true, y_pred, N=config.Embedding_Dim, inte_margin=config.MARGIN, intra_margin=0.01,
                          beta=0.01):
    anchor, positive, negative = y_pred[:, :N], y_pred[:, N:2 * N], y_pred[:, 2 * N:]
    pos_dist = tf.square(tf.subtract(anchor, positive))
    neg_dist = tf.square(tf.subtract(anchor, negative))
    # pos_dist = tf.reduce_mean(tf.square(tf.subtract(anchor, positive)), axis=1)
    # neg_dist = tf.reduce_mean(tf.square(tf.subtract(anchor, negative)), axis=1)

    loss1 = tf.math.log1p(tf.math.exp(pos_dist - neg_dist + inte_margin))
    loss2 = tf.math.log1p(tf.math.exp(pos_dist - intra_margin))
    # loss2 = tf.math.maximum(pos_dist, intra_margin)
    loss = loss1 + beta * loss2
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
