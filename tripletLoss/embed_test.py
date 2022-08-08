# import the necessary packages
import metrics
import config
import utils
import model
import time
import numpy as np
import data_generator
import sample_select
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# from tensorflow.keras.utils import plot_model
# import tensorflow_addons as tfa

# load MNIST dataset
flows, labels = data_generator.load_data()
flows = np.abs(flows)

unique_labels = set(labels)
print(unique_labels)

(known_class, known_label), (unknown_class, unknown_label) = utils.split_unknown(flows, labels,
                                                                                 config.UNKNOWN_CATEGORIES)
print('[INFO...] known class shape')
print(known_class.shape, known_label.shape)
print('[INFO...] unknown class shape')
print(unknown_class.shape, unknown_label.shape)

with open(config.NEIGHBOR_FILE, 'r') as f:
    record_map = json.loads(f.readline())
    name2id = json.loads(f.readline())
    id2name = json.loads(f.readline())

numerical_labels = []
for lab in known_label:
    numerical_labels.append(name2id[lab])
print(['INFO... Split train and test'])
identical_labels = set(numerical_labels)

# load the model
print("[INFO] loading siamese model...")
print("[INFO] model path:" + config.ENCODER_PATH)
encoder = load_model(config.ENCODER_PATH, compile=False)
neighbors_num = 10


# 使用embedding之后的向量来计算簇中心以及最近的N个邻居

def cal_euclidean_distance(arrA, arrB):
    dist = np.sqrt(np.sum(np.square(arrA - arrB)))
    # return np.linalg.norm(arrA - arrB)
    return dist


def embedding_all(data, labels, encoder, N):
    sorted_neighbors = {}
    total_class_num = len(set(labels))
    idx = [np.where(labels == i)[0] for i in range(0, total_class_num)]

    for i in range(len(idx)):
        # return a index list
        input = []
        for j in idx[i]:
            input.append(data[j])

        input = np.array(input)
        input = np.expand_dims(input, axis=-1)
        embed_input = encoder(input)
        center = np.mean(embed_input, axis=0)

        mapping = {}
        for id in range(len(embed_input)):
            mapping[id] = cal_euclidean_distance(center, embed_input[id, :])

        tmp = sorted(mapping.items(), key=lambda kv: (kv[1], kv[0]))
        flag = min(len(tmp), N)
        sorted_neighbors[i] = [embed_input[int(tmp[i][0]), :] for i in range(flag)]

    return sorted_neighbors


def known_detection(data, labels, categories, threshold, encoder, mapping, N):
    """
    :param data:
    :param labels: numerical_labels
    :param categories:
    :param threshold:
    :param model:
    :param mapping: label_mapping
    :return:
    """
    # first cal the comparative samples
    KP = 0
    KN = 0
    KU = 0

    # for known class
    for i in range(len(data)):
        tested = data[i]
        tested_label = labels[i]

        # tested = np.expand_dims(tested, axis=0)
        tested = np.expand_dims(tested, axis=-1)
        tested = tested / 255.0

        target = np.array([tested for i in range(N)])
        total_category_mean_pred = {}

        for key in categories.keys():
            coll = categories[key]
            # ensure all the comparative samples which are less than
            original_sample = encoder(target[:min(len(coll), N)])
            assert len(original_sample) == len(target)
            val = []
            for i in range(len(original_sample)):
                val.append(cal_euclidean_distance(original_sample[i], coll[i]))
            total_category_mean_pred[key] = np.mean(val)

        # for k,v in total_category_mean_pred.items():
        #     print(k,v)

        if min(total_category_mean_pred.values()) > threshold:
            KU += 1
        else:
            min_dist_key = min(total_category_mean_pred, key=lambda x: total_category_mean_pred[x])
            # if mapping[min_dist_key] == tested_label:
            if int(min_dist_key) == int(tested_label):
                KP += 1
            else:
                KN += 1

    return (KP, KN, KU)


def unknown_detection(data, labels, categories, threshold, encoder, mapping, N):
    UP = 0
    UN = 0
    # normalization and expand_dims

    for i in range(len(data)):
        tested = data[i]
        tested_label = labels[i]

        tested = np.expand_dims(tested, axis=-1)
        tested = tested / 255.0
        target = np.array([tested for i in range(N)])

        total_category_mean_pred = {}

        for key in categories:
            coll = categories[key]
            original_sample = encoder(target[:min(len(coll), N)])
            assert len(original_sample) == len(coll)
            val = []
            for i in range(len(original_sample)):
                val.append(cal_euclidean_distance(original_sample[i], coll[i]))
            total_category_mean_pred[key] = np.mean(val)

        if min(total_category_mean_pred.values()) > threshold:
            UP += 1
        else:
            UN += 1

    return (UP, UN)


threshold = 12
start_time = time.time()
categories = embedding_all(known_class, np.array(numerical_labels), encoder, neighbors_num)

print(known_class.shape, known_label.shape)

(KP, KN, KU) = known_detection(known_class, np.array(numerical_labels), categories, threshold=threshold,
                               encoder=encoder,
                               mapping=name2id,
                               N=neighbors_num)

print(unknown_class.shape, unknown_label.shape)
(UP, UN) = unknown_detection(unknown_class, unknown_label, categories, threshold, encoder, name2id,
                             N=neighbors_num)
end_time = time.time()

(pr, acc, fdr, tdr) = metrics.cal_four_metrics(KP, KN, KU, UP, UN)

neighbors = 10
print('[Hyper-parameters]... neighbors:' + str(neighbors) + '; threshold:' + str(threshold))
print('[Purity Rate]...', str(pr * 100) + '%')
print('[Accuracy]...', str(acc * 100) + '%')
print('[False Detection Rate]...', str(fdr * 100) + '%')
print('[True Detection Rate]...', str(tdr * 100) + '%')
print('KP, KN, KU, UP, UN:' + str(KP) + ', ' + str(KN) + ', ' + str(KU) + ', ' + str(UP) + ', ' + str(UN))
print('*' * 30)
print('[Time cost...]', str((end_time - start_time) / 60), ' minutes')
