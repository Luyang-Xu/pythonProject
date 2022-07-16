# import the necessary packages
import metrics
import config
import utils
import json
import numpy as np
import data_generator
from tensorflow.keras.models import load_model
import time

# disable GPU devices
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load data
(flows, labels) = data_generator.load_data()
# 使用非负值进行训练
unique_labels = set(labels)
print(unique_labels)

(known_class, known_label), (unknown_class, unknown_label) = utils.split_unknown(flows, labels,
                                                                                 config.UNKNOWN_CATEGORIES)
print('[INFO...] known class shape')
print(known_class.shape, known_label.shape)
print('[INFO...] unknown class shape')
print(unknown_class.shape, unknown_label.shape)

with open('service_id_mapping.txt', 'r') as f:
    name2id = json.loads(f.readline())
    id2name = json.loads(f.readline())

numerical_labels = []
for lab in known_label:
    numerical_labels.append(name2id[lab])
identical_labels = set(numerical_labels)

# load the model
print("[INFO] loading siamese model...")
print("[INFO] model path:" + config.ENCODER_PATH)
model = load_model(config.ENCODER_PATH, compile=False)

test_distance_file = 'test_distance.txt'


def get_neighbors_for_known_classes(known_class, known_label, N):
    categories = {}

    class_num = len(np.unique(known_label))

    idx = [np.where(known_label == i)[0] for i in range(0, class_num)]
    for i in range(class_num):
        neighbors_index = np.random.choice(idx[i], min(len(idx[i]), N))
        categories[i] = np.array([known_class[i] for i in neighbors_index])

    return categories


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
    comp_samples = {}
    known_distance = []

    for key in categories.keys():
        coll = categories[key]
        coll = list(map(lambda x: np.expand_dims(x, axis=-1), coll))
        coll = list(map(lambda x: x / 255.0, coll))
        comp_samples[key] = np.array(coll)

    for i in range(len(data)):
        tested = data[i]
        tested_label = labels[i]

        # tested = np.expand_dims(tested, axis=0)
        tested = np.expand_dims(tested, axis=-1)
        tested = tested / 255.0
        comp = np.array([tested for i in range(N)])
        total_category_mean_pred = {}

        for key in comp_samples.keys():
            coll = comp_samples[key]
            original_sample = encoder(comp[:min(len(coll), N)])
            comparative_neighbors = encoder(coll)

            assert len(original_sample) == len(comparative_neighbors)
            val = []
            for i in range(len(original_sample)):
                val.append(utils.euclidean_distance_test(original_sample[i], comparative_neighbors[i]))
            total_category_mean_pred[key] = np.mean(val)

        if min(total_category_mean_pred.values()) > threshold:
            KU += 1
        else:
            min_dist_key = min(total_category_mean_pred, key=lambda x: total_category_mean_pred[x])
            if int(min_dist_key) == tested_label:
                KP += 1
            else:
                KN += 1

    return (KP, KN, KU)


def unknown_detection(data, labels, categories, threshold, encoder, mapping, N):
    UP = 0
    UN = 0
    # normalization and expand_dims
    comp_samples = {}
    unknown_distance = []

    for key in categories.keys():
        coll = categories[key]
        # coll = list(map(lambda x: np.expand_dims(x, axis=0), coll))
        coll = list(map(lambda x: np.expand_dims(x, axis=-1), coll))
        coll = list(map(lambda x: x / 255.0, coll))
        comp_samples[key] = np.array(coll)

    for i in range(len(data)):
        tested = data[i]
        tested_label = labels[i]

        # tested = np.expand_dims(tested, axis=0)
        tested = np.expand_dims(tested, axis=-1)
        tested = tested / 255.0
        comp = np.array([tested for i in range(N)])

        total_category_mean_pred = {}

        for key in comp_samples:
            coll = comp_samples[key]
            original_sample = encoder(comp[:min(len(coll), N)])
            comparative_neighbors = encoder(coll)

            assert len(original_sample) == len(comparative_neighbors)
            val = []
            for i in range(len(original_sample)):
                val.append(utils.euclidean_distance_test(original_sample[i], comparative_neighbors[i]))
            total_category_mean_pred[key] = np.mean(val)

        if min(total_category_mean_pred.values()) > threshold:
            UP += 1
        else:
            UN += 1

    return (UP, UN)


print(len(known_class), len(known_label))
print(known_class.shape, known_label.shape)
print(len(unknown_class), len(unknown_label))
print(unknown_class.shape, unknown_label.shape)

threshold_coll = np.linspace(0.1, 1.1, 11)
sampling_num_coll = [10]
neighbors = 10

start_time = time.time()
for threshold in threshold_coll:
    categories = get_neighbors_for_known_classes(known_class, np.array(numerical_labels), N=neighbors)
    (KP, KN, KU) = known_detection(known_class, np.array(numerical_labels), categories, threshold, model,
                                   name2id, N=neighbors)
    (UP, UN) = unknown_detection(unknown_class, unknown_label, categories, threshold, model, name2id, N=neighbors)

    (pr, acc, fdr, tdr) = metrics.cal_four_metrics(KP, KN, KU, UP, UN)

    print('[Hyper-parameters]... neighbors:' + str(neighbors) + '; threshold:' + str(threshold))
    print('[Purity Rate]...', str(pr * 100) + '%')
    print('[Accuracy]...', str(acc * 100) + '%')
    print('[False Detection Rate]...', str(fdr * 100) + '%')
    print('[True Detection Rate]...', str(tdr * 100) + '%')
    print('KP, KN, KU, UP, UN:' + str(KP) + ', ' + str(KN) + ', ' + str(KU) + ', ' + str(UP) + ', ' + str(UN))
    print('*' * 30)
end_time = time.time()
print('[INFO...seen running time cost (sec):]', (end_time - start_time) / 60)
