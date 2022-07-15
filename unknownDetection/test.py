# import the necessary packages
import metrics
import config
import utils
import numpy as np
import json
from tensorflow.keras.models import load_model

# disable GPU devices
# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# load MNIST dataset and scale the pixel values to the range of [0, 1]
print("[INFO] loading VPN dataset...")
path = '/home/cnic-zyd/Luyang/code/'
file = 'vpn_dataset.txt'

flows, labels = utils.read_data_from_disk(path, file)

replace_a = ['email2a', 'email2b']
replace_b = ['voipbuster1a', 'voipbuster1b']
labels = ['email' if item in replace_a else item for item in labels]
labels = ['voipbuster' if item in replace_b else item for item in labels]

# transfer the textual into numerical labels
unique_labels = set(labels)
print(unique_labels)

unknown_categories = ['email', 'voipbuster']
(known_class, known_label), (unknown_class, unknown_label) = utils.split_unknown(flows, labels, unknown_categories)
print('[INFO...] known class shape')
print(known_class.shape, known_label.shape)
print('[INFO...] unknown class shape')
print(unknown_class.shape, unknown_label.shape)

# 只构造已知流的 文字---标号转换
# training_labels = set(known_label)
#
# label_mapping = {}
# for index, value in enumerate(training_labels):
#     label_mapping[value] = index
# label_reverse = {}
# for index, value in enumerate(training_labels):
#     label_reverse[index] = value

with open('sample_record.txt', 'r') as f:
    record_map = json.loads(f.readline())
    name2id = json.loads(f.readline())
    id2name = json.loads(f.readline())

numerical_labels = []
for lab in known_label:
    numerical_labels.append(name2id[lab])

identical_labels = set(numerical_labels)

# load the model
print("[INFO] loading siamese model...")
print("[INFO] model path:" + config.MODEL_PATH)
model = load_model(config.MODEL_PATH, compile=False)


def construct_samples(known_class, record_map, N=1):
    """
    return the known samples for computing the distance between two vectors
    :return: a numpy array of the known samples
    """
    categories = {}
    for k in record_map.keys():
        flag = min(len(record_map[k]), N)
        its = [known_class[i] for i in record_map[k][:flag]]
        categories[k] = np.array(its)

    return categories


def known_detection(data, labels, categories, threshold, model, mapping, N):
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
    for key in categories.keys():
        coll = categories[key]
        # coll = list(map(lambda x: np.expand_dims(x, axis=0), coll))
        coll = list(map(lambda x: np.expand_dims(x, axis=-1), coll))
        coll = list(map(lambda x: x / 255.0, coll))
        comp_samples[key] = np.array(coll)
    # for known class
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
            # ensure all the comparative samples which are less than N
            val = np.mean(model.predict([comp[:min(len(coll), N)], coll]))
            total_category_mean_pred[key] = val

        if min(total_category_mean_pred.values()) > threshold:
            KU += 1
        else:
            min_dist_key = min(total_category_mean_pred, key=lambda x: total_category_mean_pred[x])
            if mapping[min_dist_key] == tested_label:
                KP += 1
            else:
                KN += 1

    return (KP, KN, KU)


def unknown_detection(data, labels, categories, threshold, model, mapping, N):
    UP = 0
    UN = 0
    # normalization and expand_dims
    comp_samples = {}
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
            val = np.mean(model.predict([comp[:min(len(coll), N)], coll]))
            total_category_mean_pred[key] = val

        if min(total_category_mean_pred.values()) > threshold:
            UP += 1
        else:
            UN += 1

    return (UP, UN)


print(len(known_class), len(known_label))
print(known_class.shape, known_label.shape)
print(len(unknown_class), len(unknown_label))
print(unknown_class.shape, unknown_label.shape)

# use the best threshold to compute
# threshold_coll = [0.04, 0.081, 0.12, 0.16, 0.2, 0.4, 0.8, 1]
# threshold_coll = [0.352, 0.452, 0.552, 0.652, 0.752, 0.852]
# threshold_coll = [0.3, 0.5, 0.8, 1, 1.25, 1.5, 1.75, 1.9]
threshold_coll = [0.08, 0.1, 0.15, 0.2, 0.25]
sampling_num_coll = [1, 10, 20, 50]
neighbors = 10

for threshold in threshold_coll:
    categories = construct_samples(known_class, record_map, N=neighbors)

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
