# import the necessary packages
import metrics
import config
import utils
import numpy as np
import json
from tensorflow.keras.models import load_model

# disable GPU devices
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("[INFO] loading VPN dataset...")
path = '/home/cnic-zyd/Luyang/code/'
file = 'vpn_dataset_50.txt'

flows, labels = utils.read_data_from_disk(path, file)
# 使用非负值进行训练
flows = np.abs(flows)
labels = utils.handling_vpndata_labels(labels)
print(set(labels))
print('LEN:', len(set(labels)))
# 映射为大类
category_mapping = {'vpn_skype_audio': 'VoIP', 'vpn_youtube': 'streaming', 'vpn_spotify': 'streaming',
                    'vpn_sftp': 'file_transfer', 'vpn_icq_chat': 'chat', 'vpn_skype_files': 'file_transfer',
                    'vpn_hangouts_chat': 'chat', 'vpn_netflix': 'streaming', 'vpn_facebook_chat': 'chat',
                    'vpn_facebook_audio': 'VoIP', 'vpn_bittorrent': 'P2P', 'vpn_voipbuster': 'VoIP',
                    'vpn_vimeo': 'streaming', 'vpn_hangouts_audio': 'VoIP', 'vpn_aim_chat': 'chat',
                    'vpn_ftps': 'file_transfer', 'vpn_email': 'email', 'vpn_skype_chat': 'chat'}

print('len:', len(category_mapping))
labels = [category_mapping[item] for item in labels]

# transfer the textual into numerical labels
unique_labels = set(labels)
print(unique_labels)

test_distance_file = config.LABEL + '_test_distance.txt'
name = config.LABEL if config.LABEL != 'file' else config.LABEL + '_transfer'
unknown_categories = [name]
(known_class, known_label), (unknown_class, unknown_label) = utils.split_unknown(flows, labels, unknown_categories)
print('[INFO...] known class shape')
print(known_class.shape, known_label.shape)
print('[INFO...] unknown class shape')
print(unknown_class.shape, unknown_label.shape)

with open(config.LABEL + '_neighbors.txt', 'r') as f:
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
    known_distance = []
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
            tmp_val = model.predict([comp[:min(len(coll), N)], coll])
            val = np.mean(tmp_val)
            total_category_mean_pred[key] = val
            for i in tmp_val:
                known_distance.append(float(i))

        if min(total_category_mean_pred.values()) > threshold:
            KU += 1
        else:
            min_dist_key = min(total_category_mean_pred, key=lambda x: total_category_mean_pred[x])
            # if mapping[min_dist_key] == tested_label:
            if int(min_dist_key) == int(tested_label):
                KP += 1
            else:
                KN += 1
    with open(test_distance_file, 'a') as f:
        f.write(json.dumps(known_distance))
        f.write('\n')
    return (KP, KN, KU)


def unknown_detection(data, labels, categories, threshold, model, mapping, N):
    UP = 0
    UN = 0
    unknown_distance = []
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
            tmp_val = model.predict([comp[:min(len(coll), N)], coll])
            val = np.mean(tmp_val)
            total_category_mean_pred[key] = val
            for i in tmp_val:
                unknown_distance.append(float(i))

        if min(total_category_mean_pred.values()) > threshold:
            UP += 1
        else:
            UN += 1
    with open(test_distance_file, 'a') as f:
        f.write(json.dumps(unknown_distance))
        f.write('\n')
    return (UP, UN)


print(len(known_class), len(known_label))
print(known_class.shape, known_label.shape)
print(len(unknown_class), len(unknown_label))
print(unknown_class.shape, unknown_label.shape)

# use the best threshold to compute
threshold_coll = [0.1]
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
