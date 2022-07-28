import numpy as np
from sklearn.utils import shuffle
import json
from collections import Counter
import config


def read_data_from_disk(path, file):
    data = []
    label = []

    with open(path + file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            payloads = sample[:32 * 32]
            data.append(np.array(payloads).reshape(32, 32))
            label.append(sample[-1])

    data = np.array(data)
    label = np.array(label)
    data, label = shuffle(data, label, random_state=7)
    return (data, label)

def handling_vpndata_labels(labels):
    last2spots = ['1a', '1b', '_A', '_B', '2a', '2b']
    last1spot = ['1', '2']
    exact_label = []

    for label in labels:
        flag = label[-2:]
        if flag in last2spots:
            label = label[:len(label) - 2]
            exact_label.append(label)
        elif flag[-1:] in last1spot:
            label = label[:len(label) - 1]
            exact_label.append(label)
        else:
            exact_label.append(label)

    return np.array(exact_label)


def get_label_statistics(labels):
    # sorted function returns a sorted LIST
    dictionary = Counter(labels)
    d_s = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    return d_s



def load_data():
    (vpn_data, vpn_labels) = read_data_from_disk(config.VPN_DATA_PATH, config.VPN_DATA_FILE)
    vpn_labels = handling_vpndata_labels(vpn_labels)

    return (vpn_data, vpn_labels)
