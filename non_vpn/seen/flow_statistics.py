import os
import json
import numpy as np
import matplotlib.pyplot as plt


def get_single_dir_files(dir_path):
    # root_coll: 每个子文件夹的目录
    # files_coll: 每个子文件夹下的具体文件的集合
    root_coll = []
    files_coll = []

    for root, dirs, files in os.walk(dir_path):
        root_coll.append(root)
        files_coll.append(files)

    return root_coll, files_coll


def packet_counts(path, file):
    whole_packets = []
    # 1. extract label
    characters = file.split('_')
    characters = characters[:len(characters) - 2]
    label = '_'.join(characters)
    print(label)
    # 2. extract features
    with open(path + file, 'r') as f:
        for line in f:
            data = json.loads(line)
            frame_len = data[0]
            # payload_len = data[1]
            # payload = data[2]
            whole_packets.append(len(frame_len))

    return whole_packets


def payloads_count(path, file):
    whole_payload_len = []
    characters = file.split('_')
    characters = characters[:len(characters) - 2]
    label = '_'.join(characters)
    print(label)
    with open(path + file, 'r') as f:
        for line in f:
            data = json.loads(line)
            # frame_len = data[0]
            # payload_len = data[1]
            payload = data[2]
            counter = 0
            for pckt in payload:
                counter += len(pckt)
            whole_payload_len.append(counter)

    return whole_payload_len


def write_data_to_file(whole_data_list, des_path, des_file):
    with open(des_path + des_file, 'a') as f:
        for sample in whole_data_list:
            f.write(json.dumps(sample))
            f.write('\n')


def single_plot(packet_len, step, des_name):
    bins = np.arange(min(packet_len), max(packet_len), step)
    plt.hist(packet_len, bins=bins, histtype='bar', alpha=0.5, edgecolor='k', color='steelblue')
    plt.title('the distribution of flow ' + des_name)
    plt.xlabel('packet num for each flow')
    plt.ylabel('counts')
    plt.savefig(des_name + '.pdf')


feature_path = '/home/cnic-zyd/Luyang/vpn_features/'
des_path = '/home/cnic-zyd/Luyang/code/'

root, files = get_single_dir_files(feature_path)
total_records_num = []
total_records_payload = []

for file in files[0]:
    whole_num = packet_counts(feature_path, file)
    total_records_num.extend(whole_num)

    whole_payload = payloads_count(feature_path, file)
    total_records_payload.extend(whole_payload)

print(len(total_records_num))

print(len(total_records_payload))

single_plot(total_records_num, 1, 'packet num')

single_plot(total_records_payload, 784, 'payload length')
