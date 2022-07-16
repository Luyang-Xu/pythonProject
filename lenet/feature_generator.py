import os
import json


def get_single_dir_files(dir_path):
    # root_coll: 每个子文件夹的目录
    # files_coll: 每个子文件夹下的具体文件的集合
    root_coll = []
    files_coll = []

    for root, dirs, files in os.walk(dir_path):
        root_coll.append(root)
        files_coll.append(files)

    return root_coll, files_coll


def generate_labels(path, file, N_packets):
    whole_data = []
    # 1. extract label
    characters = file.split('_')
    characters = characters[:len(characters) - 2]
    label = '_'.join(characters)
    # 2. extract features
    with open(path + file, 'r') as f:
        for line in f:
            sample = []
            data = json.loads(line)
            frame_len = data[0]
            payload_len = data[1]
            payload = data[2]

            # 倒叙删除为空的字节位置
            delete_index = []
            for i in range(len(payload)):
                if payload[i] == '':
                    delete_index.append(i)
            delete_index.reverse()
            for index in delete_index:
                frame_len.pop(index)
                payload_len.pop(index)
                payload.pop(index)

            packet_count = min(len(payload), N_packets)

            for i in range(packet_count):
                sample.append(str(frame_len[i]))
            for i in range(packet_count):
                sample.append(str(payload_len[i]))
            for i in range(packet_count):
                sample.extend(payload[i])

            # 截断至784
            flag_byte = 32 * 32
            if len(sample) < flag_byte:
                padding_len = flag_byte - len(sample)
                sample.extend([0] * padding_len)
            else:
                sample = sample[:flag_byte]

            # 处理特殊值
            replace = ['', "", 0]
            sample = ['00' if i in replace else i for i in sample]
            sample = list(map(lambda x: int(x, 16), sample))

            sample.append(label)
            whole_data.append(sample)

    return whole_data


def write_data_to_file(whole_data_list, des_path, des_file):
    with open(des_path + des_file, 'a') as f:
        for sample in whole_data_list:
            f.write(json.dumps(sample))
            f.write('\n')


feature_path = '/home/cnic-zyd/Luyang/vpn_features/'
des_path = '/home/cnic-zyd/lenet/dataset/'
des_file = 'vpn_dataset_32_32.txt'

root, files = get_single_dir_files(feature_path)
for file in files[0]:
    whole_data = generate_labels(feature_path, file, N_packets=3)
    print('SUCCESS:' + file)
    write_data_to_file(whole_data, des_path, des_file)
