import numpy as np
import config


def get_centroid(images, labels, total_labels):
    """
    return a map containing centroids of each categories
    :param images: data
    :param labels: label
    :param total_labels: the number of categories
    :return: a map
    """
    center_map = {}
    for lb in range(0, total_labels):
        center_map[lb] = []
    # the sample index for each category
    idx = [np.where(labels == i)[0] for i in range(0, total_labels)]
    for key in center_map.keys():
        index_list = idx[key]
        center = cal_mean(images, index_list)
        center_map[key] = center
    return center_map


def cal_mean(images, data_index):
    data = []
    for i in data_index:
        data.append(images[i])
    return np.mean(np.array(data), axis=0)


def cal_euclidean_distance(arrA, arrB):
    dist = np.sqrt(np.sum(np.square(arrA - arrB)))
    # return np.linalg.norm(arrA - arrB)
    return dist


def get_center_neighbors(images, labels, total_labels, center_map, N):
    """
    :param center_map:
    :return: an index map for each category for the first N nearest neighbors
    """
    # the sample index for each category
    nearest_neighbors = {}

    idx = [np.where(labels == i)[0] for i in range(0, total_labels)]
    for key in center_map.keys():
        mapping = {}
        current_records = idx[key]
        centroid = center_map[key]
        for id in range(len(current_records)):
            dist = cal_euclidean_distance(centroid, images[current_records[id]])
            mapping[current_records[id]] = dist

        nearest_neighbors[key] = mapping

    sorted_neighbor = {}
    for key in nearest_neighbors.keys():
        tmp = sorted(nearest_neighbors[key].items(), key=lambda kv: (kv[1], kv[0]))
        flag = min(len(tmp), N)
        sorted_neighbor[key] = [int(tmp[i][0]) for i in range(flag)]

    # sort nearest_neighbors by the ascending order
    return sorted_neighbor


def generate_near_pairs(images, labels, total_labels, near_neighbors, N):
    pairImages = []
    pairLabels = []

    for i in range(len(images)):
        currentImage = images[i]
        currentLabel = labels[i]
        # get the nearest
        neighbors = near_neighbors[currentLabel]
        flag = min(len(neighbors), N)

        # for positive records
        idp = np.random.choice(neighbors, flag)
        for j in range(len(idp)):
            pairImages.append([currentImage, images[idp[j]]])
            pairLabels.append([1])

        # for negative records
        for key in near_neighbors.keys():
            if key != currentLabel:
                neighbors = near_neighbors[key]
                # F = min(len(neighbors), N)
                # average_num = int(F / total_labels)
                average_num = config.NEG_RANDOM_NUMBERS if len(neighbors) != 0 else 0
                idn = np.random.choice(neighbors, average_num)
                for j in range(len(idn)):
                    pairImages.append([currentImage, images[idn[j]]])
                    pairLabels.append([0])

    return (np.array(pairImages), np.array(pairLabels))


def mining_triplets_distance(arrA, arrB):
    distance = np.sqrt(np.sum(np.square(arrA - arrB)))
    return distance


def generate_triplets(images, labels, total_class_num, near_neighbors, N):
    triple_images = []
    triple_labels = []

    for i in range(len(images)):
        currentImage = images[i]
        currentLabel = labels[i]

        neighbors = near_neighbors[currentLabel]
        flag = min(len(neighbors), N)

        idp = np.random.choice(neighbors, flag)
        for j in range(len(idp)):
            # 对于每一个正样本都构造出四十个负样本
            for key in near_neighbors.keys():
                if key != currentLabel:
                    neg_neighbors = near_neighbors[key]
                    average_num = config.NEG_RANDOM_NUMBERS if len(neg_neighbors) != 0 else 0
                    idn = np.random.choice(neg_neighbors, min(average_num, len(neg_neighbors)))
                    for k in range(len(idn)):
                        triple_images.append([currentImage, images[idp[j]], images[idn[k]]])
                        triple_labels.append([1, 1, 0])

    return (np.array(triple_images), np.array(triple_labels))


def triplets_generator(images, labels, near_neighbors, N, yield_num=100000):
    anchor = []
    pos = []
    neg = []
    triple_labels = []

    for i in range(len(images)):
        currentImage = images[i]
        currentLabel = labels[i]

        pos_neighbors = near_neighbors[currentLabel]
        flag = min(len(pos_neighbors), N)

        idp = np.random.choice(pos_neighbors, flag)
        for j in range(len(idp)):
            # 对于每一个正样本都构造出四十个负样本
            for key in near_neighbors.keys():
                if key != currentLabel:
                    neg_neighbors = near_neighbors[key]
                    average_num = config.NEG_RANDOM_NUMBERS if len(neg_neighbors) != 0 else 0
                    idn = np.random.choice(neg_neighbors, min(average_num, len(neg_neighbors)))
                    for k in range(len(idn)):
                        if mining_triplets_distance(currentImage, images[idp[j]]) < mining_triplets_distance(
                                currentImage, images[idn[k]]):
                            anchor.append(currentImage)
                            pos.append(images[idp[j]])
                            neg.append(images[idn[k]])
                            # triple_images.append([currentImage, images[idp[j]], images[idn[k]]])
                            triple_labels.append([1, 1, 0])
                        if len(anchor) >= yield_num:
                            yield np.array(anchor), np.array(pos), np.array(neg), np.array(triple_labels)
                            anchor = []
                            pos = []
                            neg = []
                            triple_labels = []


def random_generate_triplets(images, labels, total_class_num, N):
    triple_images = []
    triple_labels = []

    idx = [np.where(labels == i)[0] for i in range(0, total_class_num)]

    for epoch in range(N):

        for i in range(len(images)):
            currentImage = images[i]
            currentLabel = labels[i]

            pos_index = idx[currentLabel]

            pos = np.random.choice(pos_index)
            posImage = images[pos]

            negIdx = np.where(labels != currentLabel)[0]
            negImage = images[np.random.choice(negIdx)]

            triple_images.append([currentImage, posImage, negImage])
            triple_labels.append([1, 1, 0])

    return (np.array(triple_images), np.array(triple_labels))


def improved_generate_triplets(images, labels, total_class_num):
    triple_images = []
    triple_labels = []
    average_num = int(config.POS_RANDOM_NUMBERS / total_class_num)
    idx = [np.where(labels == i)[0] for i in range(0, total_class_num)]

    for i in range(len(images)):
        currentImage = images[i]
        currentLabel = labels[i]

        pos_index = idx[currentLabel]

        for key in range(len(idx)):
            if key != currentLabel:
                neg_neighbors = idx[key]
                if len(neg_neighbors) != 0:
                    idn = np.random.choice(neg_neighbors, min(average_num, len(neg_neighbors)))

                    for j in range(len(idn)):
                        pos = np.random.choice(pos_index)
                        posImage = images[pos]
                        triple_images.append([currentImage, posImage, images[idn[j]]])
                        triple_labels.append([1, 1, 0])

    return (np.array(triple_images), np.array(triple_labels))
