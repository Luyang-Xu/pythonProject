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
