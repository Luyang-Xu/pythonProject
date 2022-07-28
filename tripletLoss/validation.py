# import the necessary packages

import config
import utils
import numpy as np
import data_generator
import sample_select
from tensorflow.keras.models import load_model
import json
import distance_plot

# from tensorflow.keras.utils import plot_model
# import tensorflow_addons as tfa

# load dataset
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
train, val, test = utils.split_train_val_test(known_class, np.array(numerical_labels), 0.9, 0.05)
train_flows = train[0]
train_labels = train[1]
val_flows = val[0]
val_labels = val[1]
test_flows = test[0]
test_labels = test[1]

train_flows = train_flows / 255.0
val_flows = val_flows / 255.0
test_flows = test_flows / 255.0

# normolization
# mean_val = np.mean(train_flows)
# std_val = np.std(train_flows)
#
# train_flows = (train_flows - mean_val) / std_val
# val_flows = (val_flows - mean_val) / std_val
# test_flows = (test_flows - mean_val) / std_val


# add a channel dimension to the images
train_flows = np.expand_dims(train_flows, axis=-1)
val_flows = np.expand_dims(val_flows, axis=-1)
test_flows = np.expand_dims(test_flows, axis=-1)

numClasses = len(np.unique(numerical_labels))
print('[INFO]... num classes:' + str(numClasses))
# prepare the positive and negative pairs
print("[INFO] preparing triplet pairs...")
print("[INFO] preparing positive and negative pairs...")
center_map = sample_select.get_centroid(train_flows, train_labels, numClasses)
neighbors = sample_select.get_center_neighbors(train_flows, train_labels, numClasses, center_map,
                                               config.NEAR_NEIGHBOR_NUMS)
(pairTrain, labelTrain) = sample_select.random_generate_triplets(train_flows, train_labels, numClasses, 30)

center_map_val = sample_select.get_centroid(val_flows, val_labels, numClasses)
neighbors_val = sample_select.get_center_neighbors(val_flows, val_labels, numClasses, center_map_val,
                                                   config.NEAR_NEIGHBOR_NUMS)
(pairVal, labelVal) = sample_select.random_generate_triplets(val_flows, val_labels, numClasses, 30)

print("[INFO] loading siamese model...")
print("[INFO] model path:" + config.MODEL_PATH)
model = load_model(config.MODEL_PATH, compile=False)

print("[INFO] loading encoder model...")
print("[INFO] model path:" + config.ENCODER_PATH)
encoder = load_model(config.ENCODER_PATH, compile=False)

print(pairTrain.shape)


# for train_ds in sample_select.triplets_generator(train_flows, train_labels, neighbors,
#                                                  config.POS_RANDOM_NUMBERS):
#     train_dict = {'anchor': train_ds[0], 'pos': train_ds[1], 'neg': train_ds[2]}
#     train_label = {'output': train_ds[3]}
#
#     tuple_distance = model.predict(train_dict)

def mining_triplets_distance(arrA, arrB):
    distance = np.sqrt(np.sum(np.square(arrA - arrB)))
    return distance


integrate_embedding = model.predict([pairTrain[:, 0, :, :, :], pairTrain[:, 1, :, :, :], pairTrain[:, 2, :, :, :]])
print(integrate_embedding.shape)
anchors = integrate_embedding[:, : config.Embedding_Dim]
pos = integrate_embedding[:, config.Embedding_Dim:config.Embedding_Dim * 2]
neg = integrate_embedding[:, config.Embedding_Dim * 2:]
print(anchors.shape, pos.shape, neg.shape)

pos_distance = []
for i in range(len(anchors)):
    pos_distance.append(float(mining_triplets_distance(anchors[i], pos[i])))

neg_distance = []
for i in range(len(anchors)):
    neg_distance.append(float(mining_triplets_distance(anchors[i], neg[i])))

distance_plot.stacked_plot(pos_distance, neg_distance, 'huge_data')

pos_distance = np.array(pos_distance)
neg_distance = np.array(neg_distance)

print('POS distance')
print(np.min(pos_distance), np.max(pos_distance))
print(np.mean(pos_distance))
print('NEG distance')
print(np.min(neg_distance), np.max(neg_distance))
print(np.mean(neg_distance))
