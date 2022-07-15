import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import config
import utils
import data_generator
import numpy as np
import sample_select
from tensorflow.keras.layers import Input

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load data
# load dataset
flows, labels = data_generator.load_data()
# 使用非负值进行训练
flows = np.abs(flows)

# transfer the textual into numerical labels
unique_labels = set(labels)
print(unique_labels)

(known_class, known_label), (unknown_class, unknown_label) = utils.split_unknown(flows, labels,
                                                                                 config.UNKNOWN_CATEGORIES)
print('[INFO...] known class shape')
print(known_class.shape, known_label.shape)
print('[INFO...] unknown class shape')
print(unknown_class.shape, unknown_label.shape)

# 只构造已知流的 文字---标号转换
training_labels = set(known_label)

label_mapping = {}
for index, value in enumerate(training_labels):
    label_mapping[value] = index
label_reverse = {}
for index, value in enumerate(training_labels):
    label_reverse[index] = value

numerical_labels = []
for lab in known_label:
    numerical_labels.append(label_mapping[lab])

# 此处可以不构造测试集
print(['INFO... Split train and test'])
train, val, test = utils.split_train_val_test(known_class, np.array(numerical_labels), 0.9, 0.05)
train_flows = train[0]
train_labels = train[1]
val_flows = val[0]
val_labels = val[1]
test_flows = test[0]
test_labels = test[1]

# Min-Max normalization
# maximum = np.max(train_flows)
# minimum = np.min(train_flows)
# train_flows = (train_flows - minimum) / (maximum - minimum)
# val_flows = (val_flows - minimum) / (maximum - minimum)
# test_flows = (test_flows - minimum) / (maximum - minimum)

train_flows = train_flows / 255
val_flows = val_flows / 255
test_flows = test_flows / 255

# add a channel dimension to the images
train_flows = np.expand_dims(train_flows, axis=-1)
val_flows = np.expand_dims(val_flows, axis=-1)
test_flows = np.expand_dims(test_flows, axis=-1)

numClasses = len(np.unique(numerical_labels))
print("[INFO] preparing positive and negative pairs...")
center_map = sample_select.get_centroid(train_flows, train_labels, numClasses)
neighbors = sample_select.get_center_neighbors(train_flows, train_labels, numClasses, center_map,
                                               config.NEAR_NEIGHBOR_NUMS)
(pairTrain, labelTrain) = sample_select.generate_near_pairs(train_flows, train_labels, numClasses, neighbors,
                                                            config.POS_RANDOM_NUMBERS)

center_map_val = sample_select.get_centroid(val_flows, val_labels, numClasses)
neighbors_val = sample_select.get_center_neighbors(val_flows, val_labels, numClasses, center_map_val,
                                                   config.NEAR_NEIGHBOR_NUMS)
(pairVal, labelVal) = sample_select.generate_near_pairs(val_flows, val_labels, numClasses, neighbors_val,
                                                        config.POS_RANDOM_NUMBERS)

print("[INFO] loading siamese model...")
model = load_model(config.MODEL_PATH, compile=False)

# 直接预测距离
pred_val = model.predict([pairTrain[0:10, 0, :, :, :], pairTrain[0:10, 1, :, :, :]])
print(pred_val)

print("[INFO] loading feature_extractor model...")
print("[INFO] model path:" + config.ENCODER_PATH)
encoder = load_model(config.ENCODER_PATH, compile=False)

for i in range(10):
    img1 = pairTrain[i, 0, :, :, :]
    img1 = np.expand_dims(img1, axis=0)
    feature1 = encoder(img1)

    img2 = pairTrain[i, 1, :, :, :]
    img2 = np.expand_dims(img2, axis=0)
    feature2 = encoder(img2)
    print(feature1.shape, feature2.shape)

    dis = utils.euclidean_distance((feature1, feature2))
    print(dis)
