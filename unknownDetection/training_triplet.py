# import the necessary packages
import metrics
import config
import utils
import model
import json
import numpy as np
import collections
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda

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

# normolization
# maximum = np.max(train_flows)
# minimum = np.min(train_flows)
#
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

# configure the siamese network
print("[INFO] building siamese network...")
anchor = Input(shape=config.IMG_SHAPE)
pos_anchor = Input(shape=config.IMG_SHAPE)
neg_anchor = Input(shape=config.IMG_SHAPE)

featureExtractor = model.Lenet(config.IMG_SHAPE)
feats_a = featureExtractor(anchor)
feats_ap = featureExtractor(pos_anchor)
feats_an = featureExtractor(neg_anchor)

# finally, construct the siamese network
distance = Lambda(utils.distance_pairs)([feats_a, feats_ap, feats_an])
model = Model(inputs=[anchor, pos_anchor, neg_anchor], outputs=distance, name='siamese')

# model.summary()


# compile the model
print("[INFO] compiling model...")
model.compile(loss=metrics.triplet_loss, optimizer="adam")

numClasses = len(np.unique(numerical_labels))
print('[INFO]... num classes:' + str(numClasses))
# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")

# 新增测试
(pairTrain, labelTrain, train_map) = utils.generate_triplets_with_records(train_flows, train_labels, numClasses,
                                                                          50)
(pairVal, labelVal, val_map) = utils.generate_triplets_with_records(val_flows, val_labels, numClasses, 50)
(pairTest, labelTest, test_map) = utils.generate_triplets_with_records(test_flows, test_labels, numClasses, 50)
# 序列化MAP， 同时将标签从数字转化为字符


train_string_map = {}
for k in train_map.keys():
    res = collections.Counter(train_map[k]).most_common()
    num = min(len(res), 50)
    train_string_map[k] = res[:num]

true_name = {}
for k in train_string_map.keys():
    true_name[str(label_reverse[k])] = [int(item[0]) for item in train_string_map[k]]

with open('sample_record.txt', 'w') as f:
    f.write(json.dumps(true_name))
    f.write('\n')
    # string as key to number
    f.write(json.dumps(label_mapping))
    f.write('\n')
    # number as key to string
    f.write(json.dumps(label_reverse))
    f.write('\n')

print(pairTrain.shape, labelTrain.shape)
print(pairTrain[:, 0].shape, pairTrain[:, 1].shape, pairTrain[:, 2].shape)
history = model.fit(
    [pairTrain[:, 0], pairTrain[:, 1], pairTrain[:, 2]], labelTrain[:],
    validation_data=([pairVal[:, 0], pairVal[:, 1], pairVal[:, 2]], labelVal[:]),
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS, callbacks=[utils.dynamic_LR()])

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)
print("[INFO] saving encoder model...")
featureExtractor.save(config.ENCODER_PATH)
# plot the training history
print("[INFO] plotting training history...")
utils.contrastive_plot(history, config.PLOT_PATH)

# model shape
# featureExtractor.summary()
# Testing the encoder:
print('*' * 50)
print(pairTrain.shape)
anchor_array = pairTrain[:, 0, :, :, :]
pos_array = pairTrain[:, 1, :, :, :]
neg_array = pairTrain[:, 2, :, :, :]

batch = 10000
iterations = int(len(anchor_array) / batch)

a = []
for i in range(iterations + 1):
    a.append(featureExtractor(anchor_array[batch * i:batch * (i + 1), :, :, :]))
anchor_features = tf.concat(a, axis=0)
print('anchor_features shape', anchor_features.shape)

a = []
for i in range(iterations + 1):
    a.append(featureExtractor(pos_array[batch * i:batch * (i + 1), :, :, :]))
pos_features = tf.concat(a, axis=0)
print('pos_features shape', pos_features.shape)

a = []
for i in range(iterations + 1):
    a.append(featureExtractor(neg_array[batch * i:batch * (i + 1), :, :, :]))
neg_features = tf.concat(a, axis=0)
print('neg_features shape', neg_features.shape)

pos_distance = utils.euclidean_distance((anchor_features, pos_features))
neg_distance = utils.euclidean_distance((anchor_features, neg_features))
print('distance shape:', pos_distance.shape, neg_distance.shape)

pos_distance = [float(pos_distance[i][0]) for i in range(len(pos_distance))]
neg_distance = [float(neg_distance[i][0]) for i in range(len(neg_distance))]

with open('triplet_thresholds.txt', 'w') as f:
    f.write(json.dumps(pos_distance))
    f.write('\n')
    f.write(json.dumps(neg_distance))
    f.write('\n')

min_dist = min(pos_distance)
max_dist = max(pos_distance)

bins = 100
steps = (max_dist - min_dist) / bins
threshold = []
for i in range(bins):
    threshold.append(min_dist + steps * i)

accuracy = {}
total_num = len(labelTrain)
for thre in threshold:
    correct = 0
    for idx in range(len(pos_distance)):
        if (pos_distance[idx] <= thre) and (neg_distance[idx] > thre):
            correct += 1
    accuracy[thre] = round(correct / total_num, 2)

print('[optimal Threshold...]')
best_threshold = max(accuracy, key=accuracy.get)
print(best_threshold)
print('[best accuracy...]')
print(accuracy[best_threshold])
