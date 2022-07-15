# import the necessary packages
import metrics
import config
import utils
import model
import json
import numpy as np
import collections
import sample_select
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import ModelCheckpoint

# load MNIST dataset and scale the pixel values to the range of [0, 1]
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
category_mapping = {'vpn_skype_audio':'VoIP', 'vpn_youtube':'streaming', 'vpn_spotify':'streaming', 'vpn_sftp':'file_transfer', 'vpn_icq_chat':'chat', 'vpn_skype_files':'file_transfer',
     'vpn_hangouts_chat':'chat', 'vpn_netflix':'streaming', 'vpn_facebook_chat':'chat', 'vpn_facebook_audio':'VoIP', 'vpn_bittorrent':'P2P', 'vpn_voipbuster':'VoIP',
     'vpn_vimeo':'streaming', 'vpn_hangouts_audio':'VoIP', 'vpn_aim_chat':'chat', 'vpn_ftps':'file_transfer', 'vpn_email':'email', 'vpn_skype_chat':'chat'}

print('len:', len(category_mapping))
labels = [category_mapping[item] for item in labels]

# transfer the textual into numerical labels
unique_labels = set(labels)
print(unique_labels)

unknown_categories = ['email']
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
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = model.Lenet(config.IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
# finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])
model = Model(inputs=[imgA, imgB], outputs=distance, name='siamese')

# model.summary()

# featureExtractor.summary()

# compile the model
print("[INFO] compiling model...")
model.compile(loss=metrics.contrastive_loss, optimizer="adam")
# train the model
print("[INFO] training model...")

numClasses = len(np.unique(numerical_labels))
print('[INFO]... num classes:' + str(numClasses))
# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")

# 新增测试
center_map = sample_select.get_centroid(train_flows, train_labels, numClasses)
neighbors = sample_select.get_center_neighbors(train_flows, train_labels, numClasses, center_map, 200)
(pairTrain, labelTrain) = sample_select.generate_near_pairs(train_flows, train_labels, numClasses, neighbors, 50)

center_map_val = sample_select.get_centroid(val_flows, val_labels, numClasses)
neighbors_val = sample_select.get_center_neighbors(val_flows, val_labels, numClasses, center_map_val, 200)
(pairVal, labelVal) = sample_select.generate_near_pairs(val_flows, val_labels, numClasses, neighbors_val, 50)

# (pairTest, labelTest) = sample_select.generate_near_pairs(test_flows, test_labels, numClasses, neighbors, 50)

with open('email_neighbors.txt', 'w') as f:
    f.write(json.dumps(neighbors))
    f.write('\n')
    # string as key to number
    f.write(json.dumps(label_mapping))
    f.write('\n')
    # number as key to string
    f.write(json.dumps(label_reverse))
    f.write('\n')

print(pairTrain.shape, labelTrain.shape)
# model_checkpoint = ModelCheckpoint(config.MODEL_PATH, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
history = model.fit(
    [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
    validation_data=([pairVal[:, 0], pairVal[:, 1]], labelVal[:]),
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

# get all distances
print('LEN', len(pairTrain))
pred_val = model.predict([pairTrain[:, 0, :, :, :], pairTrain[:, 1, :, :, :]])

print('Prediction Len:', pred_val.shape)

pos_index = np.where(labelTrain == 1)[0]
neg_index = np.where(labelTrain == 0)[0]

pos_dist = []
for i in range(len(pos_index)):
    pos_dist.append(float(pred_val[pos_index[i]][0]))
neg_dist = []
for i in range(len(neg_index)):
    neg_dist.append(float(pred_val[neg_index[i]][0]))

sorted(pos_dist)
sorted(neg_dist)

with open('file_thresholds.txt', 'w') as f:
    f.write(json.dumps(pos_dist))
    f.write('\n')
    f.write(json.dumps(neg_dist))
    f.write('\n')

sim_min = min(pos_dist)
sim_max = max(pos_dist)
print('similarity distance: ', sim_min, sim_max)
disi_min = min(neg_dist)
disi_max = max(neg_dist)
print('disimilarity distance: ', disi_min, disi_max)

min_dist = min(pos_dist)
max_dist = max(pos_dist)

bins = 100
steps = (max_dist - min_dist) / bins
threshold = []
for i in range(bins):
    threshold.append(min_dist + steps * i)

accuracy = {}
total_num = len(labelTrain)
for thre in threshold:
    correct = 0
    for idx in range(len(pred_val)):
        pred = 1 if pred_val[idx][0] < thre else 0
        if pred == labelTrain[idx][0]:
            correct += 1
    accuracy[thre] = round(correct / total_num, 2)

print('[optimal Threshold...]')
best_threshold = max(accuracy, key=accuracy.get)
print(best_threshold)
print('[best accuracy...]')
print(accuracy[best_threshold])
