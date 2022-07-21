# import the necessary packages
import config
import utils
import model
import numpy as np
import data_generator
import sample_select
import json
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate

# from tensorflow.keras.utils import plot_model
# import tensorflow_addons as tfa

# load MNIST dataset
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

print(['INFO... Split train and test'])
train, val, test = utils.split_train_val_test(known_class, np.array(numerical_labels), 0.9, 0.05)
train_flows, train_labels = train[0], train[1]
val_flows, val_labels = val[0], val[1]
test_flows, test_labels = test[0], test[1]

train_flows, train_labels = shuffle(train_flows, train_labels, random_state=13)

# max:5008, min: 0 ,mean:51.828 ,std:175.050
train_flows = train_flows / 255.0
val_flows = val_flows / 255.0
test_flows = test_flows / 255.0

# add a channel dimension to the images
train_flows = np.expand_dims(train_flows, axis=-1)
val_flows = np.expand_dims(val_flows, axis=-1)
test_flows = np.expand_dims(test_flows, axis=-1)

# configure the siamese network
print("[INFO] building siamese network...")
anchor = Input(shape=config.IMG_SHAPE, name='anchor')
featureExtractor = model.Lenet(config.IMG_SHAPE)
print("[INFO] compiling model...")
featureExtractor.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer="adam")
# train the model
print("[INFO] training model...")

numClasses = len(np.unique(numerical_labels))
print('[INFO]... num classes:' + str(numClasses))
# prepare the positive and negative pairs
print("[INFO] preparing triplet pairs...")
print("[INFO] preparing positive and negative pairs...")
center_map = sample_select.get_centroid(train_flows, train_labels, numClasses)
neighbors = sample_select.get_center_neighbors(train_flows, train_labels, numClasses, center_map,
                                               config.NEAR_NEIGHBOR_NUMS)

with open('neighbors_semihard.txt', 'w') as f:
    f.write(json.dumps(neighbors))
    f.write('\n')
    # string as key to numbelr
    f.write(json.dumps(label_mapping))
    f.write('\n')
    # number as key to string
    f.write(json.dumps(label_reverse))
    f.write('\n')

# history_coll = []
#
# for i in range(config.EPOCHS):
#     print('[Epochs...]' + str(i))
#     for train_ds in sample_select.triplets_generator(train_flows, train_labels, neighbors,
#                                                      config.POS_RANDOM_NUMBERS):
#         train_dict = {'anchor': train_ds[0], 'pos': train_ds[1], 'neg': train_ds[2]}
#         train_label = {'output': train_ds[3]}
#
#         history = model.fit(train_dict, train_label,
#                             validation_data=([pairVal[:, 0], pairVal[:, 1], pairVal[:, 2]], labelVal[:]),
#                             batch_size=config.BATCH_SIZE, epochs=1,
#                             callbacks=[utils.dynamic_LR()])
#         history_coll.append(history)
history = featureExtractor.fit(train_flows, train_labels,
                               validation_data=(val_flows, val_labels),
                               batch_size=config.BATCH_SIZE,
                               epochs=config.EPOCHS, callbacks=[utils.dynamic_LR()])

print("[INFO] saving encoder model...")
featureExtractor.save(config.ENCODER_PATH)
# plot the training history
print("[INFO] plotting training history...")
utils.contrastive_plot(history, config.PLOT_PATH)
