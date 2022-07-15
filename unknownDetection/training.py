# import the necessary packages
import metrics
import config
import utils
import model
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model

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
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = model.Lenet(config.IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
# finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])
model = Model(inputs=[imgA, imgB], outputs=distance)

# compile the model
print("[INFO] compiling model...")
model.compile(loss=metrics.contrastive_loss, optimizer="adam")
# train the model
print("[INFO] training model...")

numClasses = len(np.unique(numerical_labels))
print('[INFO]... num classes:' + str(numClasses))
# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")

(pairTrain, labelTrain) = utils.make_pairs_balance(train_flows, train_labels, numClasses, 50)
(pairVal, labelVal) = utils.make_pairs_balance(val_flows, val_labels, numClasses, 50)
(pairTest, labelTest) = utils.make_pairs_balance(test_flows, test_labels, numClasses, 50)

print(pairTrain.shape, labelTrain.shape)
history = model.fit(
    [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
    validation_data=([pairVal[:, 0], pairVal[:, 1]], labelVal[:]),
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS, callbacks=[utils.dynamic_LR()])

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)
# plot the training history
print("[INFO] plotting training history...")
utils.contrastive_plot(history, config.PLOT_PATH)


pred_val = []
#get all distances
for i in range(len(pairTrain)):
    imageA = pairTrain[i][0]
    imageB = pairTrain[i][1]

    label = int(labelTrain[i])
    # add a batch dimension to both images
    imageA = np.expand_dims(imageA, axis=0)
    imageB = np.expand_dims(imageB, axis=0)

    # indicating whether the images belong to the same class or not
    preds = model.predict([imageA, imageB])
    proba = preds[0][0]
    pred_val.append(proba)

min_dist = min(pred_val)
max_dist = max(pred_val)

bins = 50
steps = (max_dist - min_dist) / bins
threshold = []
for i in range(50):
    threshold.append(min_dist + steps * i)

accuracy = {}
total_num = len(labelTrain)
for thre in threshold:
    correct = 0
    for idx in range(len(pred_val)):
        pred = 1 if pred_val[idx] < thre else 0
        if pred == int(labelTrain[idx]):
            correct += 1
    accuracy[thre] = round(correct / total_num, 2)

print('[optimal Threshold...]')
best_threshold = max(accuracy, key=accuracy.get)
print(best_threshold)
print('[best accuracy...]')
print(accuracy[best_threshold])