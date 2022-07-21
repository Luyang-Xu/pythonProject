# import the necessary packages
import os

# specify the shape of the inputs for our network
IMG_SHAPE = (32, 32, 1)
# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 20  # 40
MARGIN = 0.2
Embedding_Dim = 64

NEAR_NEIGHBOR_NUMS = 50
POS_RANDOM_NUMBERS = 10
NEG_RANDOM_NUMBERS = 5

UNKNOWN_CATEGORIES = ['vpn_icq_chat', 'vpn_hangouts_chat', 'vpn_facebook_chat', 'vpn_aim_chat', 'vpn_skype_chat']

# POS distance
# 0.0 4.795831203460693
# 0.21096686142288393
# NEG distance
# 0.0 6.8112311363220215
# 5.241804395122127

VPN_DATA_PATH = '/home/cnic-zyd/lenet/dataset/'
VPN_DATA_FILE = 'vpn_dataset_32_32.txt'

# define the path to the base output directory
BASE_OUTPUT = "model_triplet"

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
ENCODER_PATH = os.path.sep.join([BASE_OUTPUT, "lenet_encoder"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "lenet_plot.pdf"])

# all_labels = {'vpn_skype_audio', 'vpn_youtube', 'vpn_spotify', 'vpn_sftp', 'vpn_icq_chat', 'vpn_skype_files',
#      'vpn_hangouts_chat', 'vpn_netflix', 'vpn_facebook_chat', 'vpn_facebook_audio', 'vpn_bittorrent', 'vpn_voipbuster',
#      'vpn_vimeo', 'vpn_hangouts_audio', 'vpn_aim_chat', 'vpn_ftps',  'vpn_email', 'vpn_skype_chat'}
