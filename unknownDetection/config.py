# import the necessary packages
import os

# specify the shape of the inputs for our network
IMG_SHAPE = (6, 50, 1)
# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 30
MARGIN = 5

NEAR_NEIGHBOR_NUMS = 100
POS_RANDOM_NUMBERS = 50
NEG_RANDOM_NUMBERS = 30

UNKNOWN_CATEGORIES = ['vpn_email', 'vpn_voipbuster']


VPN_DATA_PATH = '/home/cnic-zyd/Luyang/code/'
VPN_DATA_FILE = 'vpn_dataset_50.txt'

# define the path to the base output directory
BASE_OUTPUT = "model1"

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
ENCODER_PATH = os.path.sep.join([BASE_OUTPUT, "lenet_encoder"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "lenet_plot.pdf"])


# all_labels = {'vpn_skype_audio', 'vpn_youtube', 'vpn_spotify', 'vpn_sftp', 'vpn_icq_chat', 'vpn_skype_files',
#      'vpn_hangouts_chat', 'vpn_netflix', 'vpn_facebook_chat', 'vpn_facebook_audio', 'vpn_bittorrent', 'vpn_voipbuster',
#      'vpn_vimeo', 'vpn_hangouts_audio', 'vpn_aim_chat', 'vpn_ftps', 'vpn_email', 'vpn_skype_chat'}