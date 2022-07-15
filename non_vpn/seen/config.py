import os

VPN_DATA_PATH = '/home/cnic-zyd/seen/dataset/'
VPN_DATA_FILE = 'vpn_dataset_784.txt'
# specify the shape of the inputs for our network
IMG_SHAPE = (784, 1)
# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 20
MARGIN = 10


POS_RANDOM_NUMBERS = 50
UNKNOWN_CATEGORIES = ['vpn_email', 'vpn_voipbuster']

# define the path to the base output directory
BASE_OUTPUT = "model1"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
ENCODER_PATH = os.path.sep.join([BASE_OUTPUT, "lenet_encoder"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "lenet_plot.pdf"])

VPN_CATEGORY_MAPPING = {'vpn_skype_audio': 'vpn_voip', 'vpn_youtube': 'vpn_streaming', 'vpn_spotify': 'vpn_streaming',
                        'vpn_sftp': 'vpn_file', 'vpn_icq_chat': 'vpn_chat', 'vpn_skype_files': 'vpn_file',
                        'vpn_hangouts_chat': 'vpn_chat', 'vpn_netflix': 'vpn_streaming',
                        'vpn_facebook_chat': 'vpn_chat',
                        'vpn_facebook_audio': 'vpn_voip', 'vpn_bittorrent': 'vpn_p2p', 'vpn_voipbuster': 'vpn_voip',
                        'vpn_vimeo': 'vpn_streaming', 'vpn_hangouts_audio': 'vpn_voip', 'vpn_aim_chat': 'vpn_chat',
                        'vpn_ftps': 'vpn_file', 'vpn_email': 'vpn_email', 'vpn_skype_chat': 'vpn_chat'}

NONVPN_CATEGORY_MAPPING = {'p2p_multipleSpeed': 'p2p', 'browsing': 'browsing', 'browsing_ara': 'browsing',
                           'browsing_ger': 'browsing', 'FTP_filetransfer': 'file', 'ssl': 'file', 'p2p_vuze': 'p2p',
                           'SSL_Browsing': 'browsing', 'Skype_Audio': 'voip',
                           'Vimeo_Workstation': 'streaming', 'Youtube_HTML5_Workstation': 'streaming',
                           'Skype_Voice_Workstation': 'voip',
                           'spotify': 'streaming', 'Youtube_Flash_Workstation': 'streaming', 'skype_transfer': 'file',
                           'Hangouts_voice_Workstation': 'voip', 'Facebook_Voice_Workstation': 'voip',
                           'facebook_Audio': 'voip', 'skypechat': 'chat', 'Hangout_Audio': 'voip',
                           'facebookchat': 'chat',
                           'Email_IMAP_filetransfer': 'email', 'skype_chat': 'chat',
                           'Workstation_Thunderbird_Imap': 'p2p', 'hangoutschat': 'chat', 'hangout_chat': 'chat',
                           'Workstation_Thunderbird_POP': 'p2p', 'POP_filetransfer': 'email',
                           'facebook_chat': 'chat', 'SFTP_filetransfer': 'file', 'icqchat': 'chat',
                           'spotifyAndrew': 'streaming', 'ICQ_Chat': 'chat', 'aimchat': 'chat', 'AIM_Chat': 'chat',
                           'browsing_ara2': 'browsing',
                           'browsing2': 'browsing', 'spotify2': 'streaming'}

NONVPN_SERVICE_MAPPING = {'p2p_multipleSpeed': 'p2p', 'browsing': 'browsing', 'browsing_ara': 'browsing',
                          'browsing_ger': 'browsing', 'FTP_filetransfer': 'file', 'ssl': 'file', 'p2p_vuze': 'p2p',
                          'SSL_Browsing': 'browsing', 'Skype_Audio': 'skype_audio',
                          'Vimeo_Workstation': 'vimeo', 'Youtube_HTML5_Workstation': 'youtube',
                          'Skype_Voice_Workstation': 'skype_audio',
                          'spotify': 'spotify', 'Youtube_Flash_Workstation': 'youtube', 'skype_transfer': 'file',
                          'Hangouts_voice_Workstation': 'hangouts_audio',
                          'Facebook_Voice_Workstation': 'facebook_audio',
                          'facebook_Audio': 'facebook_audio', 'skypechat': 'skype_chat',
                          'Hangout_Audio': 'hangouts_audio',
                          'facebookchat': 'facebook_chat',
                          'Email_IMAP_filetransfer': 'email', 'skype_chat': 'skype_chat',
                          'Workstation_Thunderbird_Imap': 'p2p', 'hangoutschat': 'hangouts_chat',
                          'hangout_chat': 'hangouts_chat',
                          'Workstation_Thunderbird_POP': 'p2p', 'POP_filetransfer': 'email',
                          'facebook_chat': 'facebook_chat', 'SFTP_filetransfer': 'file', 'icqchat': 'icq_chat',
                          'spotifyAndrew': 'spotify', 'ICQ_Chat': 'icq_chat', 'aimchat': 'aim_chat',
                          'AIM_Chat': 'aim_chat', 'browsing_ara2': 'browsing',
                          'browsing2': 'browsing', 'spotify2': 'spotify'}
