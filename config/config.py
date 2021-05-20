import os
ope = os.path.exists
import numpy as np
import socket
import warnings
warnings.filterwarnings('ignore')

#sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#hostname = socket.gethostname()
#print('run on %s' % hostname)

IMG_PATHS = {
  'test': r'F:\test\npy.512.rgby',
  'custom': r'X:\TestFiles\test\images_768',
  'custom512': r'F:\TestFiles512\test\images_512',
  #'train': r'F:\train\npy.1536.rgby'
  'train': r'D:\Crops'
  #'train': r'D:\NucleiCrops\224'
    #'train': r'X:\train\npy.768.rgby'
  #  'train': r'F:\TrainCrops380'
}

LOSSES = {
    'BCE': 'HardLogLoss',
    'L1': 'L1',
    'L2': 'L2',
    'Focal': 'FocalLoss',
    'BestFitting': 'FocalSymmetricLovaszHardLogLoss',
    'ROCStar': 'ROCStar',
    'NeoFocalLoss': 'NeoFocalLoss',
    'FocalLoss': 'FocalLoss'
}

RESULT_DIR     = r"F:\result"
DATA_DIR       = "F:\\"
PRETRAINED_DIR = r"D:\BestFitting\data5\data\pretrained"
#TIF_DIR        = r"D:\HPA"
#EXTERNEL_DIR   = r"D:\BestFitting\data\data\protein"

PI  = np.pi
INF = np.inf
EPS = 1e-12

IMG_SIZE      = 512
NUM_CLASSES   = 19
ID            = 'Id'
PREDICTED     = 'Predicted'
TARGET        = 'Target'
PARENTID      = 'ParentId'
EXTERNAL      = 'External'
ANTIBODY      = 'antibody'
ANTIBODY_CODE = 'antibody_code'

LABEL_NAMES = {
    0:  "0:Nucleoplasm",
    1:  "1:Nuclear membrane",
    2:  "2:Nucleoli",
    3:  "3:Nucleoli fibrillar center",
    4:  "4:Nuclear speckles",
    5:  "5:Nuclear bodies",
    6:  "6:Endoplasmic reticulum",
    7:  "7:Golgi apparatus",
    8:  "8:Intermediate filaments",
    9:  "9:Actin filaments",
    10:  "10:Microtubules",
    11:  "11:Mitotic spindle",
    12:  "12:Centrosome",
    13:  "13:Plasma membrane",
    14:  "14:Mitochondria",
    15:  "15:Aggresome",
    16:  "16:Cytosol",
    17:  "17:Vesicles and punctate cytosolic patterns",
    18:  "18:Negative"
}
LABEL_NAME_LIST = [LABEL_NAMES[idx] for idx in range(len(LABEL_NAMES))]

NUCLEI_NAME_LIST = ['Nucleoplasm', 'Nuclear Membrane', 'Nucleoli',
       'Nucleoli Fibrillar Center', 'Nuclear Speckles', 'Nuclear Bodies', 'Mitotic Spindle']

COLOR_INDEXS = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'yellow': 0,
}
