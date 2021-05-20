import os
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset


from utils.common_util import *
import pandas as pd
from config.config import *
from datasets.tool import *
from utils.augment_util import *
from PIL import Image
import re

class ProteinDataset(Dataset):
    def __init__(self,
                 split_file,
                 img_size=512,
                 transform=None,
                 return_label=True,
                 is_trainset=True,
                 in_channels=4,
                 crop_size=0,
                 random_crop=False,
                 img_dir=None,
                 puzzle=True,
                 BGR=False,
                 nuclei=False,
                 labelmask=[],
                 ):
        self.is_trainset = is_trainset
        self.img_size = img_size
        self.return_label = return_label
        self.in_channels = in_channels
        self.transform = transform
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.BGR = BGR
        data_type = 'train' if is_trainset else 'test'
        if not isinstance(split_file, pd.DataFrame):
            split_df = pd.read_csv(split_file)
        else:
            split_df = split_file

        self.img_dir = img_dir
        self.puzzle = puzzle

        self.labelmask = labelmask

        # Identify which type of files we're dealing with
        x = os.listdir(self.img_dir)
        self.extension = x[5].split('.')[-1]

        #if EXTERNAL not in split_df.columns:
        #    split_df[EXTERNAL] = False

        self.split_df = split_df

        if nuclei:
            labeltype = NUCLEI_NAME_LIST
        else:
            labeltype = LABEL_NAME_LIST


        if len(self.split_df.columns) > 8:
            self.labels = self.split_df[labeltype].values.astype(int)
            assert self.labels.shape == (len(self.split_df), len(labeltype))
        else:
            self.labels = []


        #self.is_external = self.split_df[EXTERNAL].values
        #self.img_ids = self.split_df[ID].values
        if 'Image' in self.split_df.columns:
            self.img_ids = self.split_df['Image'].values
        elif 'Id' in self.split_df.columns:
            self.img_ids = self.split_df['Id'].values
        elif 'ImageID' in self.split_df.columns:
            self.img_ids = self.split_df['ImageID'].values
        else:
            self.img_ids = self.split_df['ID'].values
        self.num = len(self.img_ids)

    def read_crop_img(self, img):
        random_crop_size = int(np.random.uniform(self.crop_size, self.img_size))
        x = int(np.random.uniform(0, self.img_size - random_crop_size))
        y = int(np.random.uniform(0, self.img_size - random_crop_size))
        crop_img = img[x:x + random_crop_size, y:y + random_crop_size]
        return crop_img

    def read_rgby(self, img_dir, img_id, index):
        #if self.is_external[index]:
        #    img_is_external = True
        #else:
        img_is_external = False

        if str('.' + self.extension) in img_id:
            img_id = img_id.split('.')[0]

        if self.extension == 'png':
            if len(img_id.split('.')) > 1:
                img_id = img_id.split('.')[0]
            rgb = cv2.imread(opj(img_dir, img_id + '.png'), flags=cv2.IMREAD_UNCHANGED)
            y_ch = np.expand_dims(a=cv2.imread(opj(img_dir, 'y=' + img_id + '.png'), flags=cv2.IMREAD_UNCHANGED), axis=-1)
            img = np.concatenate((rgb, y_ch), axis=2)

        elif self.extension == 'jpg':
        #if img_id[0:4] == 'test' or img_id[0:3] == 'aug':
            rgb = cv2.imread(opj(img_dir, img_id + '.jpg'), flags=cv2.IMREAD_UNCHANGED)
            y_ch = np.expand_dims(a=cv2.imread(opj(img_dir, 'y=' + img_id + '.jpg'), flags=cv2.IMREAD_UNCHANGED), axis=-1)
            img = np.concatenate((rgb, y_ch), axis=2)
        else:
            if self.extension == 'npy':
                suffix = '.npy'
            elif self.extension == 'npz':
                suffix = '.npz'
            if self.in_channels == 3:
                colors = ['red', 'green', 'blue']
            elif self.in_channels == 1:
                colors = ['green']
            else:
                colors = ['red', 'green', 'blue', 'yellow']

            if suffix == '.npz':
                img = np.load(opj(img_dir, img_id + suffix))['arr_0']
                if self.in_channels == 1:
                    img = img[..., 1] # Green channel only
                    img = np.expand_dims(img, axis=2)
            if suffix == '.npy':
                img = np.load(opj(img_dir, img_id + suffix))
                if self.in_channels == 1:
                    img = img[..., 1] # Green channel only
                    img = np.expand_dims(img, axis=2)
            else:
                flags = cv2.IMREAD_GRAYSCALE
                img = [cv2.imread(opj(img_dir, img_id + '_' + color + suffix), flags)
                       for color in colors]
                img = np.stack(img, axis=-1)
            if self.BGR: # Make correction for models trained on OpenCV but using numpy inputs
                img = np.concatenate((np.expand_dims(img[:, :, 2], axis=2), np.expand_dims(img[:, :, 1], axis=2),
                                      np.expand_dims(img[:, :, 0], axis=2), np.expand_dims(img[:, :, 3], axis=2)), axis=2)
        if self.random_crop and self.crop_size > 0:
            img = self.read_crop_img(img)
        return img

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        #if self.is_external[index]:
        #    img_dir = self.external_img_dir
        #else:
        img_dir = self.img_dir

        #if '_' in img_id:
        #    img_dir = img_dir.replace('train', 'ext')

        if img_dir == 'F:\\TrainCrops380':
            img_id = img_id.split('.')[0] + '.npz'

        image = self.read_rgby(img_dir, img_id, index)
        if image[0] is None:
            print(img_dir, img_id)

        h, w = image.shape[:2]

        if not self.puzzle:
            if self.crop_size > 0:
                if self.crop_size != h or self.crop_size != w:
                    image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            else:
                if self.img_size != h or self.img_size != w:
                    image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if self.transform is not None:
            image = self.transform(image)

        if self.puzzle: # Need to resize back up to full
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        image = image / 255.0
        image = image_to_tensor(image)

        if self.return_label:
            label = self.labels[index]
            if len(self.labelmask) != 0:
                label = np.multiply(label, self.labelmask)
            return image, label, index

        else:
            return image, index

    def __len__(self):
        return self.num
