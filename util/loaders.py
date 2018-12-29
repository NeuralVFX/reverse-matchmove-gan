import random
import os
import numpy as np
import cv2
import pandas as pd
from torch.utils.data.sampler import *

cv2.setNumThreads(0)


############################################################################
#  Loader Utilities
############################################################################


class NormDenorm:
    # Store mean and std for transforms, apply normalization and de-normalization
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def norm(self, img):
        # normalize image to feed to network
        return (img - self.mean) / self.std

    def denorm(self, img, cpu=True, variable=True):
        # reverse normalization for viewing
        if cpu:
            img = img.cpu()
        if variable:
            img = img.data
        img = img.numpy().transpose(1, 2, 0)
        return img * self.std + self.mean


def cv2_open(fn):
    # Get image with cv2 and convert from bgr to rgb
    try:
        im = cv2.imread(str(fn), cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR).astype(
            np.float32) / 255
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f'Image Open Failure:{fn}  Error:{e}')


############################################################################
# Image / Matrix Augmentation Stuff
############################################################################


def make_img_square(input_img):
    # Take rectangular image and crop to square
    height = input_img.shape[0]
    width = input_img.shape[1]

    if height > width:
        input_img = input_img[height // 2 - (width // 2):height // 2 + (width // 2), :, :]
    if width > height:
        input_img = input_img[:, width // 2 - (height // 2):width // 2 + (height // 2), :]
    return input_img


class ResizeCV(object):
    # resize image
    def __init__(self, output_size):
        self.output_size = int(output_size)

    def __call__(self, sample):
        image = sample['image']
        image = make_img_square(image)
        image = cv2.resize(image, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)
        return {'image': image}


def aug_mat(start_mat, av, rot, zoom):
    #### generate new world mat ####
    rot_mat = cv2.getRotationMatrix2D((1, 1), rot, 1)
    new_mat = np.array(np.eye(4))
    new_mat[:2, :2] = rot_mat[:, :2].T
    new_mat = start_mat @ new_mat
    av *= zoom
    return new_mat, av


def aug_im(start, rot, zoom):
    r, c, z = start.shape
    rot_mat = cv2.getRotationMatrix2D((c / 2, r / 2), rot, zoom)
    aug_img = cv2.warpAffine(start, rot_mat, (c, r), flags=cv2.INTER_AREA)

    return aug_img


class TranImgMat(object):
    # apply random transform to image a and b #

    def __init__(self, rot=15, zoom=.2, zoom_offset=1.2, res=256):
        # store range of possible transformations
        self.rot = rot
        self.zoom = zoom
        self.zoom_offset = zoom_offset
        self.resize = ResizeCV(res)

    def get_random_transform(self, image, mat, focal_len, seed):
        # create random transformation matrix
        rot = ((random.random() - .5) * 2) * self.rot

        zoom = (random.random() * self.zoom) + self.zoom_offset

        image = aug_im(image, -rot, zoom)
        img_dict = self.resize({'image': image})
        mat, focal_len = aug_mat(mat, focal_len, rot, zoom)

        return img_dict['image'], mat, focal_len

    def __call__(self, sample):
        # get transform and apply to both images
        image_a = sample['image']
        matrix = sample['matrix']
        focal_len = sample['focal_len']
        seed = sample['seed']
        image, mat, focal_len = self.get_random_transform(image_a, matrix, focal_len, seed)

        return {'image': image, 'matrix': mat, 'focal_len': focal_len}


############################################################################
#  Dataset and Loader
############################################################################


class ImageMatrixDataset:
    # Load Images from User Supplied Path and Apply Augmentation

    def __init__(self, path_a, transform, output_res=256, train=True, repo=False):
        # add image colorspace transform
        # add output res conversion
        d_dict = {'m0': np.float64,
                  'm1': np.float64,
                  'm2': np.float64,
                  'm3': np.float64,
                  'm4': np.float64,
                  'm5': np.float64,
                  'm6': np.float64,
                  'm7': np.float64,
                  'm8': np.float64,
                  'm9': np.float64,
                  'm10': np.float64,
                  'm11': np.float64,
                  'm12': np.float64,
                  'm13': np.float64,
                  'm14': np.float64,
                  'm15': np.float64,
                  'focal_length': np.float64
                  }
        self.df = pd.read_csv(path_a, dtype=d_dict)
        self.path = os.path.dirname(path_a)
        self.output_res = output_res
        self.data_transforms = TranImgMat(res=output_res, zoom=.2, zoom_offset=1.2)
        self.prev_data_transforms = TranImgMat(res=output_res, rot=0, zoom=0, zoom_offset=1.3)
        self.transform = transform
        self.train = train
        self.offset_id = 0
        self.epoch_seed = 1.0
        self.batch_size = 1
        self.repo = repo

    def appy_augmentation(self, image, matrix, focal_len, seed):
        if self.train:
            data_dict = self.data_transforms({'image': image, 'matrix': matrix, 'focal_len': focal_len, 'seed': seed})
        else:
            data_dict = self.prev_data_transforms(
                {'image': image, 'matrix': matrix, 'focal_len': focal_len, 'seed': seed})
        norm_img = np.rollaxis(self.transform.norm(data_dict['image']), 2)
        matrix = data_dict['matrix'].reshape(16)[:12]
        matrix = np.append(matrix, data_dict['focal_len'])
        return norm_img, matrix.reshape(13, 1, 1)

    def __getitem__(self, index):
        matrix = self.df.iloc[index, 0:16]
        matrix = np.array(matrix).reshape(4, 4).astype('float64')
        focal_len = self.df.iloc[index, 16]
        filename = self.df.iloc[index, 17]
        if self.repo:
            image = np.ones([self.output_res, self.output_res, 3])
        else:
            image = cv2_open(os.path.join(self.path, filename))
        seed = (int((index - self.offset_id) / self.batch_size) * self.epoch_seed)
        image, matrix = self.appy_augmentation(image, matrix, focal_len, seed)

        return torch.FloatTensor(image), torch.FloatTensor(matrix)

    def __len__(self):
        return self.df.shape[0]
