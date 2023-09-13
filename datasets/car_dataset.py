""" Stanford Cars (Car) Dataset
Created: Nov 15,2019 - Yuchong Gu
Revised: Nov 15,2019 - Yuchong Gu
"""
import os
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from utils import get_transform

DATAPATH = '/data/cj/dataset/stanford_cars'


class CarDataset(Dataset):
    """
    # Description:
        Dataset for retrieving Stanford Cars images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', data_path=DATAPATH, resize=500):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.data_path = data_path
        self.num_classes = 196


        if phase == 'train':
            list_path = os.path.join(self.data_path, 'train.mat')
        else:
            list_path = os.path.join(self.data_path, 'test.mat')

        list_mat = loadmat(list_path)
        self.images = [f.item() for f in list_mat['annotations']['relative_im_path'][0]]
        self.labels = [f.item() for f in list_mat['annotations']['class'][0]]

        # transform
        self.transform = get_transform(self.resize, self.phase)
    def __getitem__(self, item):
        # image
        image = Image.open(os.path.join(self.data_path, self.images[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, self.labels[item] - 1  # count begin from zero

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    ds = CarDataset('val')
    # print(len(ds))
    for i in range(0, 100):
        image, label = ds[i]
        # print(image.shape, label)
