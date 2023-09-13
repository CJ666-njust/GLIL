""" CUB-200-2011 (Bird) Dataset
Created: Oct 11,2019 - Yuchong Gu
Revised: Oct 11,2019 - Yuchong Gu
"""
import os
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from utils import get_transform

DATAPATH = '/data/cj/dataset/CUB_200_2011'
image_path = {}
image_label = {}


class BirdDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels

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
        self.data_path = data_path
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.num_classes = 200
        # self.transform = torchvision.transforms.Compose([transforms.ToTensor()])
        # get image path from images.txt
        with open(os.path.join(self.data_path, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                image_path[id] = path

        # get image label from image_class_labels.txt
        with open(os.path.join(self.data_path, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                image_label[id] = int(label)

        # get train/test image id from train_test_split.txt
        with open(os.path.join(self.data_path, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)

                if self.phase == 'train' and is_training_image:
                    self.image_id.append(image_id)
                if self.phase in ('val', 'test') and not is_training_image:
                    self.image_id.append(image_id)

        # transform
        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]

        # image
        address = os.path.join(self.data_path, 'images', image_path[image_id])
        image = Image.open(os.path.join(self.data_path, 'images', image_path[image_id])).convert('RGB')  # (C, H, W)
        
        image = self.transform(image)
        # image = np.array(image, dtype=np.uint8)
        
        # return image and label
        return image, image_label[image_id] - 1  # count begin from zero
        # return address and label
        return address, image_label[image_id] - 1
    def __len__(self):
        return len(self.image_id)


if __name__ == '__main__':
    ds = BirdDataset('train')
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        print(image.shape, label)
