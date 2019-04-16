# encoding: utf-8

"""
Read images and corresponding labels.
"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class HW_Dataset(Dataset):
    def __init__(self, labels_dir, imgs_dir, data_list, transform=None, image_ftype="png", label_ftype="json"):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing list of images and labels for this dataset
            transform: optional transform to be applied on each sample. useful for augmentation.
        """
        with open(data_list) as f:
            names = json.load(f)

        image_names = []
        labels = []
        for fname in names:
            try:
                with open("%s/%s.%s"%(labels_dir, fname, label_ftype)) as f:
                    label = json.load(f)

                labels.append(label)
                image_names.append("%s/%s.%s"%(imgs_dir, fname, image_ftype))
            except Exception as e:
                print("ERROR: opening label file '{}': {}".format(fname, e))

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('L')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_names)
