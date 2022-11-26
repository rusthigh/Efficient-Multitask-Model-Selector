import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(txtnames, datadir, class_to_idx):
    images = []
    labels = []
    for txtname in txtnames:
        with open(txtname, 'r') as lines:
            for line in lines:
                classname = line.split('/')[0]
                _img = os.path.join(datadir, 'images', line.strip())
                print(_img)
                assert os.path.isfile(_img)
                images.append(_img)
                labels.append(class_to_idx[classname])

    return images, labels


class DTD(data.Dataset):
    # datasets.dtd.DTD ./data/DTD train
    def __init__(self, root, split, transform=None, target_transform=None, download=None):
        classes, class_to_idx = find_classes(os.path.join(root, 'images'))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.partition = '1'

        filename = [os.path.j