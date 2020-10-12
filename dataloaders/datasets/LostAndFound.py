import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloaders import custom_transforms as tr

class LostAndFound(data.Dataset):
    def __init__(self, par, dataset_dir):
        self.dataset_dir = dataset_dir
        self.par = par
        self.img_list = [x for x in range(100)]
        print("Found {} images".format(len(self.img_list)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = '{}/{}.png'.format(self.dataset_dir, index)
        lbl_path = '{}/{}_label.png'.format(self.dataset_dir, index)

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _target = Image.fromarray(_tmp)
        sample = {'image': _img, 'label': _target}
        return self.transform_ts(sample)

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(resize_ratio=self.par.resize_ratio),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

