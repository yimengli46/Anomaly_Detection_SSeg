import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloaders import custom_transforms as tr

class CityscapesDataset_fewer(data.Dataset):
    def __init__(self, par, dataset_dir, split='train'):

        self.dataset_dir = dataset_dir
        self.split = split
        self.par = par

        self.img_list = np.load('{}/{}_img_list.npy'.format(self.dataset_dir, self.split), allow_pickle=True).tolist()
        if split == 'train':
            self.img_list.extend(self.img_list)

        self.void_classes = [0, 1, 2, 3, 4, 5, 10, 14, 15, 16, -1]
        self.valid_classes = [7, 11, 17, 21, 23, 24, 26, 31]
        self.class_names = ['unlabelled', 'road', 'building', \
                            'pole', 'vegetation', 'sky', 'person', 'car', 'train', ]

        self.ignore_index = 255
        self.NUM_CLASSES = len(self.valid_classes)
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        print("Found {} {} images".format(len(self.img_list), self.split))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        img_path = '{}/{}'.format(self.dataset_dir, self.img_list[index]['rgb_path'])
        lbl_path = '{}/{}'.format(self.dataset_dir, self.img_list[index]['semSeg_path'])

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        #merge ambiguous classes
        mask[mask == 6] = 7 # ground -> road
        mask[mask == 8] = 7 # sidewalk -> road
        mask[mask == 9] = 7 # parking -> road
        mask[mask == 22] = 21 # terrain -> vegetation
        mask[mask == 25] = 24 # rider -> person
        mask[mask == 32] = 24 # motorcycle -> person
        mask[mask == 33] = 24 # bicycle -> person
        mask[mask == 27] = 26 # truck -> car
        mask[mask == 28] = 26 # bus -> car
        mask[mask == 29] = 26 # caravan -> car
        mask[mask == 30] = 26 # trailer -> car
        mask[mask == 12] = 11 # wall -> building
        mask[mask == 13] = 11 # fence -> building
        mask[mask == 19] = 17 # traffic light -> pole
        mask[mask == 20] = 17 # traffic sign -> pole
        mask[mask == 18] = 17 # pole group -> pole

        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomCrop(self.par.base_size, self.par.crop_size, fill=255),
            tr.RandomColorJitter(),
            tr.RandomHorizontalFlip(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(resize_ratio=self.par.resize_ratio),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(resize_ratio=self.par.resize_ratio),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

'''
if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = CityscapesSegmentation(args, split='train')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
'''
