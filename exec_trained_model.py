import os
import numpy as np
from modeling.models import deeplabv3plus_mobilenet, deeplabv3plus_resnet50
from parameters import Parameters
import torch
import torch.nn.functional as F
from PIL import Image
from dataloaders.datasets import LostAndFound, RoadAnomaly, cityscapes, cityscapes_fewer_classes
from torch.utils.data import DataLoader
import cv2

mode = 'resNet' #'resNet', 'mobileNet'
saved_folder = 'results_deeplab/{}'.format('resNet_LostAndFound') 
num_class = 8

dataset = 'LostAndFound' #LostAndFound, RoadAnomaly, cityscapes

if dataset == 'LostAndFound':
    dataset_folder = '/projects/kosecka/yimeng/Datasets/Lost_and_Found'
elif dataset == 'RoadAnomaly':
    dataset_folder = '/projects/kosecka/yimeng/Datasets/RoadAnomaly'
elif dataset == 'cityscapes':
    dataset_folder = 'data/cityscapes'

#====================================================== change the parameters============================================================
par = Parameters()
par.test_batch_size = 2

'''mobileNet
par.resume = 'run/cityscapes/deeplab_duq/experiment_1/checkpoint.pth.tar'
par.duq_model_output_size = 128
'''

#'''ResNet
par.resume = 'run/cityscapes/deeplab_resnet/experiment_0/checkpoint.pth.tar'
#'''

#=========================================================== Define Dataloader ==================================================
if dataset == 'LostAndFound':
    dataset_val = LostAndFound.LostAndFound(par, dataset_dir=dataset_folder)
    dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))
elif dataset == 'RoadAnomaly':
    dataset_val = RoadAnomaly.RoadAnomaly(par, dataset_dir=dataset_folder)
    dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))
elif dataset == 'cityscapes':
    dataset_val = cityscapes_fewer_classes.CityscapesDataset_fewer(par, dataset_dir=dataset_folder, split='val')
    dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))

#big_outlier_list = [2, 3, 4, 7, 10, 11, 15, 16, 25, 27, 31, 33, 34, 35, 38, 40, 45, 46, 48, 50, 51, 54, 57, 60, 61, 63, 65, 
#    68, 71, 72, 74, 76, 83, 84, 85, 86, 91, 93, 95]
big_outlier_list = [2, 4, 10, 16, 27, 38, 45, 50, 51, 61, 65, 68, 74, 76, 83, 84, 95]

#================================================================================================================================
# Define network
model = deeplabv3plus_resnet50(num_classes=num_class, output_stride=par.out_stride).cuda()

#===================================================== Resuming checkpoint ====================================================
if par.resume is not None:
    if not os.path.isfile(par.resume):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(par.resume))
    checkpoint = torch.load(par.resume)
    par.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(par.resume, checkpoint['epoch']))

#======================================================== evaluation stage =====================================================
    model.eval()
    count = 0
    for iter_num, sample in enumerate(dataloader_val):
        if dataset == 'cityscapes' and iter_num == 10:
            break
        print('iter_num = {}'.format(iter_num))
        images, targets = sample['image'], sample['label']
        #print('images = {}'.format(images.shape))
        #print('targets = {}'.format(targets))
        images = images.cuda()

        N, H, W = targets.shape
        input_shape = (int(H/2), int(W/2))

        #================================================ compute loss =============================================
        with torch.no_grad():
            output, _ = model(images) #output.shape = batch_size x num_classes x H x W
            output = output.data.cpu().numpy()
            print('output.shape = {}'.format(output.shape))

        for i in range(N):
            pred = output[i]
            class_pred = np.argmax(pred, axis=0).astype('uint8')
            print('class_pred.shape = {}'.format(class_pred.shape))
            class_pred = cv2.resize(class_pred, input_shape[::-1], cv2.INTER_NEAREST)
            
            result = {}
            result['sseg'] = class_pred
            #assert 1==2

            if dataset == 'LostAndFound' and count in big_outlier_list:
                np.save('{}/{}_result.npy'.format(saved_folder, count), result)
            if dataset == 'cityscapes':
                np.save('{}/{}_result.npy'.format(saved_folder, count), result)

            #assert 1==2
            count += 1



