import os
import numpy as np
from modeling.models import deeplabv3plus_mobilenet, deeplabv3plus_resnet50
from parameters import Parameters
import torch
import torch.nn.functional as F
from PIL import Image
from dataloaders.datasets import LostAndFound, RoadAnomaly, cityscapes
from torch.utils.data import DataLoader

mode = 'resNet' #'resNet', 'mobileNet'
saved_folder = 'results_deeplab/{}'.format('resNet_features') 
num_class = 19

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
par.resume = 'run/cityscapes/deeplab_resnet/experiment_6/checkpoint.pth.tar'
#'''

#=========================================================== Define Dataloader ==================================================
if dataset == 'LostAndFound':
    dataset_val = LostAndFound.LostAndFound(par, dataset_dir=dataset_folder)
    dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))
elif dataset == 'RoadAnomaly':
    dataset_val = RoadAnomaly.RoadAnomaly(par, dataset_dir=dataset_folder)
    dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))
elif dataset == 'cityscapes':
    dataset_val = cityscapes.CityscapesDataset(par, dataset_dir=dataset_folder, split='val')
    dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))

big_outlier_list = [2, 3, 4, 7, 10, 11, 15, 16, 25, 27, 31, 33, 34, 35, 38, 40, 45, 46, 48, 50, 51, 54, 57, 60, 61, 63, 65, 
    68, 71, 72, 74, 76, 83, 84, 85, 86, 91, 93, 95]

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
        if dataset == 'cityscapes' and iter_num == 5:
            break
        print('iter_num = {}'.format(iter_num))
        images, targets = sample['image'], sample['label']
        #print('images = {}'.format(images.shape))
        #print('targets = {}'.format(targets))
        images = images.cuda()

        #================================================ compute loss =============================================
        with torch.no_grad():
            output, z = model(images) #output.shape = batch_size x num_classes x H x W
            print('z.shape = {}'.format(z.shape))
            input_shape = targets.shape[-2:]
            z_interpolated = F.interpolate(z, size=input_shape, mode='bilinear', align_corners=False)
            print('interpolated z.shape = {}'.format(z_interpolated.shape))
            
            z_interpolated = z_interpolated.data.cpu().numpy()
            targets = targets.numpy().astype(np.uint8)
            #assert 1==2

        N, _, _, _ = z.shape
        for i in range(N):
            result = {}
            result['feature'] = z_interpolated[i]
            result['label'] = targets[i]

            if dataset == 'LostAndFound' and count in big_outlier_list:
                np.save('{}/{}_{}.npy'.format(saved_folder, count, dataset), result)
            else:
                np.save('{}/{}_{}.npy'.format(saved_folder, count, dataset), result)

            #assert 1==2
            count += 1



