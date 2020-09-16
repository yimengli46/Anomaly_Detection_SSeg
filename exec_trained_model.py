import os
import numpy as np
from modeling.models import deeplabv3plus_duq_mobilenet, deeplabv3plus_duq_resnet50
from parameters import Parameters
import torch
import torch.nn.functional as F
from PIL import Image
from dataloaders.datasets import LostAndFound, RoadAnomaly
from torch.utils.data import DataLoader

mode = 'resNet' #'resNet', 'mobileNet'
saved_folder = 'results/{}'.format('resNet_lostAndFound') #'results/resNet_lostAndFound_2', 'mobileNet_lostAndFound', mobileNet_roadAnomaly
num_class = 19

dataset = 'LostAndFound' #LostAndFound, RoadAnomaly

if dataset == 'LostAndFound':
    dataset_folder = '/projects/kosecka/yimeng/Datasets/Lost_and_Found'
elif dataset == 'RoadAnomaly':
    dataset_folder = '/projects/kosecka/yimeng/Datasets/RoadAnomaly'

#====================================================== change the parameters============================================================
par = Parameters()
par.test_batch_size = 2

'''mobileNet
par.resume = 'run/cityscapes/deeplab_duq/experiment_1/checkpoint.pth.tar'
par.duq_model_output_size = 128
'''

#'''ResNet
par.resume = 'run/cityscapes/deeplab_duq/experiment_4/checkpoint.pth.tar'
par.duq_model_output_size = 64
#'''

#=========================================================== Define Dataloader ==================================================
if dataset == 'LostAndFound':
    dataset_val = LostAndFound.LostAndFound(par, dataset_dir=dataset_folder)
    dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))
elif dataset == 'RoadAnomaly':
    dataset_val = RoadAnomaly.RoadAnomaly(par, dataset_dir=dataset_folder)
    dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))

#================================================================================================================================
# Define network
if mode == 'mobileNet':
    model = deeplabv3plus_duq_mobilenet(num_classes=num_class, output_stride=par.out_stride, par=par).cuda()
elif mode == 'resNet':
    model = deeplabv3plus_duq_resnet50(num_classes=num_class, output_stride=par.out_stride, par=par).cuda()

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
        print('iter_num = {}'.format(iter_num))
        images, targets = sample['image'], sample['label']
        #print('images = {}'.format(images.shape))
        #print('targets = {}'.format(targets))
        images = images.cuda()

        #================================================ compute loss =============================================
        with torch.no_grad():
            output, _, _ = model(images) #output.shape = batch_size x num_classes x H x W
            output_copy = output.data.cpu().numpy()

        N, _, _, _ = output_copy.shape
        for i in range(N):
            pred = output_copy[i]
            class_prediction = np.argmax(pred, axis=0)
            uncertainty_mat = np.amax(pred, axis=0)

            pred = np.sort(pred, axis=0)
            print('pred.shape = {}'.format(pred.shape))

            result = {}
            result['sseg'] = class_prediction
            result['uncertainty'] = uncertainty_mat
            #result['all_pred'] = pred
            np.save('{}/{}_result.npy'.format(saved_folder, count), result)
            count += 1



