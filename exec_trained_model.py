import os
import numpy as np
from modeling.models import deeplabv3plus_duq_mobilenet, deeplabv3plus_duq_resnet50
from parameters import Parameters
import torch
import torch.nn.functional as F
from PIL import Image
from dataloaders.datasets import LostAndFound, RoadAnomaly, Fishyscapes
from torch.utils.data import DataLoader

dataset = 'fishyscapes' # lostAndFound, roadAnomaly, fishyscapes
saved_folder = 'results_duq/resNet_{}'.format(dataset) # lostAndFound, roadAnomaly, fishyscapes
num_class = 8

if dataset == 'lostAndFound':
    dataset_folder = '/projects/kosecka/yimeng/Datasets/Lost_and_Found'
elif dataset == 'roadAnomaly':
    dataset_folder = '/projects/kosecka/yimeng/Datasets/RoadAnomaly'
elif dataset == 'fishyscapes':
    dataset_folder = '/projects/kosecka/yimeng/Datasets/Fishyscapes_Static'

#====================================================== change the parameters============================================================
par = Parameters()
par.test_batch_size = 2

#'''ResNet
par.resume = 'run/cityscapes/deeplab_duq/experiment_0/checkpoint.pth.tar'
#'''

#=========================================================== Define Dataloader ==================================================
if dataset == 'lostAndFound':
    dataset_val = LostAndFound.LostAndFound(par, dataset_dir=dataset_folder)
    dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))
elif dataset == 'roadAnomaly':
    dataset_val = RoadAnomaly.RoadAnomaly(par, dataset_dir=dataset_folder)
    dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))
elif dataset == 'fishyscapes':
    dataset_val = Fishyscapes.Fishyscapes(par, dataset_dir=dataset_folder)
    dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))

#================================================================================================================================
# Define network
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

            # 1 means large uncertainty
            uncertainty_mat = 1 - uncertainty_mat

            #assert 1==2

            result = {}
            result['sseg'] = class_prediction
            result['uncertainty'] = uncertainty_mat
            
            np.save('{}/{}_result.npy'.format(saved_folder, count), result)
            count += 1

            #assert 1==2

