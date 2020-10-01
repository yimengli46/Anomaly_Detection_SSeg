import os
import numpy as np
from modeling.models import deeplabv3plus_mobilenet, deeplabv3plus_resnet50
from parameters import Parameters
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from dataloaders.datasets import cityscapes, cityscapes_fewer_classes
from torch.utils.data import DataLoader
import cv2
import random
from utils.metrics import Evaluator

par = Parameters()
par.resume = 'run/cityscapes/deeplab_resnet/experiment_1/checkpoint.pth.tar'
saved_folder = 'run/cityscapes/duq'

#=========================================================== Define Dataloader ==================================================
dataset_train = cityscapes_fewer_classes.CityscapesDataset_fewer(par, dataset_dir='data/cityscapes', split='train')
num_class = dataset_train.NUM_CLASSES
dataloader_train = DataLoader(dataset_train, batch_size=par.batch_size, shuffle=True, num_workers=int(par.batch_size/2))

dataset_val = cityscapes_fewer_classes.CityscapesDataset_fewer(par, dataset_dir='data/cityscapes', split='val')
dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))

#================================================================================================================================
# Define network
deeplab_model = deeplabv3plus_resnet50(num_classes=num_class, output_stride=par.out_stride).cuda()
evaluator = Evaluator(num_class)
#===================================================== Resuming checkpoint ====================================================
if par.resume is not None:
    if not os.path.isfile(par.resume):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(par.resume))
    checkpoint = torch.load(par.resume)
    par.start_epoch = checkpoint['epoch']
    deeplab_model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(par.resume, checkpoint['epoch']))

deeplab_model.eval()
#======================================================== Training stage =====================================================

class Model_bilinear(nn.Module):
    def __init__(self, input_dim, model_output_size, num_classes, centroid_size, par): # num_classes equals num_classes
        super().__init__()
        
        self.gamma = par.duq_gamma
        self.sigma = par.duq_length_scale
        
        self.fc1 = nn.Linear(input_dim, model_output_size)
        self.fc2 = nn.Linear(model_output_size, model_output_size)
        
        # centroid_size is # of centroids
        self.W = nn.Parameter(torch.normal(torch.zeros(centroid_size, num_classes, model_output_size), 1))
        
        self.register_buffer('N', torch.ones(num_classes) * 20) # self.N.shape = torch.Size([2])
        self.register_buffer('m', torch.normal(torch.zeros(centroid_size, num_classes), 1)) # self.m.shape = torch.Size([10, 2])
        
        self.m = self.m * self.N.unsqueeze(0) # self.m.shape = torch.Size([10, 2])

    def embed(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # i is batch, m is centroid_size, n is num_classes (classes)
        x = torch.einsum('ij,mnj->imn', x, self.W)
        return x

    def bilinear(self, z):
        embeddings = self.m / self.N.unsqueeze(0) #embeddings.shape = torch.Size([10, 2])
        #print('embeddings.shape = {}'.format(embeddings.shape))
        # implement Eq (1) in the paper
        diff = z - embeddings.unsqueeze(0)            
        y_pred = (- diff**2).mean(1).div(2 * self.sigma**2).exp()
        return y_pred

    def forward(self, x):
        z = self.embed(x) # z: batch_size x num_centroids x num_classes, z.shape = torch.Size([64, 10, 2])
        y_pred = self.bilinear(z) # y_pred.shape = torch.Size([64, 2])
        return z, y_pred

    def update_embeddings(self, x, y):
        z = self.embed(x)
        # normalizing value per class, assumes y is one_hot encoded
        # implement Eq (4)
        self.N = torch.max(self.gamma * self.N + (1 - self.gamma) * y.sum(0), torch.ones_like(self.N))
        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum('ijk,ik->jk', z, y)
        # implement Eq (5)
        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum

def calc_gradient_penalty(x, y_pred):
    gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

    gradients = gradients.flatten(start_dim=1)
    # L2 norm
    grad_norm = gradients.norm(2, dim=1)
    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
    # One sided penalty - down
#     gradient_penalty = F.relu(grad_norm - 1).mean()
    return gradient_penalty

#=====================================================================================================================
model = Model_bilinear(256, par.duq_model_output_size, num_class, par.duq_centroid_size, par).cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=par.lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)

#========================================================================================================================
best_pred = 0.0

for epoch in range(par.epochs):
    train_loss = 0.0

    num_img_tr = len(dataloader_train)
    
    for iter_num, sample in enumerate(dataloader_train):
        print('epoch = {}, iter_num = {}'.format(epoch, iter_num))
        #if iter_num > 10:
        #    break

        model.train()
        optimizer.zero_grad()

        images, targets = sample['image'], sample['label']
        #print('images = {}'.format(images.shape))
        #print('targets = {}'.format(targets.shape))
        images, targets = images.cuda(), targets.cuda()

        #print('targets.shape = {}'.format(targets.shape))
        targets = targets.reshape(-1, 1).long().squeeze(1)
        idx_unignored = (targets < 255)
        targets = targets[idx_unignored]
        targets = F.one_hot(targets, num_class).float()
        #print('targets.shape = {}'.format(targets.shape))
        # random pick a portion of positions
        N, _ = targets.shape
        indice = random.sample(range(N), int(par.duq_batch_p * N))
        indice = torch.tensor(indice).cuda()
        targets = targets[indice]
        #print('targets.shape = {}'.format(targets.shape))

        with torch.no_grad():
            _, z, _ = deeplab_model(images) #output.shape = batch_size x num_classes x H x W
            #print('z.shape = {}'.format(z.shape))
            H, W = images.shape[-2:]
            input_shape = (H, W)
            z_interpolated = F.interpolate(z, size=input_shape, mode='bilinear', align_corners=False)
            #print('interpolated z.shape = {}'.format(z_interpolated.shape))

        _, C, _, _ = z_interpolated.shape
        z_interpolated = z_interpolated.permute(0, 2, 3, 1).reshape(-1, C)
        z_interpolated = z_interpolated[idx_unignored]
        z_interpolated = z_interpolated[indice]
        #print('interpolated z.shape = {}'.format(z_interpolated.shape))
        #assert 1==2
        if par.duq_l_gradient_penalty > 0.0:
            z_interpolated.requires_grad_(True)
        
        #================================================ compute loss =============================================
        _, y_pred = model(z_interpolated)
        #print('y_pred.shape = {}'.format(y_pred.shape))

        loss1 =  F.binary_cross_entropy(y_pred, targets)

        if par.duq_l_gradient_penalty > 0.0:
            gradient_penalty = par.duq_l_gradient_penalty * calc_gradient_penalty(z_interpolated, y_pred)
            
        loss = loss1 + gradient_penalty
        print('loss1 = {:.4f}, gradient_penalty = {:.4f}'.format(loss1, gradient_penalty))

        #================================================= compute gradient =================================================
        loss.backward()
        optimizer.step()

        z_interpolated.requires_grad_(False)

        with torch.no_grad():
            model.eval()
            model.update_embeddings(z_interpolated, targets)

        #================================================= print something ==============================================
        train_loss += loss.item()
        print('Train loss: %.3f' % (train_loss / (iter_num + 1)))

        # Show 10 * 3 inference results each epoch
        if iter_num % (num_img_tr // 10) == 0:
            global_step = iter_num + num_img_tr * epoch

    print('[Epoch: %d, numImages: %5d]' % (epoch, iter_num * par.batch_size + images.data.shape[0]))
    print('*********Loss: %.3f' % train_loss)

#=============================================== eval stage ===========================================================

    if epoch % par.eval_interval == 0:
        model.eval()
        evaluator.reset()
        test_loss = 0.0
        for iter_num, sample in enumerate(dataloader_val):
            print('epoch = {}, iter_num = {}'.format(epoch, iter_num))
            images, targets = sample['image'], sample['label']
            #print('images = {}'.format(images))
            #print('targets = {}'.format(targets))
            images, targets = images.cuda(), targets.cuda()
            N, _, H, W = images.shape

            with torch.no_grad():
                _, z, _ = deeplab_model(images) #output.shape = batch_size x num_classes x H x W
                #print('z.shape = {}'.format(z.shape))
                input_shape = (H, W)
                z_interpolated = F.interpolate(z, size=input_shape, mode='bilinear', align_corners=False)
                #print('interpolated z.shape = {}'.format(z_interpolated.shape))

                _, C, _, _ = z_interpolated.shape
                z_interpolated = z_interpolated.permute(0, 2, 3, 1).reshape(-1, C)

                #================================================ compute loss =============================================
                n_rows, _ = z_interpolated.shape
                _, y_pred_0 = model(z_interpolated[:int(n_rows/4)])
                _, y_pred_1 = model(z_interpolated[int(n_rows/4):int(n_rows/2)])
                _, y_pred_2 = model(z_interpolated[int(n_rows/2):int(n_rows/4*3)])
                _, y_pred_3 = model(z_interpolated[int(n_rows/4*3):int(n_rows)])

                y_pred = torch.cat((y_pred_0, y_pred_1, y_pred_2, y_pred_3), 0)
                #print('y_pred.shape = {}'.format(y_pred.shape))

                y_pred = y_pred.reshape(N, H, W, num_class).permute(0, 3, 1, 2)
                #print('y_pred.shape = {}'.format(y_pred.shape))

            pred = y_pred.data.cpu().numpy()
            targets = targets.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(targets, pred)

        # Fast test during the training
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, iter_num * par.batch_size + images.data.shape[0]))
        print("Acc:{:.5}, Acc_class:{:.5}, mIoU:{:.5}, fwIoU: {:.5}".format(Acc, Acc_class, mIoU, FWIoU))

        new_pred = mIoU
        if new_pred > best_pred:
            best_pred = new_pred
            print('updating best pred: {:.5}'.format(best_pred))
            torch.save(model.state_dict(), '{}/best_duq.pth'.format(saved_folder))

    torch.save(model.state_dict(), '{}/duq.pth'.format(saved_folder))
    scheduler.step(train_loss)





