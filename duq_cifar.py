import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import numpy as np
import sklearn.datasets

import matplotlib.pyplot as plt
import seaborn as sns

from utils.resnet_duq import ResNet_DUQ
from utils.datasets import all_datasets

l_gradient_penalty = 0.1

ds = all_datasets["CIFAR10"]()
input_size, num_classes, dataset, test_dataset = ds

print('num_classes = {}'.format(num_classes))

idx = list(range(len(dataset)))
val_size = int(len(dataset) * 0.8)
train_dataset = torch.utils.data.Subset(dataset, idx[:val_size])
val_dataset = torch.utils.data.Subset(dataset, idx[val_size:])

val_dataset.transform = (
    test_dataset.transform
)  # Test time preprocessing for validation

batch_size = 7
centroid_size = 512
model_output_size = 512
length_scale = 0.1
gamma = 0.999

model = ResNet_DUQ(input_size, num_classes, centroid_size, model_output_size, length_scale, gamma)
model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

def calc_gradients_input(x, y_pred):
    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]
    #print('A gradients.shape = {}'.format(gradients.shape))
    gradients = gradients.flatten(start_dim=1)
    #print('B gradients.shape = {}'.format(gradients.shape))
    return gradients

def calc_gradient_penalty(x, y_pred):
    #print('x.shape = {}'.format(x.shape))
    #print('y_pred.shape = {}'.format(y_pred.shape))
    gradients = calc_gradients_input(x, y_pred)
    #print('gradients = {}'.format(gradients))
    # L2 norm
    grad_norm = gradients.norm(2, dim=1)
    #print('D grad_norm.shape = {}'.format(grad_norm.shape))
    #print('grad_norm = {}'.format(grad_norm))

    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
    #print('E gradient_penalty.shape = {}'.format(gradient_penalty.shape))
    #print('gradient_penalty = {}'.format(gradient_penalty))

    return gradient_penalty

def bce_loss_fn(y_pred, y):
    bce = F.binary_cross_entropy(y_pred, y, reduction="sum").div(
        num_classes * y_pred.shape[0]
    )
    return bce

def step(batch):
    model.train()

    optimizer.zero_grad()

    x, y = batch
    x, y = x.cuda(), y.cuda()

    if l_gradient_penalty > 0:
        x.requires_grad_(True)

    z, y_pred = model(x)
    #print('y_pred = {}'.format(y_pred))
    #print('y = {}'.format(y))
    y = F.one_hot(y, num_classes).float()

    loss = bce_loss_fn(y_pred, y)
    print('loss = {:.5f}'.format(loss.item()))

    if l_gradient_penalty > 0:
        gradient_penalty = l_gradient_penalty * calc_gradient_penalty(x, y_pred)
        print('gradient_penalty = {:.5f}'.format(gradient_penalty))
        loss += gradient_penalty
    
    loss.backward()
    optimizer.step()

    x.requires_grad_(False)

    with torch.no_grad():
        model.eval()
        model.update_embeddings(x, y)

    return loss.item()


def eval_step(batch):
    model.eval()

    x, y = batch
    x, y = x.cuda(), y.cuda()

    x.requires_grad_(True)

    z, y_pred = model(x)

    return y_pred, y, x


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=False)


num_epochs = 30
for epoch_num in range(num_epochs):
    epoch_loss = []

    for iter_num, batch in enumerate(train_loader):
        #if iter_num > 10:
        #    break

        #images, targets = images.to(device), targets.to(device)
        #print('targets = {}'.format(targets))
        #assert 1==2
        
        #================================ compute loss ============================
        loss = step(batch)
        #assert 1==2
        #============================ print loss =====================================
        epoch_loss.append(loss)

        if iter_num % 10 == 0:
            print('Epoch: {} | Iteration: {} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, np.mean(epoch_loss)))






