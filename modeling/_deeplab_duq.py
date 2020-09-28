import torch
from torch import nn
from torch.nn import functional as F
from ._deeplab import ASPP
import torchvision.transforms as transforms

class DeepLabHeadV3Plus_duq(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36], par=None):
        super(DeepLabHeadV3Plus_duq, self).__init__()

        assert par != None
        self.num_classes= num_classes

        self.project = nn.Sequential( 
            nn.utils.spectral_norm(nn.Conv2d(low_level_channels, 48, 1, bias=False)),
            #nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(304, 256, 3, padding=1, bias=False)),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, par.duq_model_output_size, 1)
        )
        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

#========================================================== duq variables =======================================================
        self.gamma = par.duq_gamma

        self.W = nn.Parameter(
            torch.zeros(par.duq_centroid_size, num_classes, par.duq_model_output_size)
        )
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        self.register_buffer('N', torch.ones(num_classes)*par.duq_model_output_size)
        self.register_buffer(
            'm', torch.normal(torch.zeros(par.duq_centroid_size, num_classes), 0.05)
        )
        self.m = self.m * self.N

        self.sigma = par.duq_length_scale

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

        return diff

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        #print('feature[out].shape = {}'.format(feature['out'].shape))
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        temp_z = torch.cat([low_level_feature, output_feature], dim=1)
        z = self.classifier(temp_z) # z.shape = batch_size x 256 x 192 x 192
        #print('z.shape = {}'.format(z.shape))
        #assert 1==2

        N, FEATURE_SIZE, H, W = z.shape
        z = z.permute(0, 2, 3, 1) # z.shape = batch_size x 192 x 192 x 256
        z = z.reshape(-1, FEATURE_SIZE) #z.shape = (batch_size x h x w) x 256 = 110592 x 256
        #print('z.shape = {}'.format(z.shape))
        #assert 1==2

        y_pred = self.rbf(z) #y.shape = (batch_size x h x w) x num_classes = 110592 x 19
        #print('y_pred.shape = {}'.format(y_pred.shape))
        #assert 1==2

        # reshape z and y_pred
        y_pred = y_pred.reshape(N, H, W, -1)
        assert y_pred.shape[3] == self.num_classes
        y_pred = y_pred.permute(0, 3, 1, 2) #y_pred.shape = batch_size x num_classes x 192 x 192
        #print('y_pred.shape = {}'.format(y_pred.shape))

        # z is not necessary
        #'''
        z = z.reshape(N, H, W, -1)
        assert z.shape[3] == FEATURE_SIZE
        z = z.permute(0, 3, 1, 2) #z.shape = batch_size x 256 x 192 x 192
        #print('z.shape = {}'.format(z.shape))
        #'''

        return y_pred, z

#'''
    def update_embeddings(self, feature, y_targets):

        #input y_targets.shape = batch_size x 192 x 192, dtype=long
        #print('y_targets.shape = {}'.format(y_targets.shape))
        y_targets = y_targets.reshape(-1, 1).long().squeeze(1) #y_targets.shape = (batch_size x 192 x 192) x 1
        idx_unignored = (y_targets < 255)
        y_targets = y_targets[idx_unignored]
        y_targets = F.one_hot(y_targets, self.num_classes).float() # y_targets.shape = (?) x num_classes
        #print('y_targets.shape = {}'.format(y_targets.shape)) 

        self.N = self.gamma * self.N + (1 - self.gamma) * y_targets.sum(0) # Eq. 4

        # get features of each datapoint and name the variable z
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        z = self.classifier(torch.cat([low_level_feature, output_feature], dim=1)) # z.shape = batch_size x 256 x 192 x 192

        # reshape z
        N, FEATURE_SIZE, H, W = z.shape
        z = z.permute(0, 2, 3, 1) # z.shape = batch_size x 192 x 192 x 256
        #print('z.shape = {}'.format(z.shape))
        z = z.reshape(-1, FEATURE_SIZE) #z.shape = (batch_size x h x w) x 256 = (batch_size x 192 x 192) x 256
        z = z[idx_unignored]
        #print('z.shape = {}'.format(z.shape))

        z = torch.einsum("ij,mnj->imn", z, self.W) #z.shape = (batch_size x 192 x 192) x centroid_size x num_classes
        #print('z after einsum.shape = {}'.format(z.shape))
        embedding_sum = torch.einsum("ijk,ik->jk", z, y_targets)

        self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum
#'''