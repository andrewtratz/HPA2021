import re

from layers.backbone.efficientnet import *
from layers.loss import *

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

## networks  ######################################################################
class EfficientNetClass(nn.Module):
    def load_pretrain(self, pretrain_file):
        return None

    def freeze_backbone(self):
        for child in self.children():
            for param in child.parameters():
                param.requires_grad = False
        # Don't freeze final layer
        for param in self.logit.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for child in self.children():
            for param in child.parameters():
                param.requires_grad = True

    def __init__(self,feature_net='efficientnet-b0', num_classes=6, num_features=1792,
                 in_channels=3,
                 pretrained_file=None,
                 dropout=False,
                 large=False,
                 ):
        super().__init__()
        self.dropout = dropout
        self.in_channels = in_channels
        self.large = large
        self.puzzle = False

        self.mean = []
        self.std = []
        self.dummy_data = torch.tensor([0]).cuda()

        for i in range(0, torch.cuda.device_count()):
            mean = torch.FloatTensor([0.074598, 0.050630, 0.050891, 0.076287]).cuda(device=i)#rgby
            std = torch.FloatTensor([0.122813, 0.085745, 0.129882, 0.119411]).cuda(device=i)
            self.mean.append(mean)
            self.std.append(std)

        #for i in range(0, torch.cuda.device_count()):
            #mean = torch.FloatTensor([0.074598, 0.050630, 0.050891]).cuda() #rgb
            #std = torch.FloatTensor([0.122813, 0.085745, 0.129882]).cuda()
            #mean = torch.FloatTensor([0.06743479]).to('cuda:' + str(i))  # rgb
            #std = torch.FloatTensor([0.11647667]).to('cuda:' + str(i))
            #self.mean.append(mean)
            #self.std.append(std)

        self.bn1 = nn.BatchNorm1d(num_features * 2).cuda()
        self.fc1 = nn.Linear(num_features * 2, num_features).cuda()
        self.bn2 = nn.BatchNorm1d(num_features).cuda()
        self.relu = nn.ReLU(inplace=True).cuda()
        self.logit = nn.Linear(num_features, num_classes).cuda()

        if feature_net=='efficientnet-b0' or feature_net == 'efficientnet-b4':
            self.backbone = EfficientNetHPA(feature_net=feature_net, in_channels=in_channels, num_classes=num_classes, dropout=False, pretrained_file=pretrained_file).cuda()

    def add_puzzle(self):
        assert(False)

    def forward(self, x):
        gpu = x.device.index

        for i in range(self.in_channels):
            x[:,i,:,:] = (x[:,i,:,:] - self.mean[gpu][i]) / self.std[gpu][i]

        x = self.backbone.model.extract_features(x)

        x = torch.cat((nn.AdaptiveAvgPool2d(1)(x), nn.AdaptiveMaxPool2d(1)(x)), dim=1)
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.logit(x)

        if self.puzzle:
            return x, self.puzzle_features
        else:
            return x, self.dummy_data

def class_efficientnet_b0(**kwargs):
    num_classes = kwargs['num_classes']
    in_channels = kwargs['in_channels']
    pretrained_file = kwargs['pretrained_file']
    model = EfficientNetClass(feature_net='efficientnet-b0', num_classes=num_classes,
                        in_channels=in_channels, pretrained_file=pretrained_file, dropout=True)
    return model

def class_efficientnet_b4(**kwargs):
    num_classes = kwargs['num_classes']
    in_channels = kwargs['in_channels']
    pretrained_file = kwargs['pretrained_file']
    model = EfficientNetClass(feature_net='efficientnet-b4', num_classes=num_classes,
                        in_channels=in_channels, pretrained_file=pretrained_file, dropout=True)
    return model