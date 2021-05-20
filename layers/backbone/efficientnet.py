import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from efficientnet_pytorch import EfficientNet

class EfficientNetHPA(nn.Module):
    def __init__(self, feature_net, in_channels, num_classes, pretrained_file, dropout):
        super(EfficientNetHPA, self).__init__()

        if pretrained_file is not None:
            self.model = EfficientNet.from_pretrained(feature_net, weights_path=pretrained_file, in_channels=in_channels, num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name(feature_net)

        #self.model.to(dtype=torch.half)
        # Unfreeze model weights
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.model(x)
        return out

def class_efficientnetb0(**kwargs):
    num_classes = kwargs['num_classes']
    in_channels = kwargs['in_channels']
    pretrained_file = kwargs['pretrained_file']
    model = EfficientNetHPA(feature_net='efficientnet-b0', num_classes=num_classes,
                        in_channels=in_channels, pretrained_file=pretrained_file, dropout=True)
    return model

def class_efficientnetb4(**kwargs):
    num_classes = kwargs['num_classes']
    in_channels = kwargs['in_channels']
    pretrained_file = kwargs['pretrained_file']
    model = EfficientNetHPA(feature_net='efficientnet-b4', num_classes=num_classes,
                        in_channels=in_channels, pretrained_file=pretrained_file, dropout=True)
    return model