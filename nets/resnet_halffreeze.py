import torch.nn as nn
from torchvision import models as torchvisionmodels
from torchvision.models import ResNet50_Weights

from domainlab.compos.nn_zoo.nn import LayerId
from domainlab.compos.nn_zoo.nn_torchvision import NetTorchVisionBase


class ResNetBase(NetTorchVisionBase):
    """
    Since ResNet can be fetched from torchvision
    """

    def fetch_net(self, flag_pretrain):
        """fetch_net.

        :param flag_pretrain:
        """
        if flag_pretrain:
            self.net_torchvision = torchvisionmodels.resnet.resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V2
            )
        else:
            self.net_torchvision = torchvisionmodels.resnet.resnet50(weights="None")
        # CHANGEME: user can modify this line to choose other neural
        # network architectures from 'torchvision.models'


class ResNet4DeepAll(ResNetBase):
    """
    Change the size of the last layer and freeze specified layers
    """

    def __init__(self, flag_pretrain, dim_y):
        """__init__.

        :param flag_pretrain:
        :param dim_y:
        """
        super().__init__(flag_pretrain)
        num_final_in = self.net_torchvision.fc.in_features
        self.net_torchvision.fc = nn.Linear(num_final_in, dim_y)
        
        # Freeze all layers first
        for param in self.net_torchvision.parameters():
            param.requires_grad = False

        # Unfreeze the last half of the layers
        self.unfreeze_last_half_layers()

    def unfreeze_last_half_layers(self):
        layers_to_unfreeze = ['layer3', 'layer4', 'fc']
        for layer_name in layers_to_unfreeze:
            layer = getattr(self.net_torchvision, layer_name)
            for param in layer.parameters():
                param.requires_grad = True

class ResNetNoLastLayer(ResNetBase):
    """ResNetNoLastLayer."""

    def __init__(self, flag_pretrain):
        """__init__.

        :param flag_pretrain:
        """
        super().__init__(flag_pretrain)
        self.net_torchvision.fc = nn.Identity()

# Function to build the feature extraction network
def build_feat_extract_net(dim_y, remove_last_layer):
    """
    This function is compulsory to return a neural network feature extractor.
    :param dim_y: number of classes to be classified, can be None
    if remove_last_layer = True
    :param remove_last_layer: for resnet for example, whether
    remove the last layer or not.
    """
    if remove_last_layer:
        return ResNetNoLastLayer(flag_pretrain=True)
    return ResNet4DeepAll(flag_pretrain=True, dim_y=dim_y)

# Example usage
# model = build_feat_extract_net(dim_y=1000, remove_last_layer=False)
# for name, param in model.named_parameters():
#     print(name, param.requires_grad)
