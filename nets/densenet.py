import torch.nn as nn
from torchvision import models as torchvisionmodels
from torchvision.models import DenseNet121_Weights

from domainlab.compos.nn_zoo.nn import LayerId
from domainlab.compos.nn_zoo.nn_torchvision import NetTorchVisionBase


class DenseNetBase(NetTorchVisionBase):
    """
    Since DenseNet can be fetched from torchvision
    """

    def fetch_net(self, flag_pretrain):
        """fetch_net.

        :param flag_pretrain:
        """
        if flag_pretrain:
            self.net_torchvision = torchvisionmodels.densenet.densenet121(
                weights=DenseNet121_Weights.IMAGENET1K_V1
            )
        else:
            self.net_torchvision = torchvisionmodels.densenet.densenet121(weights=None)
        # CHANGEME: user can modify this line to choose other neural
        # network architectures from 'torchvision.models'


class DenseNet4DeepAll(DenseNetBase):
    """
    Change the size of the last layer
    """

    def __init__(self, flag_pretrain, dim_y):
        """__init__.

        :param flag_pretrain:
        :param dim_y:
        """
        super().__init__(flag_pretrain)
        num_final_in = self.net_torchvision.classifier.in_features
        self.net_torchvision.classifier = nn.Linear(num_final_in, dim_y)
        # CHANGEME: user should change "classifier" to their chosen neural
        # network's last layer's name


class DenseNetNoLastLayer(DenseNetBase):
    """DenseNetNoLastLayer."""

    def __init__(self, flag_pretrain):
        """__init__.

        :param flag_pretrain:
        """
        super().__init__(flag_pretrain)
        self.net_torchvision.classifier = LayerId()
        # CHANGEME: user should change "classifier" to their chosen neural
        # network's last layer's name


# CHANGEME: user is required to implement the following function
# with **exact** signature to return a neural network architecture for
# classification of dim_y number of classes if remove_last_layer=False
# or return the same neural network without the last layer if
# remove_last_layer=False.
def build_feat_extract_net(dim_y, remove_last_layer):
    """
    This function is compulsory to return a neural network feature extractor.
    :param dim_y: number of classes to be classified, can be None
    if remove_last_layer = True
    :param remove_last_layer: for densenet for example, whether
    to remove the last layer or not.
    """
    if remove_last_layer:
        return DenseNetNoLastLayer(flag_pretrain=False)
    return DenseNet4DeepAll(flag_pretrain=False, dim_y=dim_y)
