import torch.nn as nn
import timm 
from domainlab.compos.nn_zoo.nn_torchvision import NetTorchVisionBase

class ViTBase(NetTorchVisionBase):
  
    def fetch_net(self, flag_pretrain):
        model_name = "vit_base_patch16_224"
        #model_name = "swin_tiny_patch4_window7_224"

        #model_name = "vit_large_patch16_224"
        if flag_pretrain:
            self.net_torchvision = timm.create_model(model_name, pretrained=True)
            print(self.net_torchvision)
        else:
            self.net_torchvision = timm.create_model(model_name, pretrained=False)
            print(self.net_torchvision)

class ViTForClassification(ViTBase):
    """
    Adapt Vision Transformer for classification
    """
    def __init__(self, flag_pretrain, dim_y):
        """Initialize the Vision Transformer for classification.
        -param dim_y: Number of output classes.
        """
        super().__init__(flag_pretrain)
        num_final_in = self.net_torchvision.head.in_features
        self.net_torchvision.head = nn.Linear(num_final_in, dim_y) 
         # Freeze all the parameters of the model
        
def build_feat_extract_net(dim_y, remove_last_layer=True):
    """
    -param dim_y: Number of classes. If None, feature extractor is returned without classification head.
    """
    return ViTForClassification(flag_pretrain=True, dim_y=dim_y)