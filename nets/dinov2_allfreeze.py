import torch
import torch.nn as nn
from domainlab.compos.nn_zoo.nn_torchvision import NetTorchVisionBase


DINO_PATH_FINETUNED_DOWNLOADED='/lustre/groups/aih/sina.wendrich/MA_code/test/dinov2_vits_student_TCGA'

def get_dino_finetuned_downloaded():
    # load the original DINOv2 model with the correct architecture and parameters. The positional embedding is too large.
    # load vits or vitg
    architecture = 'vits'  # or 'vits' based on your needs
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    #model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    # load finetuned weights
    pretrained = torch.load(DINO_PATH_FINETUNED_DOWNLOADED, map_location=torch.device('cpu'))
    # make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['student'].items():
        if 'dino_head' in key:
            print('not used')
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
    #change shape of pos_embed, shape depending on vits or vitg
    pos_embed_shape = (1, 257, 384) if architecture == 'vits' else (1, 257, 1536)
    pos_embed = nn.Parameter(torch.zeros(pos_embed_shape))

    #pos_embed = nn.Parameter(torch.zeros(1, 257, 1536))
    model.pos_embed = pos_embed
    # load state dict
    model.load_state_dict(new_state_dict, strict=True)
    return model


class DINOv2Base(NetTorchVisionBase):
    def __init__(self, flag_finetuned, remove_last_layer=False):
        super().__init__(flag_finetuned)
        self.remove_last_layer = remove_last_layer
        self.flag_finetuned = flag_finetuned
    def fetch_net(self, flag_finetuned):
        #change to flag_finetuned?
        if flag_finetuned:
            self.net_torchvision = get_dino_finetuned_downloaded()
            #print(self.net_torchvision)
        else:
            # initializing a DINOv2 model with pretrained weights(without finetuning).
            self.net_torchvision = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

            #self.net_torchvision = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    def forward(self, tensor, layers=1, remove_last_layer=False):
        if layers == 1:
            # Utilize only the features from the last layer.
            x = self.net_torchvision.forward_features(tensor)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            # fmt: off
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)        
        else:
            # Utilize features from multiple layers.
            x = self.net_torchvision.get_intermediate_layers(tensor, n=layers, return_class_token=True)
            #linear_input = torch.cat([cls_token for _, cls_token in x] + [x[-1][0].mean(dim=1)], dim=1)
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
        if remove_last_layer:
            #ohne classification head auszuführen
            return linear_input
        else:
            #mit classification head
            return self.net_torchvision.head(linear_input)



class DINOv2ForClassification(DINOv2Base):
    """
    Adapts DINOv2 for classification tasks.
    """
    def __init__(self, flag_finetuned, dim_y, remove_last_layer=False):
        super().__init__(flag_finetuned, remove_last_layer=remove_last_layer)
        embed_dim = 384 # 1536 for vitlarge else  384
        self.net_torchvision.head = self._make_dinov2_linear_classification_head(dim_y, embed_dim, layers= 1)

        # Freeze all existing parameters in the model
        for param in self.net_torchvision.parameters():
            param.requires_grad = False

        # Unfreeze the new classification head
        for param in self.net_torchvision.head.parameters():
            param.requires_grad = True

    def _make_dinov2_linear_classification_head(self,
        dim_y,
        embed_dim: int = 384,
        layers: int = 1
    ):
        linear_head = nn.Linear((1 + layers) * embed_dim, dim_y)

        return linear_head
        


#TODO:Find out where remove_last_layer is set
def build_feat_extract_net(dim_y, remove_last_layer):
    return DINOv2ForClassification(flag_finetuned=False, dim_y=dim_y, remove_last_layer= True)
