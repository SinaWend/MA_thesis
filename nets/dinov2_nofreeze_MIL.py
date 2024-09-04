import torch
import torch.nn as nn
from domainlab.compos.nn_zoo.nn_torchvision import NetTorchVisionBase

DINO_PATH_FINETUNED_DOWNLOADED = '/lustre/groups/aih/sina.wendrich/MA_code/test/dinov2_vits_student_TCGA'

def get_dino_finetuned_downloaded():
    architecture = 'vits'
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    pretrained = torch.load(DINO_PATH_FINETUNED_DOWNLOADED, map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in pretrained['student'].items():
        if 'dino_head' not in key:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
    pos_embed_shape = (1, 257, 384) if architecture == 'vits' else (1, 257, 1536)
    model.pos_embed = nn.Parameter(torch.zeros(pos_embed_shape))
    model.load_state_dict(new_state_dict, strict=True)
    return model

class DINOv2Base(NetTorchVisionBase):
    def __init__(self, flag_finetuned, remove_last_layer=False):
        super().__init__(flag_finetuned)
        self.remove_last_layer = remove_last_layer
        self.flag_finetuned = flag_finetuned

    def fetch_net(self, flag_finetuned):
        self.net_torchvision = get_dino_finetuned_downloaded() if flag_finetuned else torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    def forward(self, tensor, layers=1, remove_last_layer=False):
        result = self.net_torchvision.forward_features(tensor)
        cls_token = result["x_norm_clstoken"]
        patch_tokens = result["x_norm_patchtokens"]
        return cls_token, patch_tokens

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)
        weighted_features = attention_weights * x
        aggregated_features = weighted_features.sum(1)
        return aggregated_features

class DINOv2ForMIL(DINOv2Base):
    def __init__(self, flag_finetuned, dim_y):
        super().__init__(flag_finetuned)
        self.pooling_layer = AttentionPooling(input_dim=384)
        self.classifier = nn.Linear(384, dim_y)

    def forward(self, x):
        _, patch_tokens = super().forward(x, layers=1, remove_last_layer=True)
        aggregated_features = self.pooling_layer(patch_tokens)
        return self.classifier(aggregated_features)

def build_feat_extract_net(dim_y, remove_last_layer=False):
    model = DINOv2ForMIL(flag_finetuned=False, dim_y=dim_y)
    if remove_last_layer:
        model.classifier = nn.Identity()
    return model
