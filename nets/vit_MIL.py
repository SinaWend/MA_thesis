import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from domainlab.compos.nn_zoo.nn_torchvision import NetTorchVisionBase

class ViTBase(NetTorchVisionBase):
    def fetch_net(self, flag_pretrain):
        model_name = "vit_base_patch16_224"  # Adjust model_name as needed for other variants like 'swin_tiny_patch4_window7_224'
        if flag_pretrain:
            self.net_torchvision = timm.create_model(model_name, pretrained=True)
        else:
            self.net_torchvision = timm.create_model(model_name, pretrained=False)

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, num_patches, feature_dim]
        attention_weights = torch.softmax(self.attention(x), dim=1)
        weighted_features = attention_weights * x
        aggregated_features = weighted_features.sum(1)  # Sum across patches
        return aggregated_features

class ViTForMIL(ViTBase):
    """
    Adapt Vision Transformer for MIL classification
    """
    def __init__(self, flag_pretrain, dim_y):
        super().__init__(flag_pretrain)
        num_final_in = self.net_torchvision.head.in_features

        # Replace the standard classification head with a custom MIL pooling layer
        self.pooling_layer = AttentionPooling(input_dim=num_final_in)
        self.classifier = nn.Linear(num_final_in, dim_y)  # Classifier after pooling

    def forward(self, x):
        x = self.net_torchvision.forward_features(x)  # Extract features from base ViT model
        x = self.pooling_layer(x)  # Pool features across all patches in a slide
        x = self.classifier(x)  # Final classification layer
        return x

def build_feat_extract_net(dim_y, flag_pretrain=True, remove_last_layer=False):
    """
    Build a feature extractor or full model for MIL with ViT.
    - param dim_y: Number of classes.
    """
    model = ViTForMIL(flag_pretrain=flag_pretrain, dim_y=dim_y)
    if remove_last_layer:
        model.classifier = nn.Identity()  # Optionally remove the classifier for feature extraction purposes
    return model
