import torch

import torch.nn as nn

from domainlab.compos.nn_zoo.nn_torchvision import NetTorchVisionBase


DINO_PATH_FINETUNED_DOWNLOADED='/lustre/groups/aih/sina.wendrich/MA_code/test/dinov2_vits_student_TCGA'


def get_dino_finetuned_downloaded():

    # Load the original DINOv2 model with the correct architecture and parameters. The positional embedding is too large.

    architecture = 'vits'  # Choose 'vits' or 'vitg' based on your needs

    model = torch.hub.load('facebookresearch/dino', 'dino_vits14')

    # Load fine-tuned weights

    pretrained = torch.load(DINO_PATH_FINETUNED_DOWNLOADED, map_location=torch.device('cpu'))

    # Prepare the correct state dict for loading

    new_state_dict = {}

    for key, value in pretrained['student'].items():

        if 'dino_head' in key:

            print('not used')

        else:

            new_key = key.replace('backbone.', '')

            new_state_dict[new_key] = value

    # Adjust shape of pos_embed, shape depending on 'vits' or 'vitg'

    pos_embed_shape = (1, 257, 384) if architecture == 'vits' else (1, 257, 1536)

    pos_embed = nn.Parameter(torch.zeros(pos_embed_shape))

    model.pos_embed = pos_embed

    # Load the modified state dict

    model.load_state_dict(new_state_dict, strict=True)

    return model


class DINOv2Base(nn.Module):

    def __init__(self, remove_last_layer=False, flag_finetuned=False):

        super().__init__()

        self.remove_last_layer = remove_last_layer

        self.flag_finetuned = flag_finetuned

        self.fetch_net(flag_finetuned)

        

    def fetch_net(self, flag_finetuned):

        if flag_finetuned:

            # Load a fine-tuned DINOv2 model.

            self.net_torchvision = get_dino_finetuned_downloaded()

        else:

            # Load the pretrained DINOv2 model without any modifications.

            self.net_torchvision = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # Freeze two-thirds of the layers and unfreeze the last one-third.

        total_layers = len(list(self.net_torchvision.parameters()))

        two_third_layers = 2 * total_layers // 3

        for i, param in enumerate(self.net_torchvision.parameters()):

            param.requires_grad = i >= two_third_layers


    def forward(self, tensor, layers=1, remove_last_layer=False):

        x = self.net_torchvision.forward_features(tensor)

        cls_token = x["x_norm_clstoken"]

        patch_tokens = x["x_norm_patchtokens"]

        linear_input = torch.cat([

            cls_token,

            patch_tokens.mean(dim=1),

        ], dim=1)


        if remove_last_layer:

            return linear_input

        else:

            return self.net_torchvision.head(linear_input)


class DINOv2ForClassification(DINOv2Base):

    """

    Adapts DINOv2 for classification tasks.

    """

    def __init__(self, dim_y, remove_last_layer=False, flag_finetuned=False):

        super().__init__(remove_last_layer=remove_last_layer, flag_finetuned=flag_finetuned)

        embed_dim = 384

        self.net_torchvision.head = self._make_dinov2_linear_classification_head(dim_y, embed_dim, layers=1)


    def _make_dinov2_linear_classification_head(self, dim_y, embed_dim=384, layers=1):

        linear_head = nn.Linear((1 + layers) * embed_dim, dim_y)

        return linear_head


# Example usage:

def build_feat_extract_net(dim_y, remove_last_layer):

    return DINOv2ForClassification(dim_y=dim_y, remove_last_layer=remove_last_layer, flag_finetuned=False)

