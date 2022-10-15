import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

from .r2plus1d import r2plus1d_34_32_ig65m

__all__ = ['dark_light_single']


class dark_light_single(nn.Module):
    def __init__(self, num_classes, length, backbone='r18'):
        super(dark_light_single, self).__init__()
        self.hidden_size = 512
        self.n_layers = 1
        self.attn_heads = 8
        self.num_classes = num_classes
        self.length = length
        self.dp = nn.Dropout(p=0.5)
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # load pretrained model
        if self.backbone == 'r18':
            self.features = nn.Sequential(*list(
                r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1, progress=True).children())[:-2])
        elif self.backbone == 'r34':
            self.features = nn.Sequential(*list(
                r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        else:
            raise NotImplementedError('backbone unknown')
        self.fc_action = nn.Linear(512, num_classes)

        for param in self.features.parameters():
            param.requires_grad = True
        nn.init.normal_(self.fc_action.weight, 0, 0.01)
        self.fc_action.bias.data.zero_()

    def forward(self, x):
        x = x  # (b,3,64,112,112)
        x = self.features(x)  # x(b,512,8,7,7)
        x = self.avgpool(x)  # b,512,8,1,1
        x = x.flatten(1)
        logits = self.fc_action(x)  # b,11
        return logits
