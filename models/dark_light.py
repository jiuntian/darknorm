import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

from .r2plus1d import r2plus1d_34_32_ig65m

__all__ = ['dark_light_single']


class dark_light_single(nn.Module):
    def __init__(self, num_classes, length, both_flow, backbone='r34'):
        super(dark_light_single, self).__init__()
        self.hidden_size = 512
        self.n_layers = 1
        self.attn_heads = 8
        self.num_classes = num_classes
        self.length = length
        self.dp = nn.Dropout(p=0.5)
        self.both_flow = both_flow
        self.backbone = backbone

        # self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.nobertpool = nn.AdaptiveAvgPool3d(1)
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
        # self.fc_action = CosSim(4096, num_classes)
        # self.bn = nn.BatchNorm1d(self.simclr_embedding)
        # self.fc_action = nn.Linear(self.hidden_size, num_classes)

        assert self.both_flow == 'False', f'Single required single flow, current set as {self.both_flow}'

        for param in self.features.parameters():
            param.requires_grad = True
        nn.init.normal_(self.fc_action.weight, 0, 0.01)
        # torch.nn.init.xavier_uniform_(self.fc_action.weight)
        # torch.nn.init.xavier_uniform_(self.mlp.weight)
        if not isinstance(self.fc_action, CosSim):
            self.fc_action.bias.data.zero_()

    def forward(self, x):
        x = x  # (b,3,64,112,112)
        # print(x.shape)

        x = self.features(x)  # x(b,512,8,7,7)

        x = self.avgpool(x)  # b,512,8,1,1
        # x = self.nobertpool(x)
        # x = x.view(x.size(0), 4096)  # x(b,4096)
        x = x.flatten(1)

        # x_proj = self.simclr_proj(x)
        # x_proj = self.bn(x_proj)

        # x_proj = self.dp(x_proj)
        logits = self.fc_action(x)  # b,11
        # return logits, x_proj, x_light_proj
        return logits
