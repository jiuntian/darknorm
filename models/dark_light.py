import torch
import torch.nn as nn
from .BERT.self_attention import self_attention

from .r2plus1d import r2plus1d_34_32_ig65m

__all__ = ['dark_light_arcface', 'dark_light', 'dark_light_noAttention']


class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

        self.centroids = nn.Parameter(torch.randn(nclass, nfeat))
        if not learn_cent:
            self.centroids.requires_grad_(False)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits

    def extra_repr(self) -> str:
        return 'in_features={}, n_class={}, learn_centroid={}'.format(
            self.nfeat, self.nclass, self.learn_cent
        )


class dark_light_arcface(nn.Module):
    def __init__(self, num_classes, length, both_flow):
        super(dark_light_arcface, self).__init__()
        self.hidden_size = 512
        self.n_layers = 1
        self.attn_heads = 8
        self.num_classes = num_classes
        self.length = length
        self.dp = nn.Dropout(p=0.8)
        self.both_flow = both_flow

        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        # load pretrained model
        self.features = nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        if self.both_flow == 'True':
            max_length = 16
        elif self.both_flow == 'False':
            max_length = 8
        self.self_attention = self_attention(self.hidden_size, max_length, hidden=self.hidden_size,
                                             n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.self_attention.parameters() if p.requires_grad))
        # self.fc_action = nn.Linear(self.hidden_size, num_classes)
        self.fc_action = CosSim(self.hidden_size, num_classes)

        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()

    def forward(self, x):
        if self.both_flow == 'True':
            # (b,3,64,112,112)
            x, x_light = x

            x = self.features(x)  # x(b,512,8,7,7)
            x_light = self.features(x_light)  # x(b,512,8,7,7)
            # x = x * self.fuse_weights[0] + x_light * self.fuse_weights[1]
            x = self.avgpool(x)  # b,512,8,1,1
            x = x.view(x.size(0), self.hidden_size, 8)  # x(b,512,8)
            x = x.transpose(1, 2)  # x(b,8,512)
            x_light = self.avgpool(x_light)  # b,512,8,1,1
            x_light = x_light.view(x_light.size(0), self.hidden_size, 8)  # x(b,512,8)
            x_light = x_light.transpose(1, 2)  # x
            x_cat = torch.cat((x, x_light), 1)
            output, maskSample = self.self_attention(x_cat)  # output(b,9,512),masksample(b,9)
        elif self.both_flow == 'False':
            _, x = x
            x = self.features(x)  # x(b,512,8,7,7)

            x = self.avgpool(x)  # b,512,8,1,1
            x = x.view(x.size(0), self.hidden_size, 8)  # x(b,512,8)
            x = x.transpose(1, 2)  # x(b,8,512)

            output, maskSample = self.self_attention(x)  # output(b,9,512),masksample(b,9)
        classificationOut = output[:, 0, :]  # class(b,512)

        output = self.dp(classificationOut)  # b,512
        x = self.fc_action(output)  # b,11
        return x


class dark_light(nn.Module):
    def __init__(self, num_classes, length, both_flow):
        super(dark_light, self).__init__()
        self.hidden_size = 512
        self.n_layers = 1
        self.attn_heads = 8
        self.num_classes = num_classes
        self.length = length
        self.dp = nn.Dropout(p=0.8)
        self.both_flow = both_flow

        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        # load pretrained model
        self.features = nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        if self.both_flow == 'True':
            max_length = 16
        elif self.both_flow == 'False':
            max_length = 8
        self.self_attention = self_attention(self.hidden_size, max_length, hidden=self.hidden_size,
                                             n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.self_attention.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)

        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()

    def forward(self, x):
        if self.both_flow == 'True':
            # (b,3,64,112,112)
            x, x_light = x

            x = self.features(x)  # x(b,512,8,7,7)
            x_light = self.features(x_light)  # x(b,512,8,7,7)
            # x = x * self.fuse_weights[0] + x_light * self.fuse_weights[1]
            x = self.avgpool(x)  # b,512,8,1,1
            x = x.view(x.size(0), self.hidden_size, 8)  # x(b,512,8)
            x = x.transpose(1, 2)  # x(b,8,512)
            x_light = self.avgpool(x_light)  # b,512,8,1,1
            x_light = x_light.view(x_light.size(0), self.hidden_size, 8)  # x(b,512,8)
            x_light = x_light.transpose(1, 2)  # x
            x_cat = torch.cat((x, x_light), 1)
            output, maskSample = self.self_attention(x_cat)  # output(b,9,512),masksample(b,9)
        elif self.both_flow == 'False':
            _, x = x
            x = self.features(x)  # x(b,512,8,7,7)

            x = self.avgpool(x)  # b,512,8,1,1
            x = x.view(x.size(0), self.hidden_size, 8)  # x(b,512,8)
            x = x.transpose(1, 2)  # x(b,8,512)

            output, maskSample = self.self_attention(x)  # output(b,9,512),masksample(b,9)
        classificationOut = output[:, 0, :]  # class(b,512)

        output = self.dp(classificationOut)  # b,512
        x = self.fc_action(output)  # b,11
        return x


class dark_light_noAttention(nn.Module):
    def __init__(self, num_classes, length, both_flow):
        super(dark_light_noAttention, self).__init__()
        self.hidden_size = 512
        self.n_layers = 1
        self.attn_heads = 8
        self.num_classes = num_classes
        self.length = length
        self.dp = nn.Dropout(p=0.8)
        self.both_flow = both_flow

        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.nobertpool = nn.AdaptiveAvgPool3d(1)
        self.features = nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        self.fc_action = nn.Linear(self.hidden_size, num_classes)

        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()

    def forward(self, x):
        if self.both_flow == 'True':
            # (b,3,64,112,112)
            x, x_light = x
            x = self.features(x)  # x(b,512,8,7,7)
            x_light = self.features(x_light)  # x(b,512,8,7,7)

        elif self.both_flow == 'False':
            _, x = x
            x = self.features(x)  # x(b,512,8,7,7)
        x = self.nobertpool(x)
        x = x.view(-1, 512)
        x = self.fc_action(x)  # b,11

        return x
