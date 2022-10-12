import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from .BERT.self_attention import self_attention

from .r2plus1d import r2plus1d_34_32_ig65m

__all__ = ['dark_light_simclr_r34', 'dark_light_single', 'dark_light_simclr',
           'dark_light_arcface', 'dark_light', 'dark_light_noAttention']


class dark_light_simclr_r34(nn.Module):
    def __init__(self, num_classes, length, both_flow):
        super(dark_light_simclr_r34, self).__init__()
        self.hidden_size = 512
        self.n_layers = 1
        self.attn_heads = 8
        self.num_classes = num_classes
        self.length = length
        self.dp = nn.Dropout(p=0.5)
        self.both_flow = both_flow
        self.simclr_embedding = 128

        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.nobertpool = nn.AdaptiveAvgPool3d(1)
        # load pretrained model
        # self.features = nn.Sequential(*list(
        #     r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1, progress=True).children())[:-2])
        self.features = nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        # self.simclr_proj = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size // 2),
        #     nn.Linear(self.hidden_size // 2, self.simclr_embedding)
        # )
        self.fc_action = nn.Linear(self.hidden_size * 2, num_classes)
        # self.bn = nn.BatchNorm1d(self.simclr_embedding)
        # self.fc_action = nn.Linear(self.hidden_size, num_classes)

        assert self.both_flow == 'True', f'Simclr required both flow, current set as {self.both_flow}'

        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()

    def forward(self, x):
        # (b,3,64,112,112)
        x, x_light = x

        x = self.features(x)  # x(b,512,8,7,7)
        x_light = self.features(x_light)  # x(b,512,8,7,7)

        x = self.avgpool(x)  # b,512,8,1,1
        x = self.nobertpool(x)
        x = x.view(x.size(0), self.hidden_size)  # x(b,512)

        x_light = self.avgpool(x_light)  # b,512,8,1,1
        x_light = self.nobertpool(x_light)
        x_light = x_light.view(x_light.size(0), self.hidden_size)  # x(b,512)

        # x_proj = self.simclr_proj(x)
        # x_proj = self.bn(x_proj)
        # x_light_proj = self.simclr_proj(x_light)
        # x_light_proj = self.bn(x_light_proj)

        x_cat = torch.cat((x, x_light), 1)
        # x_cat = torch.cat((x_proj, x_light_proj), 1)
        # x_proj = self.dp(x_proj)
        logits = self.fc_action(x_cat)  # b,11
        # return logits, x_proj, x_light_proj
        return logits, x, x_light


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

        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.nobertpool = nn.AdaptiveAvgPool3d(1)
        # load pretrained model
        if self.backbone == 'r18':
            self.features = nn.Sequential(*list(
                r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1, progress=True).children())[:-2])
        elif self.backbone == 'r34':
            self.features = nn.Sequential(*list(
                r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        else:
            raise NotImplementedError('backbone unknown')
        # self.mlp = nn.Linear(512, 512)
        self.fc_action = nn.Linear(4096, num_classes)
        # self.fc_action = CosSim(4096, num_classes)
        # self.bn = nn.BatchNorm1d(self.simclr_embedding)
        # self.fc_action = nn.Linear(self.hidden_size, num_classes)

        assert self.both_flow == 'False', f'Single required single flow, current set as {self.both_flow}'

        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        # torch.nn.init.xavier_uniform_(self.mlp.weight)
        if not isinstance(self.fc_action, CosSim):
            self.fc_action.bias.data.zero_()

    def forward(self, x):
        x = x  # (b,3,64,112,112)
        # print(x.shape)

        x = self.features(x)  # x(b,512,8,7,7)

        x = self.avgpool(x)  # b,512,8,1,1
        # x = self.nobertpool(x)
        x = x.view(x.size(0), 4096)  # x(b,512)

        # x_proj = self.simclr_proj(x)
        # x_proj = self.bn(x_proj)

        # x_proj = self.dp(x_proj)
        logits = self.fc_action(x)  # b,11
        # return logits, x_proj, x_light_proj
        return logits


class dark_light_simclr(nn.Module):
    def __init__(self, num_classes, length, both_flow, backbone='r34'):
        super(dark_light_simclr, self).__init__()
        self.hidden_size = 512
        self.n_layers = 1
        self.attn_heads = 8
        self.num_classes = num_classes
        self.length = length
        self.dp = nn.Dropout(p=0.5)
        self.both_flow = both_flow
        self.simclr_embedding = 128
        self.backbone = backbone

        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.nobertpool = nn.AdaptiveAvgPool3d(1)
        # load pretrained model
        if self.backbone == 'r18':
            self.features = nn.Sequential(*list(
                r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1, progress=True).children())[:-2])
        elif self.backbone == 'r34':
            self.features = nn.Sequential(*list(
                r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        else:
            raise NotImplementedError('backbone unknown')
        self.simclr_proj = nn.Sequential(
            nn.Linear(4096, self.simclr_embedding)
        )
        # self.mlp = nn.Linear(512, 512)
        self.fc_action = nn.Linear(4096, num_classes)
        # self.bn = nn.BatchNorm1d(self.simclr_embedding)
        # self.fc_action = nn.Linear(self.hidden_size, num_classes)

        assert self.both_flow == 'True', f'Simclr required both flow, current set as {self.both_flow}'

        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        # torch.nn.init.xavier_uniform_(self.mlp.weight)
        self.fc_action.bias.data.zero_()

    def forward(self, x):
        # (b,3,64,112,112)
        x, x_light = x

        x = self.features(x)  # x(b,512,8,7,7)
        x_light = self.features(x_light)  # x(b,512,8,7,7)

        x = self.avgpool(x)  # b,512,8,1,1
        # x = self.nobertpool(x)
        x = x.view(x.size(0), 4096)  # x(b,512)

        x_light = self.avgpool(x_light)  # b,512,8,1,1
        # x_light = self.nobertpool(x_light)
        x_light = x_light.view(x_light.size(0), 4096)  # x(

        x_proj = self.simclr_proj(x)
        # x_proj = self.bn(x_proj)
        x_light_proj = self.simclr_proj(x_light)
        # x_light_proj = self.bn(x_light_proj)

        # x_cat = torch.cat((x, x_light), 1)
        # x_cat = torch.cat((x_proj, x_light_proj), 1)
        # x_proj = self.dp(x_proj)
        logits = self.fc_action(x)  # b,11
        # return logits, x_proj, x_light_proj
        return logits, x_proj, x_light_proj


class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

        self.weight = nn.Parameter(torch.randn(nclass, nfeat))
        if not learn_cent:
            self.weight.requires_grad_(False)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.weight, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.weight, norms_c)
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
        # self.fc_action.bias.data.zero_()

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
