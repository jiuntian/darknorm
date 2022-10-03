import torch
import torch.nn.functional as F
import torch.nn as nn


class ArcFaceLoss(nn.Module):
    def __init__(self, ce=1, s=10, m=0.2, **kwargs):
        super(ArcFaceLoss, self).__init__()
        self.ce = ce
        self.s = s
        self.m = m

    def forward(self, logits, labels, onehot=True):
        if onehot:
            labels = labels.argmax(1)

        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
        arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
        logits = torch.cos(arc_logits + y_onehot)
        margin_logits = self.s * logits

        loss_ce = F.cross_entropy(margin_logits, labels)

        self.losses['ce'] = loss_ce
        loss = self.ce * loss_ce
        return loss
