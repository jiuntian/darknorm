import torch
import torch.nn.functional as F
import torch.nn as nn


class SupSimClrLoss(nn.Module):
    def __init__(self):
        super(SupSimClrLoss, self).__init__()
        self.supconloss = SupConLoss()

    def forward(self, outputs, labels):
        logits, x_proj, x_light_proj = outputs
        x_proj = F.normalize(x_proj, dim=1)
        x_light_proj = F.normalize(x_light_proj, dim=1)
        x_cat = torch.stack([x_proj, x_light_proj], 1)
        loss_simclr = self.supconloss(x_cat, labels)

        loss_ce = F.cross_entropy(logits, labels)

        loss = 1. * loss_simclr + loss_ce
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1))

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ArcFaceLoss(nn.Module):
    def __init__(self, ce=1, s=3, m=0.1, **kwargs):
        super(ArcFaceLoss, self).__init__()
        self.ce = ce
        self.s = s
        self.m = m

    def forward(self, logits, labels, onehot=False, epsilon=3.0, alpha=0.1):
        if onehot:
            labels = labels.argmax(1)

        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
        arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
        logits = torch.cos(arc_logits + y_onehot)
        margin_logits = self.s * logits

        # smooth_labels = labels * (1 - alpha) + alpha / logits.shape[1]
        # one_minus_pt = smooth_labels * (1 - F.log_softmax(margin_logits, dim=1)).sum(dim=1)
        loss_ce = F.cross_entropy(margin_logits, labels)
        # loss_ce = loss_ce + epsilon * one_minus_pt

        loss = self.ce * loss_ce
        return loss.mean()


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        self.losses = {}

    def forward(self, logits, labels, onehot=False):
        if onehot:
            labels = labels.argmax(1)
        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), 1)
        loss = 1 - F.cosine_similarity(logits, y_onehot)
        return loss.mean()
