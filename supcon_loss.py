"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


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
        """device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))"""
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).cuda() #to(device) #生成大小为batch_size，对角线为0，其余为1的矩阵
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            #contiguous()确保张量在内存中是连续存储的，将原始标签数据从一维张量变成二维张量，其中每个元素都在一个单独的行中。
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().cuda() #to(device)
            #这一步使用torch.eq函数比较labels张量和其转置（labels.T）之间的元素是否相等。
            # 它会返回一个布尔值的张量，其中相等的元素对应位置的值为True，不相等的元素对应位置的值为False。
            # 这个操作的结果是一个对称的布尔值矩阵，用于表示labels中元素之间的相等关系。
            #这段代码的目的是根据labels张量中元素的相等关系创建一个对称的浮点数矩阵mask，其中1.0表示相等，0.0表示不相等。
            # 这种矩阵通常用于在深度学习中执行一些特定的操作，例如在自注意力机制中控制不同位置之间的注意力权重。

        else:
            mask = mask.float().cuda #to(device)

        contrast_count = features.shape[1] #
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #shape=(512,512)
        #将原始特征张量 features 中的特征按列拆分成多个子特征，并将这些子特征按行堆叠在一起，形成一个新的特征张量 contrast_feature。
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
            self.temperature) #锚点特征和对比特征的乘积除以温度系数
        # for numerical stability 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        #它在 anchor_dot_contrast 张量的每一行中找到最大的元素，并保持维度保持不变
        logits = anchor_dot_contrast - logits_max.detach() #以防止数值上溢或数值不稳定性问题。

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        #生成一个新的张量，其形状为 (anchor_count * mask.shape[0], contrast_count * mask.shape[1])
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(), #to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) ) #计算负对损失
        #log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        """pos_per_sample = mask.sum(1) #B
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample #mask.sum(1)"""
        #mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    

        loss = loss.view(anchor_count, batch_size).mean() #view(2,256)

        return loss
