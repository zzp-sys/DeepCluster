"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x)) #
        #features = self.backbone(x) #做tsne用到，结束改回上行代码
        features = F.normalize(features, dim = 1)
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0) # backbone_dim:512, nclusters:10
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])

        self.pseudo_top_layer = nn.Linear(512, 100) #cifar10, cifar100
        #self.pseudo_top_layer = nn.Linear(512, 10) #stl10
        self.top_layer = nn.Linear(512, 10) #cifar10 stl10
        #self.top_layer = nn.Linear(512, 50) #消融实验不同聚类簇
        #self.top_layer = nn.Linear(512, 20) #cifar100

    def forward(self, x, flag, is_main, forward_pass='default' ): #flag是pseudo-label模型用的，加个is_main=true表示是锚点和邻居数据对训练
        if forward_pass == 'default':
            features = self.backbone(x) #features.shape=torch.size([256,512]) x.shape=torch.size([256,3,32,32])
            #out = [cluster_head(features) for cluster_head in self.cluster_head]
            features = F.normalize(features, dim=1)
            y = self.top_layer(features) #y.shape=torch.size([256,10])
            pseudo_y = self.pseudo_top_layer(features) #pseudo_y.shape=torch.size([256,100])
            
            if flag:
                return pseudo_y, y
            
            elif is_main:
                out = [cluster_head(features) for cluster_head in self.cluster_head]
            
            
            else:
                out = features
            """elif is_supcon:
                feat = self.top_layer(features)
                out = F.normalize(feat, dim=1)"""
        elif forward_pass == 'backbone':
            out = self.backbone(x)
            y = self.top_layer(out)
            pseudo_y = self.pseudo_top_layer(out)
            if flag:
                return pseudo_y, y

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]
            y = self.top_layer(out)
            pseudo_y = self.pseudo_top_layer(out)
            if flag:
                return pseudo_y, y

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            y = self.top_layer(features)
            pseudo_y = self.pseudo_top_layer(features)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
            #把特征分类为10类
            
            if flag:
                return pseudo_y, y
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

            
        return out #torch.size([512])
