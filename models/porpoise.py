#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_coattn.py
@Time    :   2022/07/07 16:43:59
@Author  :   Innse Xu 
@Contact :   innse76@gmail.com
'''

# Here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class BilinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling

    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        device = vec1.device
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1).to(device)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1).to(device)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        # x = x.unsqueeze(0)
        ## x: N x L
        # x = x.squeeze(0)
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights( A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N



class PORPOISE(nn.Module):
    def __init__(self, fusion='bilinear', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, in_dim=1024,
                model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(PORPOISE, self).__init__()
        self.fusion = fusion
        self.n_classes = n_classes
        self.dropout = dropout
        self.size_dict_WSI = {"small": [in_dim, 256, 256], "big": [in_dim, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [in_dim, in_dim, in_dim, 256]}

        feature_dim = self.size_dict_WSI[model_size_wsi]
        self.wsi_net = nn.Sequential(nn.Linear(feature_dim[0], feature_dim[1]), 
                                    nn.ReLU(), 
                                    nn.Dropout(0.25))

        self.mlp = nn.Sequential(
                nn.Linear(50, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU()
            )


        self.attention_path = Attention_Gated(L=feature_dim[1], D=128, K=1)
        self.attention_omic = Attention_Gated(L=feature_dim[1], D=128, K=1)


        self.mm_multi = BilinearFusion(dim1=feature_dim[-1], dim2=feature_dim[-1], scale_dim1=8, scale_dim2=8, mmhid=feature_dim[-1])
        ### Classifier
        self.classifier = nn.Linear(feature_dim[-1], n_classes)
        self.apply(initialize_weights)

    def forward(self, x, y, return_WSI_attn = False, return_WSI_feature = False):


        h_path_bag = self.wsi_net(x)
        h_omic_bag = self.mlp(y)

        AA_path = self.attention_path(h_path_bag)
        h_path_bag = torch.mm(AA_path, h_path_bag)
        AA_omic = self.attention_omic(h_omic_bag)
        h_omic_bag = torch.mm(AA_omic, h_omic_bag)
        fusion = self.mm_multi(h_omic_bag, h_path_bag)

        logits = self.classifier(fusion.reshape(1, -1)) ## K x num_cls

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)    
        return hazards, S, Y_hat, None, None
    

if __name__ == '__main__':
    data_wsi = torch.randn((6000, 512))
    data_omic = torch.randn((6000, 50))
    data_wsi, data_omic = data_wsi.cuda(), data_omic.cuda()
    model = PORPOISE(fusion='bilinear', n_classes=4, model_size_wsi='small', model_size_omic='small', dropout=0.25, in_dim=512).cuda()
    print(model.eval())
    out = model(x=data_wsi, y=data_omic)