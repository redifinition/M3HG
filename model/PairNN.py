import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.global_variables import DEVICE

class PairNN(nn.Module):
    def __init__(self, gcn_dim, pos_emb_dim, K):
        super(PairNN, self).__init__()
        self.K = K
        self.pairing_feat_dim = 2 * gcn_dim +  pos_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.pos_layer = nn.Embedding(2 * self.K + 1, self.pos_emb_dim)
        nn.init.xavier_uniform_(self.pos_layer.weight)
        self.pairing_mlp_layer1 = nn.Linear(self.pairing_feat_dim, self.pairing_feat_dim)
        self.pairing_mlp_layer2 = nn.Linear(self.pairing_feat_dim, 1)
        
    def forward(self, uttr_output_h_emotion, uttr_output_h_cause):
        batch, _, _ = uttr_output_h_emotion.size() # B*N*D
        couples_h, rel_pos, emo_caus_pos = self.pairing(uttr_output_h_emotion, uttr_output_h_cause, self.K)
        rel_pos = rel_pos + self.K
        rel_pos_emb = self.pos_layer(rel_pos)
        kernel = self.kernel_generator(rel_pos)
        kernel = kernel.unsqueeze(0).expand(batch, -1, -1)
        rel_pos_emb = torch.matmul(kernel, rel_pos_emb)

        couples_h = torch.cat([couples_h, rel_pos_emb], dim=2)
        couples_h = F.relu(self.pairing_mlp_layer1(couples_h))
        couples_pred = self.pairing_mlp_layer2(couples_h)
        return couples_pred.squeeze(2), emo_caus_pos

    def kernel_generator(self, rel_pos):
        n_couple = rel_pos.size(1) # N*N
        rel_pos_ = rel_pos[0].type(torch.FloatTensor).to(DEVICE)
        kernel_left = torch.cat([rel_pos_.reshape(-1, 1)] * n_couple, dim=1)
        kernel = kernel_left - kernel_left.transpose(0, 1)
        return torch.exp(-(torch.pow(kernel, 2)))
    
    def pairing(self, H_emo, H_cause, k):
        batch, uttr_len, feat_dim = H_emo.size()
        P_left = torch.cat([H_emo] * uttr_len, dim=2) # B*N*(N*D)
        P_left = P_left.reshape(-1, uttr_len * uttr_len, feat_dim) 
        P_right = torch.cat([H_cause] * uttr_len, dim=1)
        P = torch.cat([P_left, P_right], dim=2)

        base_idx = np.arange(1, uttr_len + 1)
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * uttr_len, axis=1).reshape(1, -1)[0]
        cau_pos = np.concatenate([base_idx] * uttr_len, axis=0)

        rel_pos = cau_pos - emo_pos
        rel_pos = torch.LongTensor(rel_pos).to(DEVICE)
        emo_pos = torch.LongTensor(emo_pos).to(DEVICE)
        cau_pos = torch.LongTensor(cau_pos).to(DEVICE)
        # 距离敏感
        if uttr_len > k:
            rel_mask = np.array(list(map(lambda x: -k <= x <= k, rel_pos.tolist())), dtype=int) # 长度为N*N, 每个元素1表示该位置的情绪原因index之差的绝对值在k之内，0表示不在k之内
            rel_mask = torch.BoolTensor(rel_mask).to(DEVICE)
            rel_pos = rel_pos.masked_select(rel_mask)
            emo_pos = emo_pos.masked_select(rel_mask)
            cau_pos = cau_pos.masked_select(rel_mask)

            rel_mask = rel_mask.unsqueeze(1).expand(-1, 2 * feat_dim)  #【N*N,2*D】
            rel_mask = rel_mask.unsqueeze(0).expand(batch, -1, -1) #[B,N*N,2*D]
            P = P.masked_select(rel_mask)
            P = P.reshape(batch, -1, 2 * feat_dim)
        assert rel_pos.size(0) == P.size(1)

        rel_pos = rel_pos.unsqueeze(0).expand(batch, -1)

        emo_cau_pos = []
        for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
            emo_cau_pos.append([emo, cau])
        return P, rel_pos, emo_cau_pos