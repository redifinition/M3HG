import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from model.GAT import GATLayer
from model.MultiHeadAttention import MultiHeadAttention
from model.PairNN import PairNN
from utils.global_variables import DEVICE, EMOTION_MAPPING, GRAPH_CONFIG_T


def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

class M3HG(nn.Module):
    def __init__(self, lm_config, config, data_name="ECF", activation='relu', modality = ['textual']):
        super(M3HG, self).__init__()
        self.gcn_dim = config.hidden_dim
        if 'textual' in modality:
            self.bert = BertModel.from_pretrained(config.textual_pretrain_model_dir)
            self.attention_head_size = int(lm_config.hidden_size / lm_config.num_attention_heads)
            self.turnAttention = MultiHeadAttention(lm_config.num_attention_heads, lm_config.hidden_size,
                                                self.attention_head_size, self.attention_head_size,
                                                lm_config.attention_probs_dropout_prob)
            self.linear_t = nn.Linear(lm_config.hidden_size, self.gcn_dim)
        self.pred = Pre_Predictions(2 * self.gcn_dim, len(set(EMOTION_MAPPING[data_name].values())))
        self.ec_pairing = PairNN(2 * self.gcn_dim, config.pos_emb_dim, config.rel_pos_k)
        self.gat_layers = config.gat_layers
        self.graph_attention_size = int(self.gcn_dim / config.num_graph_attention_heads)
        # 图注意力层
        self.GAT_layers = nn.ModuleList([GATLayer(meta_paths= GRAPH_CONFIG_T['meta_paths'],
                                                  in_size=self.gcn_dim, out_size=self.graph_attention_size,
                                                  layer_num_heads=config.num_graph_attention_heads) for _ in range(self.gat_layers)])
        self.ffn_layers = nn.ModuleList([PositionWiseFeedForward(self.gcn_dim, self.gcn_dim, 0.2) for _ in range(self.gat_layers)])

    def forward(self,
                input_ids=None,
                attention_masks=None,
                token_type_ids=None,
                position_ids=None,
                head_masks=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False,
                speaker_ids=None,
                mention_ids=None,
                emotion_ids=None,
                turn_masks=None,
                uttr_indices = None,
                graphs = None,
                uttr_len = None,
                audio_features = None,
                video_features = None,
                modality = []):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.size(0)
        # 清理显存的函数
        def clear_cache(*vars):
            for var in vars:
                del var
            torch.cuda.empty_cache()
        if 'textual' in modality:
            outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_masks,
                            token_type_ids=token_type_ids)
            sequence_outputs_h = outputs[0] # B*S*D
            pooled_outputs = outputs[1]   # B*D
            sequence_outputs_h, _ = self.turnAttention(sequence_outputs_h, sequence_outputs_h, sequence_outputs_h, turn_masks) # turn_masks: B*S*S
            sequence_outputs_h = self._batched_index_select(sequence_outputs_h, uttr_indices, mention_ids) # B*N*D
            sequence_outputs_h = self.linear_t(sequence_outputs_h)
            pooled_outputs = self.linear_t(pooled_outputs).unsqueeze(1)

        h_dict = {'utterance':None,'conversation':None,'emotion':None,'cause':None} # 用于存储不同类型的节点特征
        # h_dict = {'utterance_t':None,'utterance_a':None, 'utterance_v':None,
        #           'conversation_t':None,'conversation_a':None,'conversation_v':None,
        #           'emotion':None,'cause':None} # 用于存储不同类型的节点特征
        graph_in = sequence_outputs_h # B*N*H
        # initialize graph nodes
        for i in range(len(graphs)):
            sequence_outputs_h_i = sequence_outputs_h[i][:uttr_len[i]] # N*D(N要去除padding的部分)
            conversation_h_i =  pooled_outputs[i] # H
            if h_dict['utterance'] is not None:
                h_dict['utterance'] = torch.cat([h_dict['utterance'], sequence_outputs_h_i], dim=0)
            else:
                h_dict['utterance'] = sequence_outputs_h_i
            assert uttr_len[i] == (graphs[i].num_nodes()-1)/3 == sequence_outputs_h_i.shape[0] # 除去对话节点

            if h_dict['conversation'] is not None:
                h_dict['conversation'] = torch.cat([h_dict['conversation'], conversation_h_i], dim=0)
            else:
                h_dict['conversation'] = conversation_h_i
            if h_dict['emotion'] is not None:
                h_dict['emotion'] = torch.cat([h_dict['emotion'], sequence_outputs_h_i], dim=0)
            else:
                h_dict['emotion'] = sequence_outputs_h_i
            if h_dict['cause'] is not None:
                h_dict['cause'] = torch.cat([h_dict['cause'], sequence_outputs_h_i], dim=0)
            else:
                h_dict['cause'] = sequence_outputs_h_i
        
        # construct big graph:
        graph_big = dgl.batch(graphs)
        uttr_nodes_num = int((graph_big.num_nodes() - batch_size)/3) # 除去对话节点的话语节点数/3，因为总共有三个类型节点，除去对话节点


        for layer_num, GAT_layer in enumerate(self.GAT_layers):
            # graph_features 
            graph_features = GAT_layer(graph_big, h_dict) # M * H
            graph_features = self.ffn_layers[layer_num](graph_features.unsqueeze(1)).squeeze(1) # M * H
            h_dict['conversation'] = graph_features[:batch_size]
            h_dict['utterance'] = graph_features[batch_size:uttr_nodes_num+batch_size]
            h_dict['emotion'] = graph_features[uttr_nodes_num+batch_size: 2*uttr_nodes_num+batch_size]
            h_dict['cause'] = graph_features[2*uttr_nodes_num+batch_size:]
        graphs = dgl.unbatch(graph_big) 

        if h_dict['utterance'].dim() > 2:
            h_dict['utterance'], h_dict['conversation'],h_dict['emotion'], h_dict['cause'] = \
            h_dict['utterance'].squeeze(0), h_dict['conversation'].squeeze(0), h_dict['emotion'].squeeze(0), h_dict['cause'].squeeze(0)
        # get the output of the last GAT layer
        fea_idx = 0
        max_uttr_num = max(uttr_len)
        gragh_h_emotion = None # 用于存储图结构的情绪节点特征
        gragh_h_cause = None # 用于存储图结构的原因节点特征

        for i in range(len(graphs)):
            node_num = int((graphs[i].num_nodes() - 1)/3) # M， 这里包括了一个对话的对话节点和话语节点
            graph_h_emotion_i = h_dict['emotion'][fea_idx:fea_idx+node_num]
            padded_graph_h_emotion_i = F.pad(graph_h_emotion_i, (0, 0, 0, max_uttr_num - graph_h_emotion_i.size(0))) # padding N*H
            graph_h_cause_i = h_dict['cause'][fea_idx:fea_idx+node_num]
            padded_graph_h_cause_i = F.pad(graph_h_cause_i, (0, 0, 0, max_uttr_num - graph_h_cause_i.size(0))) # padding N*H
            fea_idx += node_num
            if gragh_h_emotion is None:
                gragh_h_emotion = padded_graph_h_emotion_i.unsqueeze(0)
            else:
                gragh_h_emotion = torch.cat([gragh_h_emotion, padded_graph_h_emotion_i.unsqueeze(0)], dim=0)
            if gragh_h_cause is None:
                gragh_h_cause = padded_graph_h_cause_i.unsqueeze(0)
            else:
                gragh_h_cause = torch.cat([gragh_h_cause, padded_graph_h_cause_i.unsqueeze(0)], dim=0)

        em_logits, ca_logits = self.pred(torch.cat([graph_in, gragh_h_emotion], dim = -1), torch.cat([graph_in, gragh_h_cause], dim = -1))
        emotion_g_h = torch.cat([graph_in, gragh_h_emotion], dim = -1)
        cause_g_h = torch.cat([graph_in, gragh_h_cause], dim = -1)
        couples_pred, emo_cau_pos = self.ec_pairing(emotion_g_h, cause_g_h)
        return em_logits, ca_logits, couples_pred, emo_cau_pos
        
    # def _batched_index_select(self, sequence_outputs , uttr_indices):
    #     dummy = uttr_indices.unsqueeze(2).expand(uttr_indices.size(0), uttr_indices.size(1), sequence_outputs.size(2))
    #     doc_sents_h = sequence_outputs.gather(1, dummy)
    #     return doc_sents_h

    def _batched_index_select(self, sequence_outputs , uttr_indices, mention_ids):
        max_uttr_len = uttr_indices.size(1)
        slen = sequence_outputs.shape[1]
        feature_dim = sequence_outputs.size(-1)
        batch_size = sequence_outputs.size(0)
        # 初始化一个填充张量
        doc_sents_h = get_cuda(torch.zeros(batch_size, max_uttr_len, feature_dim))
        for i in range(sequence_outputs.size(0)): # 对于每个batch
            sequence_output = sequence_outputs[i]
            mention_num = torch.max(mention_ids[i]) # 话语数
            mention_index = get_cuda((torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen))  # N * S (4*138)
            mentions = mention_ids[i].unsqueeze(0).expand(mention_num, -1) # shape:N*512
            select_metrix = (mention_index == mentions).float()
            word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen) # 每个话语的词数
            select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)
            uttr_h = torch.mm(select_metrix, sequence_output) # 根据话语长度做加权平均
            doc_sents_h[i, :uttr_h.size(0), :] = uttr_h
        return doc_sents_h
    
    def loss_rank(self, couples_pred, emo_cau_pos, ec_pair, uttr_mask, test=False):
        couples_true, couples_mask = self.truncate_ec_pairs(couples_pred, emo_cau_pos, ec_pair, uttr_mask, test)
        couples_mask = torch.BoolTensor(couples_mask).to(DEVICE)
        couples_true = torch.FloatTensor(couples_true).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        couples_true = couples_true.masked_select(couples_mask)
        couples_pred = couples_pred.masked_select(couples_mask)
        loss_couple = criterion(couples_pred, couples_true)
        return loss_couple

    def truncate_ec_pairs(self, couples_pred, emo_cau_pos, ec_pair, uttr_mask, test=False):
        batch, n_couple = couples_pred.size()
        couples_true, couples_mask = [], []
        for i in range(batch):
            uttr_mask_i = uttr_mask[i]
            uttr_len = uttr_mask_i.sum().item()
            uttr_couples_i = ec_pair[i] # 该对话的真实配对
            couples_true_i = []
            couples_mask_i = []
            for couple_idx, emo_cau in enumerate(emo_cau_pos): # 对每个笛卡尔积后的情绪原因配对（经过截断后的），但是经过batch的padding后，可能有一些配对是无效的
                if emo_cau[0] > uttr_len or emo_cau[1] > uttr_len: # 情绪idx或者原因idx如果超出了该对话的长度
                    couples_mask_i.append(0) 
                    couples_true_i.append(0)
                else:
                    couples_mask_i.append(1)
                    couples_true_i.append(1 if emo_cau in uttr_couples_i.tolist() else 0)
            couples_true.append(couples_true_i)
            couples_mask.append(couples_mask_i)
        return couples_true, couples_mask
    
    def loss_pre(self, pred_e, pred_c, gold_e, gold_c):
        uttr_len = pred_e.size(1)
        # 这里进行截断
        gold_e = gold_e[:,:uttr_len]
        gold_e = gold_e[:,:uttr_len]-1 # 计算时还是要从0-6
        criterion_e = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1) # 为-1的值被忽略
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        loss_e = criterion_e(pred_e.permute(0,2,1), gold_e)
        # 创建 mask 张量
        mask = gold_c != -1  # 将 bool 类型转换为 uint8 类型
        # 计算需要填充的宽度
        c_pad_width = gold_c.shape[1] - pred_c.shape[1]
        pred_c = F.pad(pred_c, (0, c_pad_width), "constant", value=6)
        pred_c = pred_c.masked_select(mask)
        gold_c = gold_c.masked_select(mask)
        loss_c = criterion(pred_c, gold_c.float())
        return loss_e, loss_c

class Pre_Predictions(nn.Module):
    def __init__(self, feat_dim, num_classes=7):
        super(Pre_Predictions, self).__init__()
        self.feat_dim = feat_dim
        self.out_e = nn.Linear(self.feat_dim, num_classes)
        self.out_c = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_emo, doc_sents_cau):
        pred_e = self.out_e(doc_sents_emo)
        pred_c = self.out_c(doc_sents_cau)
        return pred_e.squeeze(2), pred_c.squeeze(2)
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(in_channels=1, out_channels=input_size, kernel_size=1)
        self.w2 = nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=1)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w2(F.elu(self.w1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x