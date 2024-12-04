import torch
from model.CME import ACME, ACMELayer
from model.GAT import GATLayer
from model.MultiHeadAttention import MultiHeadAttention
from model.PairNN import PairNN
import torch.nn as nn
from transformers import BertModel
from model.Transformer import AdditiveAttention, TransEncoder
from model.UME import RUME, RUMELayer
from model.baselines.rankcp.rank_cp import RankNN
from pretrained_models.textual.RoBERTa.SRoBERTa import RobertaModel, RobertaPreTrainedModel
from transformers import Wav2Vec2Model
from utils.global_variables import DEVICE, EMOTION_MAPPING, GRAPH_CONFIG_T
import torch.nn.functional as F
import dgl

def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

# class MECPECModel(nn.Module):
#     _keys_to_ignore_on_load_missing = [r"position_ids"]
#     def __init__(self, lm_config, config, data_name="ECF", activation='relu', modality = ['textual']):
class MECPECModel(nn.Module):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, lm_config, config, data_name="ECF", activation='relu', modality = ['textual']):
        super(MECPECModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(config.pretrain_model_dir)
        # super(MECPECModel, self).__init__()
        self.gcn_dim = config.hidden_dim
        if 'textual' in modality:
            # self.bert = BertModel(lm_config)
            self.attention_head_size = int(lm_config.hidden_size / lm_config.num_attention_heads)
            self.turnAttention = MultiHeadAttention(lm_config.num_attention_heads, lm_config.hidden_size,
                                                self.attention_head_size, self.attention_head_size,
                                                lm_config.attention_probs_dropout_prob)
            self.linear_t = nn.Linear(lm_config.hidden_size, self.gcn_dim)

        if 'audio' in modality:
            self.linear_a = nn.Linear(512, self.gcn_dim)
            rumeLayer = RUMELayer(feature_size=self.gcn_dim, dropout=config.rnn_drop, rnn_type=config.rnn_type, use_vanilla=config.use_vanilla, use_rnnpack=config.use_rnnpack, no_cuda=False)
            self.rume = RUME(rumeLayer,num_layers=config.rnn_n_layers)
        if 'video' in modality:
            self.linear_v = nn.Linear(1000, self.gcn_dim)
            rumeLayer = RUMELayer(feature_size=self.gcn_dim, dropout=config.rnn_drop, rnn_type=config.rnn_type, use_vanilla=config.use_vanilla, use_rnnpack=config.use_rnnpack, no_cuda=False)
            self.rume = RUME(rumeLayer,num_layers=config.rnn_n_layers)

            # self.attention_head_size = int(768 / lm_config.num_attention_heads)
            # self.turnAttention = MultiHeadAttention(lm_config.num_attention_heads, 768,
            #                                     self.attention_head_size, self.attention_head_size,
            #                                     lm_config.attention_probs_dropout_prob)
        
        acmeLayer = ACMELayer(feature_size=self.gcn_dim, nheads=config.cross_num_head,dropout=config.cross_drop,no_cuda=config.no_cuda)
        self.acme = ACME(acmeLayer, num_layers=config.cross_n_layers)
        self.linear_cat = nn.Linear(2*self.gcn_dim, self.gcn_dim)
        self.drop_cat = nn.Dropout(config.cross_drop)
        self.transform = nn.Linear(self.gcn_dim * 2, self.gcn_dim)

#         # self.bert = BertModel.from_pretrained(config.pretrain_model_dir)
#         self.xl_net =XLNetModel.from_pretrained(config.pretrain_model_dir)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."
        self.dropout = nn.Dropout(config.dropout)

        # self.emotion_mlp = EmotionMLP(2 * self.gcn_dim, len(set(EMOTION_MAPPING[data_name].values())))
        # self.cause_mlp = CauseMLP(2 * self.gcn_dim)

        self.emotion_mlp = EmotionMLP(2 * self.gcn_dim, len(set(EMOTION_MAPPING[data_name].values())))
        self.cause_mlp = CauseMLP(2 * self.gcn_dim)
        
        # 转化为图神经网络的维度
        
        self.gat_layers = config.gat_layers
        self.graph_attention_size = int(self.gcn_dim / config.num_graph_attention_heads)
        # 图注意力层
        self.GAT_layers = nn.ModuleList([GATLayer(meta_paths= GRAPH_CONFIG_T['meta_paths'],
                                                  in_size=self.gcn_dim, out_size=self.graph_attention_size,
                                                  layer_num_heads=config.num_graph_attention_heads) for _ in range(self.gat_layers)])
        self.ffn_layers = nn.ModuleList([PositionWiseFeedForward(self.gcn_dim, self.gcn_dim, 0.2) for _ in range(self.gat_layers)])
        #self.gru_conversation = nn.GRU(self.gcn_dim, self.gcn_dim, batch_first=False)
        #self.gru_utterance = nn.GRU(self.gcn_dim, self.gcn_dim, batch_first=False)
        # self.hgc_layers = nn.ModuleList([HeteroGraphConvLayer(in_size=self.gcn_dim, out_size=self.graph_attention_size,
        #                                           layer_num_heads=config.num_graph_attention_heads) for _ in range(self.gat_layers)])
        # 情绪原因配对
        self.ec_pairing = PairNN(2 * self.gcn_dim, config.pos_emb_dim, config.rel_pos_k)
        

        
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
        em_logits, ec_logits  = 0, 0
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.size(0)
        # 清理显存的函数
        def clear_cache(*vars):
            for var in vars:
                del var
            torch.cuda.empty_cache()
        if 'textual' in modality:
            # outputs = self.roberta(
            #     input_ids,
            #     attention_mask=attention_masks,
            #     token_type_ids= None,
            #     position_ids=position_ids,
            #     speaker_ids=speaker_ids,
            #     head_mask=head_masks,
            #     inputs_embeds=inputs_embeds,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )
            outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_masks,
                            token_type_ids=token_type_ids)
            # outputs = self.xl_net(input_ids=input_ids,
            #                         attention_mask=attention_masks,
            #                         token_type_ids=token_type_ids)
            # B: batch size
            # N: 对话的话语长度（根据batch进行padding）
            # S：对话输入的序列token长度，填充至max-seq-length
            # D: Roberta的隐藏层维度
            # M: 这个batch的所有图的节点数之和
            # H: 图神经网络的隐层维度

            sequence_outputs_h = outputs[0] # B*S*D
            pooled_outputs = outputs[1]   # B*D
            sequence_outputs_h, _ = self.turnAttention(sequence_outputs_h, sequence_outputs_h, sequence_outputs_h, turn_masks) # turn_masks: B*S*S
            sequence_outputs_h = self._batched_index_select(sequence_outputs_h, uttr_indices, mention_ids) # B*N*D
            sequence_outputs_h = self.linear_t(sequence_outputs_h)
            pooled_outputs = self.linear_t(pooled_outputs).unsqueeze(1)
            # 清理不再使用的变量
            clear_cache(outputs)
            # pooled_outputs = torch.mean(sequence_outputs, dim=1)  # B*D
        if 'audio' in modality:
            conv_audio_in = torch.mean(audio_features, dim=1).unsqueeze(1)
            audio_in = self.linear_a(audio_features)
            conv_audio_in = self.linear_a(conv_audio_in)
            audio_in = self.rume(audio_in, uttr_len)
            conv_audio_in = self.rume(conv_audio_in, uttr_len)
            # 清理不再使用的变量
            clear_cache(audio_features)
        if 'video' in modality:
            conv_video_in = torch.mean(video_features, dim=1).unsqueeze(1)
            video_in = self.linear_v(video_features)
            conv_video_in = self.linear_v(conv_video_in)
            video_in = self.rume(video_in, uttr_len)
            conv_video_in = self.rume(conv_video_in, uttr_len)
            # 清理不再使用的变量
            clear_cache(video_features)


        # audio_mask = (uttr_indices != 0).float()
        # featcross_t, featcross_a = self.acme(sequence_outputs_h, audio_in , audio_mask) # audio_mask: B * S
        # conv_featcross_t, conv_featcross_a = self.acme(pooled_outputs, conv_audio_in, audio_mask[:, 0].unsqueeze(-1))

        # featcross_cat = torch.concat([featcross_t, featcross_a], dim=-1)
        # sequence_outputs_h = self.drop_cat(F.relu(self.linear_cat(featcross_cat)))
        # conv_featcross_cat = torch.concat([conv_featcross_t, conv_featcross_a], dim=-1)
        # pooled_outputs = self.drop_cat(F.relu(self.linear_cat(conv_featcross_cat)))
        # # 清理不再使用的变量
        # clear_cache(featcross_t, featcross_a, conv_featcross_t, conv_featcross_a, featcross_cat, conv_featcross_cat)
        # graph_in = self.transform(sequence_outputs_h) # B*N*H
            

        # initialize graph nodes
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

        
        # 筛选配对
        # em_logits = self.emotion_mlp(graph_in)
        # ca_logits = self.cause_mlp(graph_in)
        # couples_pred, emo_caus_pos = self.ec_pairing(graph_in, graph_in) # B*(N*N) 情绪原因配对logits
        em_logits = self.emotion_mlp(torch.cat([graph_in, gragh_h_emotion], dim = -1)) # B*N*E 情绪类别logits
        ca_logits = self.cause_mlp(torch.cat([graph_in, gragh_h_cause], dim = -1)) # B*N*1 原因类别logits
        emotion_g_h = torch.cat([graph_in, gragh_h_emotion], dim = -1)
        cause_g_h = torch.cat([graph_in, gragh_h_cause], dim = -1)
        couples_pred, emo_caus_pos = self.ec_pairing(emotion_g_h, cause_g_h) # B*(N*N) 情绪原因配对logits
        # em_logits = self.emotion_mlp(gragh_h) # B*N*E 情绪类别logits
        
        return em_logits, ca_logits, couples_pred, emo_caus_pos
    
    def _audio_batched_index_select(self, audio_features, audio_uttr_indices):
        max_uttr_len = audio_uttr_indices.size(1)
        feature_dim = audio_features.size(-1)
        batch_size = audio_features.size(0)
        doc_sents_h = get_cuda(torch.zeros(batch_size, max_uttr_len, feature_dim))
        for i in range(audio_features.size(0)):
            uttr_idx = 1
            while uttr_idx < max_uttr_len and audio_uttr_indices[i, uttr_idx] != 0:
                doc_sents_h[i, uttr_idx-1, :] = torch.mean(audio_features[i, audio_uttr_indices[i][uttr_idx-1] : audio_uttr_indices[i][uttr_idx], :], dim = 0)
                uttr_idx += 1
            if audio_uttr_indices[i, uttr_idx-1] != 0:
                doc_sents_h[i, uttr_idx-1, :] = torch.mean(audio_features[i, audio_uttr_indices[i][uttr_idx-1]:, :], dim=0)
        return doc_sents_h

    
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
        # dummy = uttr_indices.unsqueeze(2).expand(uttr_indices.size(0), uttr_indices.size(1), sequence_outputs.size(2))
        # doc_sents_h = sequence_outputs.gather(1, dummy)
        return doc_sents_h
    # def _batched_index_select(self, sequence_outputs , uttr_indices):
    #     dummy = uttr_indices.unsqueeze(2).expand(uttr_indices.size(0), uttr_indices.size(1), sequence_outputs.size(2))
    #     doc_sents_h = sequence_outputs.gather(1, dummy)
    #     return doc_sents_h
    
    def loss_pre_emo(self, pred_e, gold_e):
        uttr_len = pred_e.size(1)
        gold_e = gold_e[:, :uttr_len] - 1
        pred_e_cpu = pred_e.permute(0, 2, 1).cpu()
        gold_e_cpu = gold_e.cpu()
        # criterion = FocalLoss(reduction='mean', ignore_index=-1)
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1) # 为-1的值被忽略
        loss_e = criterion(pred_e_cpu, gold_e_cpu)
        return loss_e



    def loss_pre_cau(self, pred_c, gold_c):
        uttr_len = pred_c.size(1)
        gold_c = gold_c[:, :uttr_len]
        c_mask = gold_c != -1  # 为 -1 的位置为填充值
        pred_c = pred_c.squeeze(-1).masked_select(c_mask).cpu()
        true_c = gold_c.masked_select(c_mask).float().cpu()
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        loss_c = criterion(pred_c, true_c)
        # loss_c = focal_loss_bce(pred_c, true_c, alpha=1, gamma=2, reduction='mean')
        return loss_c

    def loss_pre_ec(self, couples_pred, emo_cau_pos, ec_pair, uttr_mask, test=False):
        couples_true, couples_mask = self.truncate_ec_pairs(couples_pred, emo_cau_pos, ec_pair, uttr_mask, test)
        couples_mask = torch.BoolTensor(couples_mask).to(DEVICE)
        couples_true = torch.FloatTensor(couples_true).to(DEVICE)
        couples_true = couples_true.masked_select(couples_mask)
        couples_pred = couples_pred.masked_select(couples_mask)
        loss_couple = focal_loss_bce(couples_pred, couples_true, alpha=1, gamma=2, reduction='mean')
        return loss_couple
    # def loss_pre_emo(self, pred_e, gold_e):
    #     uttr_len = pred_e.size(1)
    #     # 这里进行截断
    #     gold_e = gold_e[:,:uttr_len]
    #     gold_e = gold_e[:,:uttr_len]-1 # 计算时还是要从0-6
    #     # 该行代码可能导致计算的不确定性，影响可复现性
    #     # 将预测和标签移至CPU
    #     pred_e_cpu = pred_e.permute(0, 2, 1).cpu()
    #     gold_e_cpu = gold_e.cpu()
    #     criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1) # 为-1的值被忽略
    #     loss_e = criterion(pred_e_cpu, gold_e_cpu)
    #     return loss_e

    # def loss_pre_cau(self, pred_c, gold_c):
    #     uttr_len = pred_c.size(1)
    #     # 这里进行截断
    #     gold_c = gold_c[:,:uttr_len]
    #     c_mask = gold_c != -1  # 为 -1 的位置为填充值
    #     pred_c = pred_c.squeeze(-1).masked_select(c_mask).cpu()
    #     true_c = gold_c.masked_select(c_mask).float().cpu()
    #     criterion = nn.BCEWithLogitsLoss(reduction='mean')
    #     loss_c = criterion(pred_c, true_c)
    #     return loss_c


    # def loss_pre_ec(self, couples_pred, emo_cau_pos, ec_pair, uttr_mask, test = False):
    #     couples_true, couples_mask = self.truncate_ec_pairs(couples_pred, emo_cau_pos, ec_pair, uttr_mask, test)
    #     couples_mask = torch.BoolTensor(couples_mask).to(DEVICE)
    #     couples_true = torch.FloatTensor(couples_true).to(DEVICE)
    #     criterion = nn.BCEWithLogitsLoss(reduction='mean')
    #     couples_true = couples_true.masked_select(couples_mask)
    #     couples_pred = couples_pred.masked_select(couples_mask)
    #     loss_couple = criterion(couples_pred, couples_true)
    #     return loss_couple

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
    

class EmotionMLP(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(EmotionMLP, self).__init__()
        self.emotion_mlp_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.emotion_mlp_2 = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, sequence_outputs_h):
        em_hiddens_h = self.emotion_mlp_1(sequence_outputs_h)
        em_logits = self.emotion_mlp_2(sequence_outputs_h)
        return em_logits
    
class CauseMLP(nn.Module):
    def __init__(self, hidden_dim):
        super(CauseMLP, self).__init__()
        self.cause_mlp_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.cause_mlp_2 = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, sequence_outputs_h):
        cause_hiddens_h = self.cause_mlp_1(sequence_outputs_h)
        ca_logits = self.cause_mlp_2(sequence_outputs_h)
        return ca_logits
    

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

# class PositionwiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

#     def forward(self, x):
#         residual = x
#         x = F.gelu(self.fc1(x))
#         x = self.fc2(x)
#         x = self.dropout(x)
#         x += residual
#         x = self.layer_norm(x)
#         return x
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, pred, target):
        logpt = -self.ce(pred, target)
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

def focal_loss_bce(pred, target, alpha=1, gamma=2, reduction='mean'):
    bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * ((1 - pt) ** gamma) * bce_loss
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss