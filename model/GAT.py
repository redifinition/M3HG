import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import dgl

from model.PositionWiseGATConv import PositionWiseGATConv
from utils.global_variables import GRAPH_CONFIG_T

class GATLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout=0.2):
        super(GATLayer, self).__init__()

        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(PositionWiseGATConv(in_feats = in_size, out_feats = out_size, num_heads = layer_num_heads, 
                                           feat_drop = dropout, attn_drop = dropout, activation =F.leaky_relu,
                                           allow_zero_in_degree=True))
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self._cached_graph = None # 用于存储当前正在处理的图对象 g
        self._cached_coalesced_graph = {} # 用于存储按元路径生成的可达子图（coalesced graphs）

    def forward(self, g, h_dict): # h_dict 用于存储不同类型的节点特征
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, [meta_path]) # 将不同元路径的可达子图存储在 _cached_coalesced_graph 中
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            # 这里需要注意获取结果的顺序一定是先对话节点再话语节点
            if meta_path[0] == meta_path[-1]: # 如果元路径的第一个节点类型和最后一个节点类型相同，则说明是对话节点之间的元路径
                hidden_embeddings = self.gat_layers[i](new_g, h_dict[meta_path[0]]).flatten(1)
            else:
                hidden_embeddings = self.gat_layers[i](new_g, (h_dict[meta_path[0]], h_dict[meta_path[-1]])).flatten(1)
                # 由于这里只有指定节点类型的特征进行了自环更新，所以还要添加另一个类型的节点特征
            if meta_path[-1] == 'utterance':
                hidden_embeddings = torch.cat([h_dict['conversation'], hidden_embeddings,h_dict['emotion'],h_dict['cause']], dim=0)
            elif meta_path[-1] == 'conversation':
                hidden_embeddings = torch.cat([hidden_embeddings, h_dict['utterance'],h_dict['emotion'],h_dict['cause']], dim=0)
            elif meta_path[-1] == 'emotion':
                hidden_embeddings = torch.cat([h_dict['conversation'], h_dict['utterance'],hidden_embeddings,h_dict['cause']], dim=0)
            elif meta_path[-1] == 'cause':
                hidden_embeddings = torch.cat([h_dict['conversation'], h_dict['utterance'],h_dict['emotion'], hidden_embeddings], dim=0)
            semantic_embeddings.append(hidden_embeddings)
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1) # Node_num * Meta_path_num * Embedding_dim
        return self.semantic_attention(semantic_embeddings)
    



class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0) # 大小为meta path的数量
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)