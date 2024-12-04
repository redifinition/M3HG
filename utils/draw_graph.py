import dgl

import networkx as nx
import matplotlib.pyplot as plt

def visualize_dgl_graph(dgl_graph):
    # 将DGL异构图转换为NetworkX图
    nx_graph = dgl.to_networkx(dgl_graph)

    # 定义布局
    pos = nx.spring_layout(nx_graph)
    
    # 预定义不同边类型的颜色和样式
    edge_colors = {
        'global_t': '#0C1844',
        'global_a': '#0C1844',
        'global_v': '#0C1844',
        'same_speaker_t': '#C80036',
        'same_speaker_a': '#C80036',
        'same_speaker_v': '#C80036',
        'different_speaker_t': '#83B4FF',
        'different_speaker_a': '#83B4FF',
        'different_speaker_v': '#83B4FF',
        'emotional_link_t': '#FFD700',
        'emotional_link_a': '#FFD700',
        'emotional_link_v': '#FFD700',
        'causal_link_t': '#FFA500',
        'causal_link_a': '#FFA500',
        'causal_link_v': '#FFA500',
        't_a': '#FFA500',
        't_v': '#FFA500',
        'a_v': '#FFA500',
    }
    edge_styles = {
        'global_t': 'dotted',   # 使用有效的线样式
        'global_a': 'dotted',
        'global_v': 'dotted',
        'same_speaker_t': 'solid',
        'same_speaker_a': 'solid',
        'same_speaker_v': 'solid',
        'different_speaker_t': 'solid',
        'different_speaker_a': 'solid',
        'different_speaker_v': 'solid',
        'emotional_link_t': 'solid',
        'emotional_link_a': 'solid',
        'emotional_link_v': 'solid',
        'causal_link_t': 'solid',
        'causal_link_a': 'solid',
        'causal_link_v': 'solid',
        't_a': 'solid',
        't_v': 'solid',
        'a_v': 'solid',

    }
    
    # 预定义不同节点类型的颜色
    node_colors = {
        'utterance_t': ['#F1F8E8', '#55AD9B'],
        'utterance_a': ['#F1F8E8', '#55AD9B'],
        'utterance_v': ['#F1F8E8', '#55AD9B'],
        'conversation_t': ['#FEFFD2', '#FFBF78'],
        'conversation_a': ['#FEFFD2', '#FFBF78'],
        'conversation_v': ['#FEFFD2', '#FFBF78'],
        'emotion': ['#FFD700', '#FFA500'],
        'cause': ['#FFA500', '#FFD700']
    }
    
    # 绘制不同类型的边
    for u, v, data in nx_graph.edges(data=True):
        etype = data['etype'][1]  # 从元组中获取边的类型
        nx.draw_networkx_edges(
            nx_graph, pos,
            edgelist=[(u, v)],
            arrowstyle='-|>',
            arrowsize=10,
            edge_color=edge_colors.get(etype, 'black'),  # 为边选择颜色
            style=edge_styles.get(etype, 'solid')        # 为边选择样式
        )
    
    # 绘制节点
    node_handles = {}
    for ntype in dgl_graph.ntypes:
        nodelist = [n for n, data in nx_graph.nodes(data=True) if data['ntype'] == ntype]
        node_color = node_colors[ntype][0]
        edge_color = node_colors[ntype][1]
        node_draw = nx.draw_networkx_nodes(nx_graph, pos, nodelist=nodelist, node_color=node_color, node_size=300, edgecolors=edge_color, linewidths=1)
        node_handles[ntype] = node_draw

    nx.draw_networkx_labels(nx_graph, pos, font_size=8, font_family='sans-serif')

    # 为图例收集句柄
    edge_handles = []
    for etype, color in edge_colors.items():
        edge_handles.append(plt.Line2D([0], [0], color=color, linestyle=edge_styles[etype], linewidth=2))
    # 创建边和节点的图例
    edge_legend = plt.legend(edge_handles, edge_colors.keys(), title="Edge Types", fontsize='small', loc='upper left', markerscale=0.7)
    node_legend = plt.legend(handles=list(node_handles.values()), labels=list(node_handles.keys()), title="Node Types", fontsize='small', loc='upper right', handlelength=2, handleheight=2,markerscale=0.7)
    plt.gca().add_artist(node_legend)
    plt.gca().add_artist(edge_legend)
    # 显示图形
    plt.title('Visualizing DGL Graph with Edge and Node Types')
    plt.axis('off')  # 关闭坐标轴
    plt.savefig('dgl_graph.pdf')