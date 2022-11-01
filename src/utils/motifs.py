# 定义常用的motifs
# 1. 不带权权重的motifs
# 2. 带权重的motifs

import networkx as nx

motifs_without_weight = {
    # 三个节点的motifs
    'motifs_3_1': nx.DiGraph([(0, 1), (1, 2)]),
    'motifs_3_2': nx.DiGraph([(0, 1), (0, 2)]),
    'motifs_3_3': nx.DiGraph([(1, 0), (2, 0)]),
    'motifs_3_4': nx.DiGraph([(0, 1), (1, 2), (2, 0)]),
    'motifs_3_5': nx.DiGraph([(0, 1), (1, 2), (0, 2)]),
}