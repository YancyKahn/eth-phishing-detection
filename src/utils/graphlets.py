# 提取以太坊交易图的graphlets
import random
import pyfpgrowth
import networkx as nx 
import pandas as pd 

# 带重启的随机游走，有概率返回到起点，有概率继续游走, 当游走到终点时，返回到起点, 可以反向游走
def random_walk(G, start, walk_length, restart_prob):
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        if random.random() < restart_prob:
            walk.append(random.choice(list(G.nodes())))
        else:
            if len(list(G.neighbors(cur))) == 0:
                walk.append(random.choice(list(G.nodes())))
            else:
                walk.append(random.choice(list(G.neighbors(cur))))

    # 获取游走子图
    subgraph = G.subgraph(walk)
    return subgraph

# 生成游走子图
def generate_walks(G, num_walks, walk_length, restart_prob):
    walks = []
    nodes = list(G.nodes())

    while len(walks) < num_walks:
        random.shuffle(nodes)
        random_start_node = random.choice(nodes)
        graph = random_walk(G, random_start_node, walk_length, restart_prob)
        # 是否为存在孤立节点的子图
        if len(list(nx.isolates(graph))) == 0:
            walks.append(graph)
            
    return walks


# 提取以太坊交易图的graphlets
def extract_graphlets(node_graphs, num_walks, walk_length, restart_prob):
    motifs = {
    'motif1': nx.DiGraph([(0, 1), (1, 2)]),
    'motif2': nx.DiGraph([(0, 1), (2, 1)]),
    'motif3': nx.DiGraph([(0, 1), (0, 2)]),
    }

    # 提取motifs特征
    motif_features = {}

    # 从node_graphs中提取motifs特征, 统计motifs的数量
    
    length = len(walks)
    index = 0

    for key in node_graphs.keys():
        print("Process: {}, {}/{}".format(key, index, length))
        index += 1
        # 提取motifs特征
        motif_features[key] = {}
        for motif in motifs.keys():
            motif_features[key][motif] = 0
            if motif == 'motif1':
                motif_features[key][motif + '_00'] = 0
                motif_features[key][motif + '_01'] = 0
                motif_features[key][motif + '_10'] = 0
                motif_features[key][motif + '_11'] = 0
                
        node_subgraph = node_graphs[key]

        # 找出sub_graph中所有匹配motifs的结构， 并统计数量
        # 随机游走获取子图
        walks = generate_walks(node_graphs[key], 5, 30, 0.8)

        # sub_graph = node_graphs[key]
        
        for walk in walks:
            for motif in motifs:
                for subgraph in nx.algorithms.isomorphism.GraphMatcher(walk, motifs[motif]).subgraph_isomorphisms_iter():
                    motif_features[key][motif] += 1
                    # 仅对motif1进行再计算, 引入边权
                    if motif == 'motif1':
                        # 获取node_subgraph中的边权
                        edges = list(subgraph.keys())
                        value1 = node_subgraph.edges[edges[0], edges[1]]['value']
                        value2 = node_subgraph.edges[edges[1], edges[2]]['value']
                        timestamp1 = node_subgraph.edges[edges[0], edges[1]]['timestamp']
                        timestamp2 = node_subgraph.edges[edges[1], edges[2]]['timestamp']

                        if value1 < value2 and timestamp1 < timestamp2:
                            motif_features[key][motif + '_00'] += 1
                        elif value1 < value2 and timestamp1 > timestamp2:
                            motif_features[key][motif + '_01'] += 1
                        elif value1 > value2 and timestamp1 < timestamp2:
                            motif_features[key][motif + '_10'] += 1 
                        elif value1 > value2 and timestamp1 > timestamp2:
                            motif_features[key][motif + '_11'] += 1

    # 将motif_features转化为df
    motif_features = pd.DataFrame(motif_features)
    motif_features = motif_features.T

    return motif_features