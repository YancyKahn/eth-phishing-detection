#预处理数据，将提取交易数据并清洗

import os
import warnings

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

#读取数据
def load_data_1d(path_data_folder_1d, label, isetherscan=False, timeseries_len=50, columns_key = {'timeStamp': 'TimeStamp', 'value': 'Value', 'from': 'From', 'to': 'To'}):
    node_features = pd.DataFrame()
    node_graphs = {}
    node_tx_timeseries = pd.DataFrame()

    index = 0
    len_files = len(os.listdir(path_data_folder_1d))
    for filename in os.listdir(path_data_folder_1d):

        print("Preproocess: {}, {}/{}".format(filename, index, len_files), end='\r')
        index += 1

        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(path_data_folder_1d, filename))
            except:
                continue
            
            # 交易数量大于0
            if len(df) < 1:
                continue
            
            # 交易数量大于1000时取后1000条交易
            if len(df) > 1000:
                df = df.iloc[-1000:, :]

            # rename columns
            if 'TimeStamp' not in df.columns or 'Value' not in df.columns or 'From' not in df.columns or 'To' not in df.columns:
                df = df.rename(columns = columns_key)
            
            # astype
            df['Value'] = df['Value'].astype(float)
            if isetherscan:
                df['Value'] = df['Value'] / 10 ** 18
            df['TimeStamp'] = df['TimeStamp'].astype(int)
            
            df = df.sort_values(by = 'TimeStamp')
            df = df.reset_index(drop = True)

            # 1. 获取节点子图
            G = nx.DiGraph()
            # 将data按照From和To分组，然后对每个分组进行遍历，将value相加
            for name, group in df.groupby(['From', 'To']):
                G.add_edge(name[0], name[1], value = group['Value'].sum(), timestamp = group['TimeStamp'].max(), count = len(group))

            node_graphs[filename.split('.')[0]] = G

            # 2. 计算节点特征
            features = {}
            features['address'] = filename.split('.')[0]
            features['value_out'] = df[df['From'] == filename.split('.')[0]]['Value'].sum()
            features['value_in'] = df[df['To'] == filename.split('.')[0]]['Value'].sum()
            features['balance'] = features['value_out'] - features['value_in']
            features['degree'] = len(df)
            features['degree_in'] = len(df[df['To'] == filename.split('.')[0]])
            features['degree_out'] = len(df[df['From'] == filename.split('.')[0]])
            features['max_value'] = df['Value'].max()
            features['min_value'] = df['Value'].min()
            features['mean_value'] = df['Value'].mean()
            features['std_value'] = df['Value'].std()
            features['median_value'] = df['Value'].median()
            features['label'] = label

            node_features = node_features.append(features, ignore_index = True)

            # 3. 预处理的节点交易, 长度为timeseries_len
            df.loc[df['From'] == filename.split('.')[0], 'Value'] = -df['Value']
            df['Balance'] = df['Value'].cumsum()

            values = df['Balance'].values
            if len(values) < timeseries_len:
                values = np.pad(values, (0, timeseries_len - len(values)), 'constant', constant_values = values[-1])
            else:
                values = values[-timeseries_len:]
            
            # 第一列为地址，最后一列为标签
            values = values.tolist()
            values.insert(0, filename.split('.')[0])
            values.append(label)
            # print(values)

            columns  = ['balance_{}'.format(i) for i in range(timeseries_len)]
            columns = ['address'] + columns + ['label']
            
            node_tx_timeseries = node_tx_timeseries.append(pd.DataFrame([values], columns = columns), ignore_index = True)

    node_tx_timeseries.fillna(0, inplace = True)
    print()

    return node_features, node_graphs, node_tx_timeseries


def test():
    # 钓鱼数据
    path_phishing_data_folder_1d = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\original_data\etherscan\1d\phish-hack'
    phishing_node_features, phishing_node_graphs, phishing_node_tx = load_data_1d(path_phishing_data_folder_1d, 0, isetherscan=True)
    print()
    print(phishing_node_features.shape, len(phishing_node_graphs), phishing_node_tx.shape)
    # 正常数据
    path_normal_data_folder_1d = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\original_data\open\非钓鱼一阶节点'
    normal_node_features, normal_node_graphs, normal_node_tx = load_data_1d(path_normal_data_folder_1d, 1, isetherscan=False)
    print()
    print(normal_node_features.shape, len(normal_node_graphs), normal_node_tx.shape)

    # 合并数据
    node_features = pd.concat([phishing_node_features, normal_node_features], axis = 0)
    node_features.to_csv(r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\datasets\etherscan\node_features.csv', index = False)

if __name__ == '__main__':
    test()