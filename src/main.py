import os
import pickle

import pandas as pd

import utils.graphlets as graphlets
import utils.preprocess as preprocess
import utils.shapelets as shapelets

# 1. load data
# phihsing data path and normal data path from ethscan
path_data_phishing = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\original_data\etherscan\1d\phish-hack'
path_data_normal = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\original_data\etherscan\1d\normal'
path_data_normal2 = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\original_data\open\非钓鱼一阶节点'
base_save_path = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\datasets\etherscan'


def process(dataset_saved_folder, dataset_flag, is_save_base, is_save_graphlets, is_save_shapelets, is_save_features, isNotPreprocess):
    save_floder = dataset_saved_folder + dataset_flag + '\\'
    

    if isNotPreprocess:
        # 2. preprocess data
        print("preprocess data...")
        phishing_node_features, phishing_node_graphs, phishing_node_tx = preprocess.load_data_1d(path_data_phishing, 0, isetherscan=True)
        print("phishing: ", phishing_node_features.shape, len(phishing_node_graphs), phishing_node_tx.shape)

        normal_node_features, normal_node_graphs, normal_node_tx = preprocess.load_data_1d(path_data_normal, 1, isetherscan=True)
        print("normal: ", normal_node_features.shape, len(normal_node_graphs), normal_node_tx.shape)

        normal2_node_features, normal2_node_graphs, normal2_node_tx = preprocess.load_data_1d(path_data_normal2, 1, isetherscan=False)
        print("normal2: ", normal2_node_features.shape, len(normal2_node_graphs), normal2_node_tx.shape)

        # 2.1 merge normal and normal2
        normal_node_features = pd.concat([normal_node_features, normal2_node_features], axis=0)
        normal_node_graphs.update(normal2_node_graphs)
        normal_node_tx = pd.concat([normal_node_tx, normal2_node_tx], axis=0)

        # 2.2 save data
        print("save base data...")
        if is_save_base:
            # 保存基础数据
            phishing_node_features.to_csv(save_floder + 'phishing_node_features.csv', index=False)
            phishing_node_tx.to_csv(save_floder + 'phishing_node_tx.csv', index=False)
            normal_node_features.to_csv(save_floder + 'normal_node_features.csv', index=False)
            normal_node_tx.to_csv(save_floder + 'normal_node_tx.csv', index=False)

            # 保存图数据
            with open(save_floder + 'phishing_node_graphs.pkl', 'wb') as f:
                pickle.dump(phishing_node_graphs, f)
            with open(save_floder + 'normal_node_graphs.pkl', 'wb') as f:
                pickle.dump(normal_node_graphs, f)
            
            print("save base data done.")

    else:
        # 2.3 load data
        print("load base data...")
        phishing_node_features = pd.read_csv(save_floder + 'phishing_node_features.csv')
        phishing_node_tx = pd.read_csv(save_floder + 'phishing_node_tx.csv')
        normal_node_features = pd.read_csv(save_floder + 'normal_node_features.csv')
        normal_node_tx = pd.read_csv(save_floder + 'normal_node_tx.csv')

        print("Number of Phishing: ", len(phishing_node_features))
        print("Number of Normal: ", len(normal_node_features))

        print("Number of Phishing Edges: ", len(phishing_node_tx))
        print("Number of Normal Edges: ", len(normal_node_tx))

        with open(save_floder + 'phishing_node_graphs.pkl', 'rb') as f:
            phishing_node_graphs = pickle.load(f)
        with open(save_floder + 'normal_node_graphs.pkl', 'rb') as f:
            normal_node_graphs = pickle.load(f)
        print("load base data done.")

    
    print("*"*40)

    # 3. merge phishing and normal
    print("merge phishing and normal...")
    node_features = pd.concat([phishing_node_features, normal_node_features], axis=0)
    node_tx = pd.concat([phishing_node_tx, normal_node_tx], axis=0)

    print("*"*40)

    # 4. extract graphlet features
    print("extract graphlet features...")

    if is_save_graphlets:
        phishing_graphlet_features = graphlets.extract_graphlets(phishing_node_graphs, 5, 30, 0.8, 0)
        normal_graphlet_features = graphlets.extract_graphlets(normal_node_graphs, 5, 30, 0.8, 1)
    # 4.1 merge phishing and normal
        graphlet_features = pd.concat([phishing_graphlet_features, normal_graphlet_features], axis=0)
        # 4.1 save graphlet features
        print("save graphlet features...")

        graphlet_features.to_csv(save_floder + 'graphlet_features.csv', index=False)
        print("save graphlet features done.")

    
    print("*"*40)

    # 5. extract shapelet features
    print("extract shapelet features...")
    if is_save_shapelets:
        phishing_shapelet_features, shapelets_vector = shapelets.extract_shapelets(node_tx, 10, 10, 10)
        
        print("save shapelet features...")
        phishing_shapelet_features.to_csv(save_floder + 'phishing_shapelet_features.csv', index=False)
        shapelets_vector.to_csv(save_floder + 'shapelets_vector.csv', index=False)

        print("save shapelet features done.")

    # 6. merge features

    print("*"*40)

    if is_save_features:
        print("merge features...")
        # 按照address合并
        features = pd.merge(node_features.drop('label', axis=1), graphlet_features.drop('label', axis=1), on='address', how='left')
        features = pd.merge(features, phishing_shapelet_features, on='address', how='left')
    
        # features = pd.merge(node_features, graphlet_features.drop('label', axis=1), on='address', how='left')   #TODO all examples 

        # 7. save features
        print("save features...")
        features.to_csv(base_save_path + r'\ex_2\features.csv', index=False)
        shapelets_vector.to_csv(base_save_path + r'\ex_2\shapelets_vector.csv', index=False)


if __name__ == '__main__':
    embedding = True
    dataset_saved_folder = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\datasets\etherscan\\'
    dataset_flag = 'ex_2'

    is_save_base = False
    is_save_graphlets = False
    is_save_shapelets = False
    is_save_features = False
    is_not_preprocess = False


    if not os.path.exists(dataset_saved_folder + dataset_flag):
        os.mkdir(dataset_saved_folder + dataset_flag)

    process(dataset_saved_folder, dataset_flag, is_save_base, is_save_graphlets, is_save_shapelets, is_save_features, is_not_preprocess)