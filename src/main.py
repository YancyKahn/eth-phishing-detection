import pandas as pd 
import src.utils.preprocess as preprocess
import src.utils.graphlets as graphlets
import src.utils.shapelets as shapelets

def process():
    # 1. load data
    # phihsing data path and normal data path
    path_data_phishing = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\original_data\etherscan\1d\phish-hack'
    path_data_normal = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\original_data\etherscan\1d\normal'

    # 2. preprocess data
    phishing_node_features, phishing_node_graphs = preprocess.load_data_1d(path_data_phishing, 0, isetherscan=True)
    normal_node_features, normal_node_graphs = preprocess.load_data_1d(path_data_normal, 1, isetherscan=True)

    print('phishing_node_features: ', phishing_node_features.shape)
    print('normal_node_features: ', normal_node_features.shape)

    # 3. extract graphlet features
    phishing_graphlet_features = graphlets.extract_graphlets(phishing_node_graphs, 5, 30, 0.8)
    normal_graphlet_features = graphlets.extract_graphlets(normal_node_graphs, 5, 30, 0.8)

    # 4. extract shapelet features

if __name__ == '__main__':
    process()