import pandas as pd 
import utils.preprocess as preprocess
import utils.graphlets as graphlets
import utils.shapelets as shapelets
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def process(isNotPreprocess=False):
    # 1. load data
    # phihsing data path and normal data path
    path_data_phishing = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\original_data\etherscan\1d\phish-hack'
    path_data_normal = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\original_data\etherscan\1d\normal'
    path_data_normal2 = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\original_data\open\非钓鱼一阶节点'
    base_save_path = r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\datasets\etherscan'

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
        phishing_node_features.to_csv(base_save_path + r'\phishing_node_features.csv', index=False)
        phishing_node_tx.to_csv(base_save_path + r'\phishing_node_tx.csv', index=False)
        normal_node_features.to_csv(base_save_path + r'\normal_node_features.csv', index=False)
        normal_node_tx.to_csv(base_save_path + r'\normal_node_tx.csv', index=False)

        with open(base_save_path + r'\phishing_node_graphs.pkl', 'wb') as f:
            pickle.dump(phishing_node_graphs, f)
        with open(base_save_path + r'\normal_node_graphs.pkl', 'wb') as f:
            pickle.dump(normal_node_graphs, f)
        print("save base data done.")
    else:
        # 2.3 load data
        print("load base data...")
        phishing_node_features = pd.read_csv(base_save_path + r'\phishing_node_features.csv')
        phishing_node_tx = pd.read_csv(base_save_path + r'\phishing_node_tx.csv')
        normal_node_features = pd.read_csv(base_save_path + r'\normal_node_features.csv')
        normal_node_tx = pd.read_csv(base_save_path + r'\normal_node_tx.csv')

        with open(base_save_path + r'\phishing_node_graphs.pkl', 'rb') as f:
            phishing_node_graphs = pickle.load(f)
        with open(base_save_path + r'\normal_node_graphs.pkl', 'rb') as f:
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
    phishing_graphlet_features = graphlets.extract_graphlets(phishing_node_graphs, 5, 30, 0.8, 0)
    normal_graphlet_features = graphlets.extract_graphlets(normal_node_graphs, 5, 30, 0.8, 1)
    # 4.1 merge phishing and normal
    graphlet_features = pd.concat([phishing_graphlet_features, normal_graphlet_features], axis=0)

    print("*"*40)

    # 5. extract shapelet features
    print("extract shapelet features...")
    phishing_shapelet_features, shapelets_vector = shapelets.extract_shapelets(node_tx, 8, 10, 10)

    # 6. merge features
    print("merge features...")
    # 按照address合并
    features = pd.merge(node_features.drop('label', axis=1), graphlet_features.drop('label', axis=1), on='address', how='left')
    features = pd.merge(features, phishing_shapelet_features, on='address', how='left')

    print("*"*40)

    # 7. save features
    print("save features...")
    features.to_csv(base_save_path + r'\features.csv', index=False)
    shapelets_vector.to_csv(base_save_path + r'\shapelets_vector.csv', index=False)


if __name__ == '__main__':
    embedding = False
    if embedding:
        process()
    else:
        # 读取数据
        data = pd.read_csv(r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\datasets\etherscan\features.csv')

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(data.drop(['label', 'address'], axis=1), data['label'], test_size=0.2, random_state=0)

        # 训练模型
        import xgboost
        model = xgboost.XGBClassifier()

        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 评估
        print(classification_report(y_test, y_pred))