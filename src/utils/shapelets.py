# 提取时间序列的shapelet
import pandas as pd
from pyts.transformation import ShapeletTransform


# 提取shapelets特征
def extract_shapelets(node_txs, min_len, max_len, num_shapelets):

    X = node_txs.iloc[:, 1:-1]
    y = node_txs.iloc[:, -1]

    st = ShapeletTransform(n_shapelets=num_shapelets, window_sizes=[min_len, max_len], verbose=1, random_state=0, n_jobs=8)

    st.fit(X, y)

    # 提取shapelet
    shapelets = st.shapelets_

    # 提取特征
    X_transformed = st.transform(X)
    shapelets_features = pd.DataFrame(X_transformed)
    # 重新进行索引
    shapelets_features.index = node_txs.index

    # 保存shapelets
    shapelets = pd.DataFrame(shapelets)
    shapelets.to_csv(r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\datasets\etherscan\ex_2\shapelets.csv', index=False)

    # 保存shapelets特征
    shapelets_features.to_csv(r'X:\Datasets\Blockchain\xblock.pro\eth-phishing-detection\datasets\etherscan\ex_2\shapelets_features.csv', index=False)

    shapelets_features['label'] = node_txs['label']
    shapelets_features['address'] = node_txs['address']

    # address放到第一列
    cols = shapelets_features.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    shapelets_features = shapelets_features[cols]

    print(shapelets_features.tail())

    return shapelets_features, shapelets