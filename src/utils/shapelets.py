# 提取时间序列的shapelet
from pyts.transformation import ShapeletTransform
import pandas as pd 

# 提取shapelets特征
def extract_shapelets(node_txs, min_len, max_len, num_shapelets):
    # 提取shapelets特征
    shapelet_features = {}

    X = node_txs.iloc[:, 1:-1]
    y = node_txs.iloc[:, -1]

    st = ShapeletTransform(n_shapelets=num_shapelets, window_sizes=[min_len, max_len], verbose=1, random_state=0, n_jobs=8)

    st.fit(X, y)

    # 提取shapelet
    shapelets = st.shapelets_

    # 提取特征
    X_transformed = st.transform(X)
    shapelets_features = pd.DataFrame(X_transformed)
    shapelets_features['label'] = node_txs['label']
    shapelets_features['address'] = node_txs['address']

    cols = shapelets_features.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    shapelets_features = shapelets_features[cols]

    return shapelets_features, shapelets