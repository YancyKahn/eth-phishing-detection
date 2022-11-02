# eth-phishing-detection
Ethereum Phishing Fraud Account Detection

# Dataset
origin data from xblock.pro
* Yuan Z, Yuan Q, Wu J. Phishing detection on Ethereum via learning representation of transaction subgraphs[C]//International Conference on Blockchain and Trustworthy Systems. Springer, Singapore, 2020: 178-191.

* Collection in Etherscan

# Experiments
1. trans2vec -> notebook/random_walk_classification.ipynb
2. gcn -> notebook/gcn_classification.ipynb
3. graph2vec -> notebook/graph2vec_classification.ipynb
4. shapelets -> motifs_mining_timeseries.ipynb
5. graphlets -> motifs_mining_topo.ipynb
6. translets -> translets_mining.ipynb


# env requirements(windows)
* base

```
pip install -r requirements.txt
```

* torch and torch-geometric
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
```
