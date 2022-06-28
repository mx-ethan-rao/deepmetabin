import numpy as np
from sklearn.decomposition import PCA

if __name__ == "__main__":
    """
    X = np.array([
        [-1, -1],
        [-2, -1],
        [-3, -2],
        [1, 1],
        [2, 1],
        [3, 2],
    ])
    """
    pca = PCA(n_components=10)
    path = "/home/gaoweicong/cami1-low-ag-graph.npy"
    load_feat = np.load(path, allow_pickle=True)
    print(load_feat.shape)
    out_feat = pca.fit_transform(load_feat)
    print(out_feat.shape)
    save_path = "/home/gaoweicong/cami1-low-ag-feat.npy"
    np.save(save_path, out_feat)
