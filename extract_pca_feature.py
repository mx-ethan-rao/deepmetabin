import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import fractional_matrix_power


def cal_laplacian_matrix(ag_adj_matrix):
    I = np.identity(ag_adj_matrix.shape[0])
    D = np.diag(np.sum(ag_adj_matrix, axis=0))
    D_half_norm = fractional_matrix_power(D, -0.5)
    adj_normalized_matrix = D_half_norm.dot(ag_adj_matrix).dot(D_half_norm)
    laplacian_matrix = I - adj_normalized_matrix
    return laplacian_matrix


if __name__ == "__main__":
    pca = PCA(n_components=10)
    path = "/home/gaoweicong/cami1-low-ag-graph.npy"
    load_feat = np.load(path, allow_pickle=True)
    print(load_feat.shape)
    laplacian_matrix = cal_laplacian_matrix(load_feat)
    out_feat = pca.fit_transform(load_feat)
    print("test for the shape of output laplacian matrix {}".format(out_feat.shape))
    save_path = "/home/eddie/cami1-low-ag-lap-feat.npy"
    np.save(save_path, out_feat)
