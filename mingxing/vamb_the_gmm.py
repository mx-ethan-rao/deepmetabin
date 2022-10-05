import numpy as np
from sklearn.mixture import GaussianMixture
import vamb
import csv
import argparse

def fit_gmm(latents, contignames):

    gmm = GaussianMixture(n_components=14, covariance_type='diag', max_iter=100, reg_covar=1e-2)
    predicts = gmm.fit_predict(latents)
    # results = []
    # for cluster, contig in zip(predicts, contignames):
    #     results.append((cluster, contig))
    with open(output_csv_path, 'w') as f:
        for contig, cluster in zip(contignames, predicts):
            f.write(f'{cluster}\t{contig}\n')

if __name__ == "__main__":
    # ------------------for vamb------------
    latent_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/gmvae_1000/latent600.npy"
    contignames_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/gmvae_1000/id.npy"
    output_csv_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/gmvae_1000/gmm_epoch_600_9_20.csv"
    # mask_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/vamb_2000/vamb_out/mask.npz"


    # latent = np.load(latent_path)
    # latent = latent['arr_0']
    # contignames = np.load(contignames_path)
    # contignames = contignames['arr_0']
    # mask = np.load(mask_path)
    # mask = mask['arr_0']

    # contignames = [c for c, m in zip(contignames, mask) if m]

# ---------------------------
    latent = np.load(latent_path)
    contignames = np.load(contignames_path)
    # contignames = np.squeeze(contignames)
    contignames = contignames.tolist()
    # mask = np.load(mask_path)
    # mask = mask['arr_0']
    # contignames = ['NODE_'+str(m) for m in contignames]

    fit_gmm(latent, contignames)