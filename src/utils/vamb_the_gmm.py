import numpy as np
from sklearn.mixture import GaussianMixture
import csv
import argparse

def fit_gmm(latents, contignames, output_csv_path, num_bins):

    gmm = GaussianMixture(n_components=num_bins, random_state=2021)
    predicts = gmm.fit_predict(latents)
    predic_probs = gmm.predict_proba(latents)
    max_predic_probs = np.max(predic_probs, axis=1)
    # delete_idx = np.where(max_predic_probs < 0.50)

    # contignames = np.delete(contignames, delete_idx, axis=0)
    # predicts = np.delete(predicts, delete_idx, axis=0)

    # results = []
    # for cluster, contig in zip(predicts, contignames):
    #     results.append((cluster, contig))
    with open(output_csv_path, 'w') as f:
        for contig, cluster in zip(contignames, predicts):
            f.write(f'{cluster}\t{contig}\n')
            # contig = contig.split('_')[1]
            # f.write(f'NODE_{contig},{cluster}\n')

if __name__ == "__main__":
    # ------------------for vamb------------
    parser = argparse.ArgumentParser(description='Split samples before binning')
    parser.add_argument('--latent_path', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/latents/latent_80_21141_best.npy', help='Fasta path (.npy)')
    parser.add_argument('--contignames_path', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/contignames.npz', help='Contignames path (.npz)')
    parser.add_argument('--output_csv_path', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/gmm.csv', help='Output path for gmm result')
    parser.add_argument('--num_bins', type=int, default=115, help='Number of bins you want')


    args = parser.parse_args()
    # latent_path = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/medium/deepmetabin_001/cami-m2-latent/latent_epoch_195.npy"
    # contignames_path = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/medium/deepmetabin_001/cami-m2-latent/contignames.npy"
    # output_csv_path = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/medium/deepmetabin_001/cami-m2-latent/gmm.csv"
    # mask_path = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/vamb_2000/vamb_out/mask.npz"


    # latent = np.load(latent_path)
    # latent = latent['arr_0']
    # contignames = np.load(contignames_path)
    # contignames = contignames['arr_0']
    # mask = np.load(mask_path)
    # mask = mask['arr_0']

    # contignames = [c for c, m in zip(contignames, mask) if m]

# ---------------------------
    latent = np.load(args.latent_path)
    contignames = np.load(args.contignames_path)['arr_0']
    # contignames = np.squeeze(contignames)
    contignames = contignames.tolist()
    # mask = np.load(mask_path)
    # mask = mask['arr_0']
    # contignames = ['NODE_'+str(m) for m in contignames]

    fit_gmm(latent, contignames, args.output_csv_path, args.num_bins)