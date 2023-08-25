from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import DistanceMetric
from sympy import centroid

def compute_inter_class_distance(labels, feats):
    clf = NearestCentroid()
    clf.fit(feats, labels)
    dist = DistanceMetric.get_metric('euclidean')
    print(dist.pairwise(clf.centroids_))

def compute_inner_class_distance(labels, feats):
    dist = DistanceMetric.get_metric('euclidean')
    distance_matrix = dist.pairwise(feats)
    dist_tran = np.triu(distance_matrix)
    bins = dict(zip(set(labels), [0]*len(set(labels))))
    for i, arr in enumerate(dist_tran):
        for j, val in enumerate(arr):
            if labels[i] == labels[j]:
                bins[labels[i]] += val
    print(bins)


    
def process_latent(gt_path, contignamepath, cutoff = 1000):
    ground_truth = dict()
    gt_bins = set()
    with open(gt_path, 'r') as f:
        for l in f.readlines():
            items = l.split(',')
            if len(items) == 3:
                continue
            temp = items[0].split('_')
            # if int(temp[3]) >= cutoff:
            # ground_truth[items[0]] = items[1]
            ground_truth[temp[1]] = items[1]
            gt_bins.add(items[1])

    temp_list = list(gt_bins)
    temp_list.sort()
    label_dict = dict(zip(temp_list, range(len(gt_bins))))
    contignames = np.load(contignamepath)
    # contignames = contignames['arr_0']

    labels = []
    delete_idx = []
    for idx, contig in enumerate(contignames):
        try:
            labels.append(label_dict[str(ground_truth[str(int(contig.split('_')[1]))])])
            # labels.append(label_dict[str(ground_truth[str(int(contig))])])
        except KeyError:
            labels.append(-1)
            delete_idx.append(idx)

    return np.array(labels), delete_idx


def load_csv(csv_path):
    labels = []
    with open(csv_path) as f:
        lines = f.readlines()
        for line in lines:
            item = line.split()
            if int(item[1].split('_')[3]) >= 3000:
                labels.append(int(item[0]))
            else:
                labels.append(-1)
    return np.array(labels)


if __name__ == "__main__":
    latent_path = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/deepmetabin/gmvae_1000/latent600.npy"
    # labels_path = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/vamb_1000/vamb_out/clusters.tsv"    
    # output_path = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/mingxing/work_with_wc/Metagenomic-Binning/graphs/test.png"

    gt_path = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/preprocess_result/issue_bins.csv"
    contignamepath = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/deepmetabin/gmvae_1000/id.npy"

    latent = np.load(latent_path) # 8045, 32
    # latent = latent['arr_0']
    # labels = load_csv(labels_path)
    labels, delete_idx = process_latent(gt_path, contignamepath)
    # labels, latent = vamb_knn_plot(gt_path, contignamepath, latent)
    # to delete
    latent = np.delete(latent, delete_idx, axis=0)
    labels = np.delete(labels, delete_idx, axis=0)
    compute_inter_class_distance(labels, latent)
    compute_inner_class_distance(labels, latent)




