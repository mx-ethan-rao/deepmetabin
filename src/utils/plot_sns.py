from sklearn import manifold
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import csv
import random


def read_species_map(path) -> dict:
    species_map = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split(',')
            species_map[temp[0].strip()] = temp[2].strip()
    return species_map

def read_label_with_threshold(gt_path, contignamepath, bin_threshold = 200000, filter_bin = True):
    bins = defaultdict(list)
    with open(gt_path, 'r') as f:
        for l in f.readlines():
            temp = l.split()
            bins[temp[0].strip()].append(int(temp[1].split('_')[3]))
    bins = dict(bins)
    valid_bins = []
    for key in bins.keys():
        if sum(bins[key]) >= bin_threshold:
            valid_bins.append(key)
    labels_dict = {}
    delete_contigs = set()
    with open(gt_path, 'r') as f:
        for l in f.readlines():
            temp = l.split()
            if temp[0].strip() in valid_bins:
                labels_dict[temp[1].strip()] = temp[0]
            else:
                delete_contigs.add(temp[1].strip())
                # labels.append('-1')
    contignames = np.load(contignamepath)
    contignames = contignames['arr_0']
    delete_idx = []
    labels = []
    for idx, contig in enumerate(contignames):
        if contig in delete_contigs:
            delete_idx.append(idx)
            labels.append('-1')
        else:
            labels.append('bin'+labels_dict[contig])
    # latent = np.delete(latent, delete_idx, axis=0)
    return np.array(labels), delete_idx

def postprocessing(contigname, latent):
    delete_idx = []
    for idx, contig in enumerate(contigname):
        if str(contig) == 'Unlabeled':
            delete_idx.append(idx)
    # latent = np.delete(latent, delete_idx, axis=0)
    # contigname = np.delete(contigname, delete_idx, axis=0)
    return delete_idx


    

def process_latent(gt_path, contignamepath, cutoff = 2000):
    ground_truth = dict()
    gt_bins = set()
    with open(gt_path, 'r') as f:
        for l in f.readlines():
            items = l.split(',')
            if len(items) == 3:
                continue
            temp = items[0].split('_')
            if int(temp[3]) >= cutoff:
                ground_truth[items[0]] = items[1].strip()
            # ground_truth[temp[1]] = items[1]
                gt_bins.add(items[1])

    species_map = read_species_map(species_map_path)
    # temp_list = list(gt_bins)
    # temp_list.sort()
    # label_dict = dict(zip(temp_list, range(len(gt_bins))))
    contignames = np.load(contignamepath)
    contignames = contignames['arr_0']

    delete_idx = []
    labels = []
    for idx, contig in enumerate(contignames):
        try:
            # labels.append(label_dict[str(ground_truth[str(int(contig.split('_')[1]))])])
            labels.append(f'{species_map[ground_truth[str(contig)]]}({ground_truth[str(contig)]})')
        except KeyError:
            labels.append('Unlabeled')
            delete_idx.append(idx)

    return np.array(labels), delete_idx

latent_path = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/vamb_2000/vamb_out/latent.npz"
labels_path = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/vamb_2000/vamb_out/clusters.tsv"    
output_path = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/mingxing/work_with_wc/Metagenomic-Binning/graphs/sns_vamb_2000_gt.png"
species_map_path = '/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/cami1_low_species.csv'

gt_path = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/mingxing/DeepBin/data/CAMI1_L/labels.csv"
contignamepath = "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/vamb_2000/vamb_out/contignames.npz"

latent = np.load(latent_path) # 8045, 32
latent = latent['arr_0']

cl_list_small, delete_idx1 = process_latent(gt_path, contignamepath)
_, delete_idx2 = read_label_with_threshold(labels_path, contignamepath)
delete_idx = list(set(delete_idx1).union(delete_idx2))


N = 100000
# embedding_small = embedding[N:2*N,]
# cl_list_small = cl_list[N:2*N]
tsne2 = manifold.TSNE(n_components=2, n_jobs = 250, random_state=2022).fit_transform(latent)

# to delete
tsne2 = np.delete(tsne2, delete_idx, axis=0)
cl_list_small = np.delete(cl_list_small, delete_idx, axis=0)

print("tsne finished")

plt.figure(figsize=(20,20))
random.seed(2022)
sns.scatterplot(x=tsne2[:,0], y=tsne2[:,1], hue=cl_list_small, hue_order=random.shuffle(sorted(pd.unique(cl_list_small).tolist(), key=lambda x: x[::-1])))
plt.savefig(output_path)