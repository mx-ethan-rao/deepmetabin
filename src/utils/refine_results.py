import scipy
import zarr
import torch
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np
from tqdm import tqdm, trange
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset
import os
import argparse

def create_data_list(contignames, other_result_path, latents):
    data_list= []
    label_map = make_label_map(other_result_path)
    contignames = np.load(contignames)['arr_0']
    latents = np.load(latents)
    for contigname, latent in zip(contignames, latents):
        data = {'name': str(contigname), 'feature': latent, 'id': np.array(float(str(contigname).split('_')[1])), 'length': float(str(contigname).split('_')[3])}
        if str(contigname) in label_map.keys():
            data['labels'] = [label_map[str(contigname)]]
            data['already_labeled'] = True
        else:
            data['labels'] = []
            data['already_labeled'] = False
        data_list.append(data)

    return data_list

def make_label_map(other_result_path):
    with open(other_result_path, 'r') as f:
        label_map = dict()
        for line in f.readlines():
            item = line.split(',')
            label_map[item[0].strip()] = int(item[1].strip())
    return label_map


def create_knn_graph(data_list, k=20, threshold=35):
    """Updates the k nearest neighbors for each contig in the dictionary. 
    
    Alerts: knn graph is created by id vector, stores the neightbors 
    and weights for each neighbor.

    Args:
        data_list (list): list format dataset.

    Returns:
        data_list (list): list format dataset.
    """

    id_list = []
    feature_list = []
    for i in range(len(data_list)):
        feature_list.append(data_list[i]["feature"])
        id_list.append(data_list[i]["id"])
    
    feature_array = np.array(feature_list, dtype="float32")
    print('start knn')
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto',  metric='euclidean', n_jobs=50).fit(feature_array)
    distances, indices = nbrs.kneighbors(feature_array)
    for i in trange(feature_array.shape[0], desc="Creating KNN graph..."):
        neighbors_array = indices[i][1:]
        distance_array = np.power(distances[i][1:], 2)
        invalid_idx = np.where(distance_array >= threshold)
        # distance_array = np.delete(distance_array, invalid_idx)
        neighbors_array = np.delete(neighbors_array, invalid_idx)
        # for idx, neight_idx in enumerate(neighbors_array): neighbors_array[idx] = (idx, id_list[neight_idx])
        data_list[i]["neighbors"] = np.float32(neighbors_array)
    for data in data_list:
        for neight_idx in data['neighbors']:
            if not data_list[int(neight_idx)]['already_labeled']:
                data_list[int(neight_idx)]['labels'].extend(data['labels'])

    return data_list

def remove_ambiguous(data_list):
    for i in range(len(data_list)):
        if (not data_list[i]['already_labeled']) and (data_list[i]['length'] >=2500.0 or len(set(data_list[i]['labels'])) > 1):
            data_list[i]['labels'] = [-1]
    return data_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split samples before binning')
    parser.add_argument('--latent', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/latents/latent_80_21141_best.npy')
    parser.add_argument('--other_result', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/kmbgi1/kmbgi/hlj/10x/metadecoder/initial_contig_bins.csv', help='ignore contig length under this threshold')
    parser.add_argument('--deepmetabin_contignames', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/contignames.npz', help='ignore contig length under this threshold')

    parser.add_argument('--output', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x', help='Output path for all splitted samples')

    args = parser.parse_args()
    data_list = create_data_list(args.deepmetabin_contignames, args.other_result, args.latent)
    data_list = create_knn_graph(data_list)
    data_list = remove_ambiguous(data_list)
    with open(os.path.join(args.output, 'refine_30.csv'), 'w') as f:
        for data in data_list:
            if len(data['labels']) != 0 and data['labels'][0] != -1:
                contigname = data['name']
                label = data['labels'][0]
                f.write(f'{contigname},{label}\n')
