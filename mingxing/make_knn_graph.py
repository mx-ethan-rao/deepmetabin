from math import tan
from os import sysconf
import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
import _pickle as pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os
from cmath import isnan
from torch.nn import functional as F
from visualize_graph import plot_knn_graph


def cal_similar(data,K, use_gpu=torch.cuda.is_available(), threshold=0.2):
    if use_gpu:
        data = data.cuda()
    N = data.shape[0]
    similar_m = []
    weight_m = []
    for idx in range(N):
        dis = torch.sum(torch.pow(data-data[idx,:],2),dim=1)
        sorted, ind = dis.sort()
        K = torch.sum(sorted[1:K+1] < threshold)
        similar_m.append(ind[1:K+1].view(1,K).cpu())
        weight_m.append(Gaussian_kernal(sorted[1:K+1]))
    # similar_m = torch.cat(similar_m,dim=0)

    return similar_m, weight_m

def Gaussian_kernal(data, sigma=1):
    # d = data
    if data.shape[0] == 1:
        return torch.tensor([1])
    data /= (2*pow(sigma,2))
    data = torch.exp(-data)
    m = nn.Softmax(dim=0)
    weights = m(data)
    # weights =  F.log_softmax(data)
    return weights

def zscore(array, axis=None, inplace=False):
    """Calculates zscore for an array. A cheap copy of scipy.stats.zscore.
    Inputs:
        array: Numpy array to be normalized
        axis: Axis to operate across [None = entrie array]
        inplace: Do not create new array, change input array [False]
    Output:
        If inplace is True: None
        else: New normalized Numpy-array"""

    if axis is not None and axis >= array.ndim:
        raise np.AxisError('array only has {} axes'.format(array.ndim))

    if inplace and not np.issubdtype(array.dtype, np.floating):
        raise TypeError('Cannot convert a non-float array to zscores')

    mean = array.mean(axis=axis)
    std = array.std(axis=axis)

    if axis is None:
        if std == 0:
            std = 1 # prevent divide by zero

    else:
        std[std == 0.0] = 1 # prevent divide by zero
        shape = tuple(dim if ax != axis else 1 for ax, dim in enumerate(array.shape))
        mean.shape, std.shape = shape, shape

    if inplace:
        array -= mean
        array /= std
        return None
    else:
        return (array - mean) / std


tnfs = np.load('/tmp/local/zmzhang/DeepMetaBin/mingxing/DeepBin/data/CAMI1_L/vamb/tnf.npz')
abundances = np.load('/tmp/local/zmzhang/DeepMetaBin/mingxing/DeepBin/data/CAMI1_L/vamb/rpkm.npz')
contigs = np.load('/tmp/local/zmzhang/DeepMetaBin/mingxing/DeepBin/data/CAMI1_L/vamb/contignames.npz')

tnfs = tnfs['arr_0']
abundances = abundances['arr_0']



zscore(tnfs, axis=0, inplace=True)
zscore(abundances, axis=0, inplace=True)
# -----------------------------
input_data = np.concatenate([tnfs, abundances], axis=1)



contigs = contigs['arr_0']

init_data = []

for idx, val in enumerate(contigs):
    temp = val.split('_')
    if int(temp[3]) >= 1000:
        data = torch.tensor(input_data[idx], dtype=torch.float32)
        init_data.append(data)


init_data = torch.stack(init_data, dim=0)
print(f'init data shape: {init_data.shape}')
similar_m, weight_m = cal_similar(init_data, K=5)


# torch.save(train_set, '../data/CAMI1_L/training_set.pkl')
# torch.save(test_set, '../data/CAMI1_L/test_set.pkl')
