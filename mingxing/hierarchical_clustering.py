from operator import mod
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import AgglomerativeClustering
import numpy as np

latentdata = '/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/deepmetabin_best/latent.npy'
contignamefile = '/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/deepmetabin_best/id.npy'
savefile = '/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/deepmetabin_best/hierarchical_52.csv'

data = np.load(latentdata)
model = AgglomerativeClustering(n_clusters=None, distance_threshold=52, linkage="ward")
model = model.fit(data)
labels = model.fit_predict(data)

contignames = np.load(contignamefile)

with open(savefile, 'w') as f:
    for contigname, clusterid in zip(contignames, labels):
        f.write('NODE_' + str(int(contigname)) + ',' + str(clusterid) + '\n')
        # f.write(str(clusterid) + ' ' + str(contigname) + '\n')
