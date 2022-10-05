
import argparse
from cProfile import label
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
import numpy as np
import os.path as osp

def process_latent(gt_path, latentpath, cutoff = 1000):
    ground_truth = dict()
    with open(gt_path, 'r') as f:
        for l in f.readlines():
            items = l.split(',')
            if len(items) == 3:
                continue
            temp = items[0].split('_')
            if int(temp[3]) >= cutoff:
                ground_truth[items[0]] = items[1]

    contignames = np.load(osp.join(osp.dirname(latentpath),'id.npy'))
    latents = np.load(osp.join(latentpath))

    labels = []
    for contig in contignames:
        try:
            labels.append(str(ground_truth[contig]))
        except KeyError:
            labels.append('-1')

    # for mu, contig in zip(mus, contigs):
    #     latents.append(mu.cpu().detach().numpy() )
    #     contignames.append(contig)

    latents = np.asarray(latents, dtype='float32')
    # latents = latents.copy()
    return latents, labels

def log_tsne_figure(labels, latent, log_path='/tmp/local/zmzhang/DeepMetaBin/mingxing/work_with_wc/Metagenomic-Binning/graphs'):
    """Log the tsne figure on latent space z using sklearn tsne function.
    
    Args:
        batch (dictionary): batch from datamodule to get the node labels.
        latent (tensor): z tensor output from the encoder.
        log_path (string): path to store the plotting tsne figure.
    
    Returns:
        result_tsne_figure_path (string): path of plotting tsne figure.
    """
    relative_path = "/gmvae_600_gt_{}-{}.png".format("tsne", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    result_tsne_figure_path = log_path + relative_path
    # tsne = TSNE(n_components=2, learning_rate='auto')
    # compressed_latent = tsne.fit_transform(latent)
    compressed_latent = TSNE(n_components=2, learning_rate='auto').fit_transform(latent)
    figure = plt.figure(figsize=(10, 10))
    plt.scatter(
        compressed_latent[:, 0],
        compressed_latent[:, 1],
        c=labels,
        marker='.',
        s=10,
    )
    figure.savefig(result_tsne_figure_path)
    return result_tsne_figure_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--checkmPath', type=str, default=None, help='Checkm result path')
    parser.add_argument('--gtPath', type=str, default='/tmp/local/zmzhang/DeepMetaBin/mingxing/DeepBin/data/CAMI1_L/labels.csv', help='Ground truth path')
    parser.add_argument('--latentPath', type=str, default=None, help='latent path')
    
    args = parser.parse_args()
    if args.checkmPath is not None:
        keys = ['Bin Id', 'Marker lineage',	'# genomes', '# markers', '# marker sets',
                '0', '1', '2', '3',	'4', '5+', 'Completeness', 'Contamination',	'Strain heterogeneity']

        checkm_results = os.listdir(args.checkmPath)
        # checkm_results = ['vamb_1000.tsv']

        binning_tools = dict()
        for path in checkm_results:
            checkm_result = []
            with open(os.path.join(args.checkmPath, path)) as f:
                lines = f.readlines()
                for line in lines:
                    elems = line.split()
                    if len(elems) == 15:
                        elems.pop(2)
                        checkm_result.append(dict(zip(keys, elems)))
            binning_tools[path[:-4]] = checkm_result

        # calculate the num of bins in different level, for precision
        nc_bin_result = dict()
        for binning_tool in binning_tools.keys():
            checkm_result = binning_tools[binning_tool]
            num_nc_bins = 0
            for record in checkm_result:
                if float(record['Completeness']) > 90 and float(record['Contamination']) < 5:
                    num_nc_bins += 1
                    # print(record['Bin Id'])
            nc_bin_result[binning_tool] = num_nc_bins
        
        print(nc_bin_result)
    elif args.latentPath is not None:
        latents, labels = process_latent(args.gtPath,args.latentPath)
        log_tsne_figure(labels, latents)

        



