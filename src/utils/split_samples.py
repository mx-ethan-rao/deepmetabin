import numpy as np
import argparse
import os
import os.path as osp
import shutil
from collections import defaultdict

def read_labels(concat_label_path) -> dict:
    label_dict = dict()
    with open(concat_label_path, 'r') as f:
        for line in f.readlines():
            item = line.split(',')
            label_dict[item[0].strip()] = item[1].strip()
    return label_dict

def binning(concat_contignames, splitby) -> dict:
    bins = defaultdict(list) 
    for idx, contigname in enumerate(concat_contignames):
        bins[str(contigname).split(splitby)[0]].append(idx)
    return dict(bins)

def out_to_samples(concat_tnf, concat_rkpm, concat_contignames, bins, outdirs, label_dict, splitby):
    tmp_concat_contignames = np.array([str(contigname).split(splitby)[1] for contigname in concat_contignames])
    for sample_path in outdirs:
        idices = bins[osp.basename(sample_path)]
        np.savez(osp.join(sample_path, 'tnf.npz'), concat_tnf[idices])
        np.savez(osp.join(sample_path, 'rpkm.npz'), concat_rkpm[idices])
        np.savez(osp.join(sample_path, 'contignames.npz'), tmp_concat_contignames[idices])
        with open(osp.join(sample_path, 'labels.csv'), 'w') as f:
            for contigname in concat_contignames[idices]:
                try:
                    f.write('{},{}\n'.format(str(contigname).split(splitby)[1], label_dict[str(contigname)]))
                except KeyError:
                    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split samples before binning')
    parser.add_argument('fasta_paths', help='Paths to input FASTA file(s)', nargs='+')
    parser.add_argument('--concat_tnf', type=str, default='', help='Concatenated tnf file path (.npz)')
    parser.add_argument('--concat_rkpm', type=str, default='', help='Concatenated rkpm file path (.npz)')
    parser.add_argument('--concat_contignames', type=str, default='', help='Concatenated contignames file path (.npz)')
    parser.add_argument('--concat_label', type=str, default='', help='Concatenated label file path (.csv)')
    parser.add_argument('--out', type=str, default='', help='Output path for all splitted samples')
    parser.add_argument('--splitby', type=str, default='C', help='Split the sample and the node id, e.g. S9CNODE_1 (sample: S9, node: 1)')

    args = parser.parse_args()

    outdirs = []
    for (pathno, inpath) in enumerate(args.fasta_paths):
        outdir = osp.join(args.out, 'S' + str(pathno + 1))
        os.makedirs(outdir, exist_ok=True)
        outdirs.append(osp.join(args.out, 'S' + str(pathno + 1)))
        shutil.copyfile(inpath, osp.join(outdir, 'contigs.fasta'))


    concat_tnf = np.load(args.concat_tnf)['arr_0']
    concat_rkpm = np.load(args.concat_rkpm)['arr_0']
    concat_contignames = np.load(args.concat_contignames)['arr_0']
    bins = binning(concat_contignames, args.splitby)
    label_dict = read_labels(args.concat_label)
    out_to_samples(concat_tnf, concat_rkpm, concat_contignames, bins, outdirs, label_dict, args.splitby)




