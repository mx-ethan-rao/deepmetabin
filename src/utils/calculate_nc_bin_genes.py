import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import os
import argparse
import glob

def get_nc_bins(checkm_path):
    keys = ['Bin Id', 'Marker lineage',	'# genomes', '# markers', '# marker sets',
                '0', '1', '2', '3',	'4', '5+', 'Completeness', 'Contamination',	'Strain heterogeneity']
    # checkm_results = ['vamb_1000.tsv']

    checkm_result = []
    with open(checkm_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            elems = line.split()
            if len(elems) == 15:
                elems.pop(2)
                checkm_result.append(dict(zip(keys, elems)))

    # calculate the num of bins in different level, for precision
    issue_bins = []
    for record in checkm_result:
        if float(record['Completeness']) > 90 and float(record['Contamination']) < 5:
            issue_bins.append(record['Bin Id'].strip())
    return issue_bins

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split samples before binning')
    parser.add_argument('--prokka_path', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/metadecoder_2000/prokka/prokka_result.tsv', help='ignore contig length under this threshold')
    parser.add_argument('--checkm_path', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/metadecoder_2000/checkm.tsv', help='Output path for all splitted samples')
    # parser.add_argument('--binned_length', type=int, default=1000, help='ignore contig length under this threshold')
    parser.add_argument('--suffix', type=str, default='fasta', help='Scg bin number mode')

    args = parser.parse_args()
    nc_bins = get_nc_bins(args.checkm_path)
    total = 0
    with open(args.prokka_path, 'r') as f:
        for line in f.readlines():
            # if line.split()[0][:-(len(args.suffix)+1)] in nc_bins:
            if 'METADECODER.{}'.format(line.split()[0].split('.')[1]) in nc_bins:
                cnt=int(line.split()[1])
                total += cnt
    print(total / 2)