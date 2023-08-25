import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import os
import argparse
import glob

def calculate_precision(predicts, ground_truth):
    predicts_dict = {}
    for clusterNo, contig in predicts:
        if clusterNo in predicts_dict.keys():
            predicts_dict[clusterNo].append(contig)
        else:
            predicts_dict[clusterNo] = [contig]

    ground_truth_dict = {}
    for contig, label in ground_truth:
        ground_truth_dict[contig] = label
    precision_dict = {}
    for key, value in predicts_dict.items():
        precision_dict[key] = {}
        for contig in value:
            if contig not in ground_truth_dict.keys():
                continue
            if ground_truth_dict[contig] in precision_dict[key].keys():
                precision_dict[key][ground_truth_dict[contig]] += 1
            else:
                precision_dict[key][ground_truth_dict[contig]] = 1
    correct_predicts = 0
    total_predicts = 0
    for label_dict in precision_dict.values():
        if len(label_dict.values()) != 0:
            correct_predicts += max(label_dict.values())
            total_predicts += sum(label_dict.values())
    return correct_predicts / total_predicts

def calculate_recall(predicts, ground_truth):
    predicts_dict = {}
    for clusterNo, contig in predicts:
        predicts_dict[contig] = clusterNo
    ground_truth_dict = {}
    for contig, label in ground_truth:
        if label in ground_truth_dict.keys():
            ground_truth_dict[label].append(contig)
        else:
            ground_truth_dict[label] = [contig]

    recall_dict = {}
    total_recalls = 0
    for key, value in ground_truth_dict.items():
        recall_dict[key] = {}
        for contig in value:
            if contig not in predicts_dict.keys():
                total_recalls += 1
                continue
            if predicts_dict[contig] in recall_dict[key].keys():
                recall_dict[key][predicts_dict[contig]] += 1
            else:
                recall_dict[key][predicts_dict[contig]] = 1
    correct_recalls = 0
    for cluster_dict in recall_dict.values():
        if len(cluster_dict.values()) != 0:
            correct_recalls += max(cluster_dict.values())
            total_recalls += sum(cluster_dict.values())
    return correct_recalls / total_recalls


def calculate_ari(predicts, ground_truth):
    ground_truth_dict = {}
    for contig, label in ground_truth:
        ground_truth_dict[contig] = label
    clusters = []
    labels_true = []
    for clusterNo, contig in predicts:
        if contig not in ground_truth_dict.keys():
            continue
        clusters.append(clusterNo)
        labels_true.append(ground_truth_dict[contig])
    return adjusted_rand_score(clusters, labels_true)

def calculate_accuracy(predicts, ground_truth):
    precision = calculate_precision(predicts, ground_truth)
    recall = calculate_recall(predicts, ground_truth)
    f1_score = 2 * (precision * recall) / (precision + recall)
    ari = calculate_ari(predicts, ground_truth)
    return precision, recall, f1_score, ari

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
        if float(record['Completeness']) > 60 and float(record['Contamination']) < 20:
            issue_bins.append(record['Bin Id'].strip())
    return issue_bins

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split samples before binning')
    parser.add_argument('--gt_path', type=str, default='/datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/mingxing/DeepBin/data/CAMI1_L/labels.csv', help='ignore contig length under this threshold')
    parser.add_argument('--checkm_path', type=str, default='/datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI1/low/metadecoder_2000/checkm.tsv', help='Output path for all splitted samples')
    parser.add_argument('--fasta_path', type=str, default='/datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI1/low/metadecoder_2000', help='Output path for all splitted samples')
    parser.add_argument('--binned_length', type=int, default=1000, help='ignore contig length under this threshold')
    parser.add_argument('--suffix', type=str, default='fasta', help='Scg bin number mode')

    args = parser.parse_args()
    
    nc_bins = get_nc_bins(args.checkm_path)
    print(nc_bins)
    predicts = []
    for fasta_file in glob.glob(f'{args.fasta_path}/*.{args.suffix}'):
        # print(nc_bins)
        # print(os.path.basename(fasta_file)[:-(len(args.suffix) + 1)])
        if os.path.basename(fasta_file)[:-(len(args.suffix)+1)] in nc_bins:
            with open(os.path.join(os.path.dirname(args.fasta_path), fasta_file), 'r') as f:
                for line in f.readlines():
                    if line.startswith('>'):
                        predicts.append((fasta_file, line[1:].strip()))
    
    ground_truth = []
    with open(args.gt_path, 'r') as f:
        for l in f.readlines():
            items = l.split(',')
            if len(items) == 3:
                continue
            temp = items[0].split('_')
            if int(temp[3]) >= 0:
                ground_truth.append((items[0], items[1]))

    print(len(predicts))
    total_length = 0
    for _, contigname in predicts:
        total_length += int(contigname.split('_')[3])
    print(total_length)
    precision, recall, f1_score, ari = calculate_accuracy(predicts, ground_truth)
    print("Valid - Precision: {:.5f} Recall: {:.5f} F1_score: {:.5f} ARI: {:.5f}".format(precision, recall, f1_score, ari))
