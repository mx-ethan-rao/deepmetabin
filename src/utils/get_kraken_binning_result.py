#!/usr/bin/env python

"""
usage:
get_binning_results.py <contig_file> <binnning_results_csv> <output> <gt>
"""
import os
import sys

gt = sys.argv[4]
gt_contig_header = set()
with open(gt, 'r') as f:
    for line in f.readlines():
        gt_contig_header.add(line.split(',')[0].strip())

contigs = sys.argv[1]
contigs_file = open(contigs, 'r')
contigs_map = {}
header = ""
content = ""
for line in contigs_file:
    if line == "": continue
    if line[0] == '>':
        if header != "": contigs_map[header] = content
        # if len(sys.argv) == 5 and sys.argv[4] == 'idonly':
        #     header = '_'.join(line.split('_')[:2])[1:]
        # else:
        header = '_'.join(line.split('_')[:2])
        header = line.split()[0][1:].strip()
        content = ""
    else: content += line.strip()
contigs_map[header] = content
contigs_file.close()

bin_map = {}
cluster = sys.argv[2]
cluster_file = open(cluster, 'r')
for line in cluster_file:
    if line == "": continue
    if ',' in line:
        items = line.strip().split(',')
    else:
        items = line.strip().split()
    if items[1] not in bin_map: bin_map[items[1]] = []
    bin_map[items[1]].append(items[0])
cluster_file.close()


output = sys.argv[3]
if not os.path.isdir(output):
    os.system(f"mkdir {output}")
for file in os.listdir(output):
    if ".fasta" in file: os.system("rm " + output + '/' + file)
for bin in bin_map:
    out = open(output + "/cluster." + bin + ".fasta", 'w')
    for header in bin_map[bin]:
        if header in gt_contig_header:
            out.write('>' + header + '\n' + contigs_map[header] + '\n')
    out.close()
