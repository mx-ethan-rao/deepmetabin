import argparse
from collections import defaultdict
import glob
import os
import shutil
from cProfile import label
import subprocess
import contextlib
import multiprocessing
import tempfile
import sys
import random
import traceback
from multiprocessing.pool import Pool
import numpy as np

def fasta_iter(fname, full_header=False):
    '''Iterate over a (possibly gzipped) FASTA file
    Parameters
    ----------
    fname : str
        Filename.
            If it ends with .gz, gzip format is assumed
            If .bz2 then bzip2 format is assumed
            if .xz, then lzma format is assumerd
    full_header : boolean (optional)
        If True, yields the full header. Otherwise (the default), only the
        first word
    Yields
    ------
    (h,seq): tuple of (str, str)
    '''
    header = None
    chunks = []
    if fname.endswith('.gz'):
        import gzip
        op = gzip.open
    elif fname.endswith('.bz2'):
        import bz2
        op = bz2.open
    elif fname.endswith('.xz'):
        import lzma
        op = lzma.open
    else:
        op = open
    with op(fname, 'rt') as f:
        for line in f:
            if line[0] == '>':
                if header is not None:
                    yield header,''.join(chunks)
                line = line[1:].strip()
                if not line:
                    header = ''
                elif full_header:
                    header = line.strip()
                else:
                    header = line.split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, ''.join(chunks)

### Return error message when using multiprocessing
def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)
        except Exception as e:
            error(traceback.format_exc())
            raise

        return result

class LoggingPool(Pool):
    def apply_async(self, func, args=(), kwds={}, callback=None):
        return Pool.apply_async(self, LogExceptions(func), args, kwds, callback)

normalize_marker_trans__dict = {
    'TIGR00388': 'TIGR00389',
    'TIGR00471': 'TIGR00472',
    'TIGR00408': 'TIGR00409',
    'TIGR02386': 'TIGR02387',
}

def get_marker(hmmout, fasta_path=None, min_contig_len=None, multi_mode=False, orf_finder = None):
    import pandas as pd
    data = pd.read_table(hmmout, sep=r'\s+',  comment='#', header=None,
                         usecols=(0,3,5,15,16), names=['orf', 'gene', 'qlen', 'qstart', 'qend'])
    if not len(data):
        return []
    data['gene'] = data['gene'].map(lambda m: normalize_marker_trans__dict.get(m , m))
    qlen = data[['gene','qlen']].drop_duplicates().set_index('gene')['qlen']

    def contig_name(ell):
        if orf_finder == 'prodigal':
            contig,_ = ell.rsplit( '_', 1)
        else:
            contig,_,_,_ = ell.rsplit( '_', 3)
        return contig

    data = data.query('(qend - qstart) / qlen > 0.4').copy()
    data['contig'] = data['orf'].map(contig_name)
    if min_contig_len is not None:
        contig_len = {h:len(seq) for h,seq in fasta_iter(fasta_path)}
        data = data[data['contig'].map(lambda c: contig_len[c] >= min_contig_len)]
    data = data.drop_duplicates(['gene', 'contig'])
    return set(data['gene']), dict(zip(data['contig'], data['gene']))
    # def extract_seeds(vs, sel):
    #     vs = vs.sort_values()
    #     median = vs[len(vs) //2]

    #     # the original version broke ties by picking the shortest query, so we
    #     # replicate that here:
    #     candidates = vs.index[vs == median]
    #     c = qlen.loc[candidates].idxmin()
    #     r = list(sel.query('gene == @c')['contig'])
    #     r.sort()
    #     return len(r)


    # if multi_mode:
    #     data['bin'] = data['orf'].str.split('.', n=0, expand=True)[0]
    #     counts = data.groupby(['bin', 'gene'])['orf'].count()
    #     res = {}
    #     for b,vs in counts.groupby(level=0):
    #         cs = extract_seeds(vs.droplevel(0), data.query('bin == @b', local_dict={'b':b}))
    #         res[b] = [c.split('.',1)[1] for c in cs]
    #     return res
    # else:
    #     counts = data.groupby('gene')['orf'].count()
    #     return extract_seeds(counts, data)

def prodigal(contig_file, contig_output):
        with open(contig_output + '.out', 'w') as prodigal_out_log:
            subprocess.check_call(
                ['prodigal',
                 '-i', contig_file,
                 '-p', 'meta',
                 '-q',
                 '-m', # See https://github.com/BigDataBiology/SemiBin/issues/87
                 '-a', contig_output
                 ],
                stdout=prodigal_out_log,
            )

def run_prodigal(fasta_path, num_process, output):

    contigs = {}
    for h, seq in fasta_iter(fasta_path):
        contigs[h] = seq

    total_len = sum(len(s) for s in contigs.values())
    split_len = total_len // num_process

    cur = split_len + 1
    next_ix = 0
    out = None
    with contextlib.ExitStack() as stack:
        for h,seq in contigs.items():
            if cur > split_len and next_ix < num_process:
                if out is not None:
                    out.close()
                out = open(os.path.join(output, 'contig_{}.fa'.format(next_ix)), 'wt')
                out = stack.enter_context(out)

                cur = 0
                next_ix += 1
            out.write(f'>{h}\n{seq}\n')
            cur += len(seq)

    with LoggingPool(num_process) if num_process != 0 else LoggingPool() as pool:
        try:
            for index in range(next_ix):
                pool.apply_async(
                    prodigal,
                    args=(
                        os.path.join(output, 'contig_{}.fa'.format(index)),
                        os.path.join(output, 'contig_{}.faa'.format(index)),
                    ))
            pool.close()
            pool.join()
        except:
            sys.stderr.write(
                f"Error: Running prodigal fail\n")
            sys.exit(1)

    contig_output = os.path.join(output, 'contigs.faa')
    with open(contig_output, 'w') as f:
        for index in range(next_ix):
            f.write(open(os.path.join(output, 'contig_{}.faa'.format(index)), 'r').read())
    return contig_output

def run_fraggenescan(fasta_path, num_process, output):
    try:
        contig_output = os.path.join(output, 'contigs.faa')
        with open(contig_output + '.out', 'w') as frag_out_log:
            # We need to call FragGeneScan instead of the Perl wrapper because the
            # Perl wrapper does not handle filepaths correctly if they contain spaces
            # This binary does not handle return codes correctly, though, so we
            # cannot use `check_call`:
            subprocess.call(
                [shutil.which('FragGeneScan'),
                 '-s', fasta_path,
                 '-o', contig_output,
                 '-w', str(0),
                 '-t', 'complete',
                 '-p', str(num_process),
                 ],
                stdout=frag_out_log,
            )
    except:
        sys.stderr.write(
            f"Error: Running fraggenescan failed\n")
        sys.exit(1)
    return contig_output + '.faa'

def gen_single_copy_gene(fasta_path, binned_length, num_process, multi_mode=False, output = None, orf_finder = 'prodigal'):
    '''Estimate number of bins from a FASTA file
    Parameters
    fasta_path: path
    binned_length: int (minimal contig length)
    num_process: int (number of CPUs to use)
    multi_mode: bool, optional (if True, treat input as resulting from concatenating multiple files)
    '''
    with tempfile.TemporaryDirectory() as tdir:
        if output is not None:
            # if os.path.exists(os.path.join(output, 'markers.hmmout')):
            #     return get_marker(os.path.join(output, 'markers.hmmout'), fasta_path, binned_length, multi_mode, orf_finder=orf_finder)
            # else:
            #     os.makedirs(output, exist_ok=True)
            #     target_dir = output
            target_dir = output
        else:
            target_dir = tdir

        run_orffinder = run_prodigal if orf_finder == 'prodigal' else run_fraggenescan
        contig_output = run_orffinder(fasta_path, num_process, tdir)

        hmm_output = os.path.join(target_dir, 'markers.hmmout')
        try:
            with open(os.path.join(tdir, 'markers.hmmout.out'), 'w') as hmm_out_log:
                subprocess.check_call(
                    ['hmmsearch',
                     '--domtblout',
                     hmm_output,
                     '--cut_tc',
                     '--cpu', str(num_process),
                     os.path.split(__file__)[0] + '/marker.hmm',
                     contig_output,
                     ],
                    stdout=hmm_out_log,
                )
        except:
            if os.path.exists(hmm_output):
                os.remove(hmm_output)
            sys.stderr.write(
                f"Error: Running hmmsearch fail\n")
            sys.exit(1)

        return get_marker(hmm_output, fasta_path, binned_length, multi_mode, orf_finder=orf_finder)



def read_deepmetabin_bins(file, valid_bin_list):
    all_bins = defaultdict(set)
    filter_bins = defaultdict(set)
    with open(file, 'r') as f:
        for l in f.readlines():
            items = l.split()
            if int(items[0]) in valid_bin_list:
                filter_bins[int(items[0])].add(items[1])
            all_bins[int(items[0])].add(items[1])
         
    return dict(all_bins), dict(filter_bins)

def read_metadecoder_bins(file, valid_bin_list):
    all_bins = defaultdict(set)
    filter_bins = defaultdict(set)
    with open(file, 'r') as f:
        for l in f.readlines():
            items = l.split(',')
            if int(items[1]) in valid_bin_list:
                filter_bins[int(items[1])].add(items[0])
            all_bins[int(items[1])].add(items[0])
         
    return dict(all_bins), dict(filter_bins)

def cal_jacard_similarity(set1:set, set2:set) -> float:
    nominater = len(set1.intersection(set2))
    denominator = len(set1) + len(set2) - len(set1.intersection(set2))
    return nominater / denominator

def get_jacard(other_nc_bins:dict, low_contamination_bins:dict) -> list:
    result = dict()
    similarity_matrix = dict()
    for i in other_nc_bins.keys():
        similarities = []
        for j in low_contamination_bins.keys():
            similarities.append((j, cal_jacard_similarity(other_nc_bins[i], low_contamination_bins[j])))
        similarities.sort(key=lambda x:x[1], reverse=True)
        result[i] = similarities[0][0]
        similarity_matrix[i] = similarities

    # while True:
    #     if len(set(result.values())) == len(other_nc_bins):
    #         break
    #     selected = list()
    #     for key in result.keys():
    #         oc_set = set()
    #         res = []
    #         for k, val in result.items():
    #             if val == result[key]:
    #                 res.append(k)

    #         selected.append(result[key])



           
    return result



def get_issue_bins(checkm_path, find_ncbins=True):
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
    issue_bin_cluster_name = []
    for record in checkm_result:
        if find_ncbins:
            # if float(record['Completeness']) > 90 and float(record['Contamination']) < 5:
            if float(record['Completeness']) >= 0:
                issue_bins.append(int(record['Bin Id'].split('.')[1]))
                issue_bin_cluster_name.append(record['Bin Id'])
        else:
            if float(record['Contamination']) < 5:
                issue_bins.append(int(record['Bin Id'].split('.')[1]))
                issue_bin_cluster_name.append(record['Bin Id'])
    return issue_bins, issue_bin_cluster_name

def merge_bins(bin_dict, issue_bins, post_bin_list):
    for bin_num in issue_bins:
        bin_dict.pop(bin_num)
    # for post_bins in post_bin_list:
    bin_dict.update(post_bin_list)
    
    merge_bin_dict = dict(zip(range(len(bin_dict)), bin_dict.values()))
    return merge_bin_dict

def run(fasta1_path, fasta2_path, contignames1, contignames2):
    random_str = ''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',10))
    os.makedirs(f'/tmp/deepmetabin/{random_str}')
    scgs, _ = gen_single_copy_gene(fasta1_path, 1000, 20, output=f'/tmp/deepmetabin/{random_str}')
    try:
        _, contig_gene_map = gen_single_copy_gene(fasta2_path, 1000, 20, output=f'/tmp/deepmetabin/{random_str}')
    except ValueError:
        contig_gene_map = {}
        pass

    for contigname in contignames2:
        if (contigname not in contignames1) and ((contigname not in contig_gene_map.keys()) or (contig_gene_map[contigname] not in scgs)):
            contignames1.add(contigname)
    return list(contignames1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--deepmetabin_results', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/postprocess/post_bins/initial_contig_bins.csv', 
                    help='binning result for deepmetabin')
    parser.add_argument('--deepmetabin_checkm', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/postprocess/post_bins/post_bins/checkm.tsv', 
                    help='Checkm result for vamb') 
    parser.add_argument('--other_result', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/kmbgi1/kmbgi/hlj/10x/metadecoder/initial_contig_bins.csv', 
                    help='binning result for other binning tool')
    parser.add_argument('--other_checkm', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/kmbgi1/kmbgi/hlj/10x/metadecoder/post_bins/checkm.tsv', 
                    help='Checkm result for vamb')
    parser.add_argument('--fasta_path1', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/kmbgi1/kmbgi/hlj/10x/metadecoder/post_bins/', 
                    help='Fasta path (.fasta) for other binning tools')
    parser.add_argument('--fasta_path2', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/postprocess/post_bins/post_bins', 
                    help='Fasta path (.fasta) for deepmetabin')
    parser.add_argument('--output_path', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x', 
                    help='Output path for all splitted samples')
    parser.add_argument('--suffix', type=str, default='fasta', help='Scg bin number mode')            
    parser.add_argument('--n_jobs', type=int, default=10, help='Number of bins being processed simoutaneously')            

    args = parser.parse_args()
    other_nc_bins_indices, issue_bin_cluster_name = get_issue_bins(args.other_checkm, find_ncbins=True)
    other_all_bins, other_nc_bins = read_metadecoder_bins(args.other_result, other_nc_bins_indices)
    _, low_contamination_bins = read_metadecoder_bins(args.deepmetabin_results, get_issue_bins(args.deepmetabin_checkm, find_ncbins=False)[0])
    min_jacard_indices = get_jacard(other_nc_bins, low_contamination_bins)
    # fasta_bin2 = glob.glob(f'{args.fasta_path}/*.{args.suffix}')
    fasta1_paths = []
    fasta2_paths = [] 
    contignames1s = [] 
    contignames2s = []
    refine_bins_list = dict()
    for i, nc_bin_index in enumerate(other_nc_bins_indices):
        fasta1_path = f'{args.fasta_path1}/cluster.{nc_bin_index}.fasta'
        deepmetabin_bin_idx = min_jacard_indices[nc_bin_index]
        fasta2_path = f'{args.fasta_path2}/cluster.{deepmetabin_bin_idx}.fasta'
        contignames1 = other_nc_bins[nc_bin_index]
        contignames2 = low_contamination_bins[deepmetabin_bin_idx]
        refine_bins_list[f'refine_{nc_bin_index}'] = run(fasta1_path, fasta2_path, contignames1, contignames2)

        

    # p = multiprocessing.Pool(args.n_jobs)
    # for fasta1_path, fasta2_path, contignames1, contignames2 in zip(fasta1_paths, fasta2_paths, contignames1s, contignames2s):
    #     refine_bins_list.append(run(fasta1_path, fasta2_path, contignames1, contignames2))
    # refine_bins_list.append(run, zip(fasta1_paths, fasta2_paths, contignames1s, contignames2s))
    other_all_bins = merge_bins(other_all_bins, other_nc_bins_indices, refine_bins_list)

    with open(os.path.join(args.output_path, f'refine_cluster.csv'), 'w') as f:
        for key, val in other_all_bins.items():
            for v in val:
                f.write(f'{str(key)}\t{str(v)}\n')

    shutil.rmtree('/tmp/deepmetabin')

