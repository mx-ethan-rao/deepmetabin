from cProfile import label
import os
import subprocess
import contextlib
import multiprocessing
import tempfile
import sys
import random
import shutil
import multiprocessing
import traceback
from multiprocessing.pool import Pool
from src.utils.bh_kmeans import bh_kmeans
import numpy as np
import glob
from collections import defaultdict
import argparse


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

def get_marker(hmmout, fasta_path=None, min_contig_len=None, multi_mode=False, orf_finder = None, bin_num_mode='median'):
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
    cannot_link = []
    for indices in data.groupby('gene').indices.values():
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                cannot_link.append((data.iloc[indices[i]]['contig'], data.iloc[indices[j]]['contig']))

    def extract_seeds(vs, sel, bin_num_mode):
        vs = vs.sort_values()
        if bin_num_mode == 'median':
            median = vs[len(vs) //2]
        elif bin_num_mode == 'max':
            median = vs[len(vs) - 1]
        # the original version broke ties by picking the shortest query, so we
        # replicate that here:
        candidates = vs.index[vs == median]
        c = qlen.loc[candidates].idxmin()
        r = list(sel.query('gene == @c')['contig'])
        r.sort()
        return len(r)


    if multi_mode:
        data['bin'] = data['orf'].str.split('.', n=0, expand=True)[0]
        counts = data.groupby(['bin', 'gene'])['orf'].count()
        res = {}
        for b,vs in counts.groupby(level=0):
            cs = extract_seeds(vs.droplevel(0), data.query('bin == @b', local_dict={'b':b}))
            res[b] = [c.split('.',1)[1] for c in cs]
        return res
    else:
        counts = data.groupby('gene')['orf'].count()
        return extract_seeds(counts, data, bin_num_mode), list(set(cannot_link))

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

def gen_cannot_link(fasta_path, binned_length, num_process, bin_num_mode, multi_mode=False, output = None, orf_finder = 'prodigal'):
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
                    #  os.path.split(__file__)[0] + '/marker.hmm',
                     './src/utils/marker.hmm',
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

        marker = get_marker(hmm_output, fasta_path, binned_length, multi_mode, orf_finder=orf_finder, bin_num_mode=bin_num_mode)

        return marker

def gen_cannot_link_indices(dataset, cl, ml, contignames, target_contig):
    target_indices = dict()
    cl_indices = list()
    ml_indices = list()
    for contig in target_contig:
        target_indices[contig] = int(np.where(contignames == contig)[0])
    bin_data = dataset[list(target_indices.values())]
    target_contig = contignames[list(target_indices.values())]

    target_indices = dict()
    for contig in target_contig:
        target_indices[str(contig)] = int(np.where(target_contig == contig)[0])

    for ml1, ml2 in ml:
        if ml1 in target_indices.keys() and ml2 in target_indices.keys():
            ml_indices.append((target_indices[ml1], target_indices[ml2]))

    for cl1, cl2 in cl:
        cl_indices.append((target_indices[cl1], target_indices[cl2]))
    return cl_indices, ml_indices, bin_data, target_contig
    

def read_bins(file):
    bins = defaultdict(list)
    with open(file, 'r') as f:
        for l in f.readlines():
            items = l.split()
            bins[int(items[0])].append(items[1])
    return dict(bins)

def read_must_link(file, contignames):
    contig_dict = dict()
    for contigname in contignames:
        contig_dict[contigname.split('_')[1]] = str(contigname)
    
    must_link = []
    with open(file, 'r') as f:
        for l in f.readlines():
            items = l.split()
            must_link.append((contig_dict[items[0]], contig_dict[items[1]]))
    return must_link

def merge_bins(bin_dict, issu_bins, post_bin_list):
    for bin_num in issue_bins:
        bin_dict.pop(bin_num)
    for post_bins in post_bin_list:
        bin_dict.update(post_bins)
    
    merge_bin_dict = dict(zip(range(len(bin_dict)), bin_dict.values()))
    return merge_bin_dict

def get_issue_bins(checkm_path):
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
        if float(record['Completeness']) > 90 and float(record['Contamination']) >= 5:
            issue_bins.append(int(record['Bin Id'].split('.')[1]))
    return issue_bins

def purify_must_link(ml_indices, cl_indices):
    delete_indices = []
    if len(ml_indices) != 0:
        for idx, (ml1, ml2) in enumerate(ml_indices):
            if ((ml1, ml2) in cl_indices) or ((ml2, ml1) in cl_indices):
                delete_indices.append(idx)

        for i in sorted(delete_indices, reverse=True):
            del ml_indices[i]
        print('del_idx', end='')
        print(delete_indices)
    return ml_indices, cl_indices

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split samples before binning')
    # parser.add_argument('--fasta_path', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/gmm_bins', help='Fasta path (.fasta)')
    # parser.add_argument('--orignal_binning_file', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/gmm.csv', help='ignore contig length under this threshold')
    parser.add_argument('--primary_out', type=str, default='./deepmetabin_out', help='Output directory for primary clustering')
    parser.add_argument('--contigname_path', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/contignames.npz', help='Output path for all splitted samples')
    parser.add_argument('--output_path', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/postprocess1', help='Output path for all splitted samples')
    # parser.add_argument('--must_link_path', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/must_link.csv', help='Output path for all splitted samples')
    # parser.add_argument('--latent_path', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/latents/latent_80_21141_best.npy', help='Output path for all splitted samples')
    # parser.add_argument('--checkm_path', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/gmm_bins/checkm.tsv', help='Output path for all splitted samples')
    parser.add_argument('--binned_length', type=int, default=1000, help='ignore contig length under this threshold')
    parser.add_argument('--mode', type=str, default='max', help='Scg bin number mode (max or median)')


    args = parser.parse_args()
    # fasta_path = '/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/deepmetabin/cami-m2-latent/gmm_bins'
    # orignal_binning_file = '/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/deepmetabin/cami-m2-latent/gmm.csv'
    # contigname_path = '/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/deepmetabin/cami-m2-latent/contignames.npy'
    # output_path = '/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/medium/postprocess_001/out'
    # output_path = sys.argv[1]
    # must_link_path = '/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/medium/postprocess_001/must_link.csv'
    # latent_path = '/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/deepmetabin/cami-m2-latent/latent_epoch_195_best.npy'
    # checkm_path = '/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/deepmetabin/cami-m2-latent/gmm_bins/checkm.tsv'
    # binned_length = 1000

    os.makedirs(args.output_path, exist_ok=True)

    fasta_bin = glob.glob(os.path.join(args.primary_out, 'results', 'pre_bins', 'cluster.*.fasta'))
    contignames = np.load(args.contigname_path)['arr_0']
    latent = np.load(os.path.join(args.primary_out, 'results', 'latent.npy'))
    bin_dict = read_bins(os.path.join(args.primary_out, 'results', 'gmm.csv'))
    must_link = read_must_link(os.path.join(args.primary_out, 'must_link.csv'), contignames)
    issue_bins = get_issue_bins(os.path.join(args.primary_out, 'results', 'pre_bins', 'checkm.tsv'))
    # issue_bins = [0, 11, 17, 23, 25, 26, 29, 39, 6, 9, 31,18, 30, 12]
    post_bin_list = []

    for bin_path in fasta_bin:
        cluster_num = int(os.path.basename(bin_path).replace('cluster.','').replace('.fasta', ''))
        if cluster_num in issue_bins:
            n_clusters, cannot_link = gen_cannot_link(bin_path, args.binned_length, 20, bin_num_mode=args.mode, output=args.output_path)
            cl_indices, ml_indices, bin_data, target_contig = gen_cannot_link_indices(latent, cannot_link, must_link, contignames, bin_dict[cluster_num])
            ml_indices, cl_indices = purify_must_link(ml_indices, cl_indices)
            if n_clusters <= 1:
                n_clusters = 2
            labels = bh_kmeans(bin_data, n_clusters, ml=ml_indices, cl=cl_indices, p=3, random_state=2021, time_limit=600)
            post_bin = defaultdict(list)
            for l, contigname in zip(labels, target_contig):
                post_bin[str(cluster_num) + '_' + str(l)].append(str(contigname))
            post_bin_list.append(dict(post_bin))
  

    bin_dict = merge_bins(bin_dict, issue_bins, post_bin_list)
    with open(os.path.join(args.output_path, f'post_cluster.csv'), 'w') as f:
        for key, val in bin_dict.items():
            for v in val:
                f.write(f'{str(key)}\t{str(v)}\n')
    

