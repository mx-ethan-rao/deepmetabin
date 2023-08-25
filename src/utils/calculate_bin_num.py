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

    def extract_seeds(vs, sel):
        vs = vs.sort_values()
        median = vs[len(vs) //2]

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
        return extract_seeds(counts, data)

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

def gen_cannot_link(fasta_path, binned_length, num_process, multi_mode=False, output = None, orf_finder = 'prodigal'):
    '''Estimate number of bins from a FASTA file
    Parameters
    fasta_path: path
    binned_length: int (minimal contig length)
    num_process: int (number of CPUs to use)
    multi_mode: bool, optional (if True, treat input as resulting from concatenating multiple files)
    '''
    with tempfile.TemporaryDirectory() as tdir:
        if output is not None:
            if os.path.exists(os.path.join(output, 'markers.hmmout')):
                return get_marker(os.path.join(output, 'markers.hmmout'), fasta_path, binned_length, multi_mode, orf_finder=orf_finder)
            else:
                os.makedirs(output, exist_ok=True)
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

        marker = get_marker(hmm_output, fasta_path, binned_length, multi_mode, orf_finder=orf_finder)

        return marker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split samples before binning')
    parser.add_argument('--fasta', type=str, default='/datahome/datasets/ericteam/csmxrao/DeepMetaBin/tmp/hlj10x/postprocess/post_bins/cluster.126.fasta')
    parser.add_argument('--binned_length', type=int, default=1000, help='ignore contig length under this threshold')

    parser.add_argument('--output', type=str, default='/tmp', help='Output path for all splitted samples')
    # parser.add_argument('--mode', type=str, default='single', help='For single sample/ multisample (single/multi)')


    args = parser.parse_args()
    print(gen_cannot_link(args.fasta, args.binned_length, 20, output=args.output) * 7 // 8)

