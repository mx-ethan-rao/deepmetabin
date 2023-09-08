#!/usr/bin/env python3

# More imports below, but the user's choice of processors must be parsed before
# numpy can be imported.
import numpy as np
import sys
import os
import argparse
import torch
import datetime
import time
import shutil
import zarr
from tqdm import trange
from absl import app, flags

from src.utils.util import (
    summary_bin_list_from_csv,
    load_graph,
    describe_dataset,
)

from src.utils import (
    parsebam,
    parsecontigs,
    vambtools
)

from src.utils.calculate_bin_num import gen_cannot_link as cal_num_bins




DEFAULT_PROCESSES = min(os.cpu_count(), 8)

# These MUST be set before importing numpy
# I know this is a shitty hack, see https://github.com/numpy/numpy/issues/11826
os.environ["MKL_NUM_THREADS"] = str(DEFAULT_PROCESSES)
os.environ["NUMEXPR_NUM_THREADS"] = str(DEFAULT_PROCESSES)
os.environ["OMP_NUM_THREADS"] = str(DEFAULT_PROCESSES)

# Append vamb to sys.path to allow vamb import even if vamb was not installed
# using pip
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)


################################# DEFINE FUNCTIONS ##########################

def log(string, logfile, indent=0):
    print(('\t' * indent) + string, file=logfile)
    logfile.flush()


def calc_tnf(outdir, fastapath, mincontiglength, logfile):
    begintime = time.time()
    log('\nLoading TNF', logfile, 0)
    log('Minimum sequence length: {}'.format(mincontiglength), logfile, 1)
    log('Loading data from FASTA file {}'.format(fastapath), logfile, 1)
    with vambtools.Reader(fastapath, 'rb') as tnffile:
        ret = parsecontigs.read_contigs(
            tnffile, minlength=mincontiglength)

    tnfs, contignames, contiglengths = ret
    vambtools.write_npz(os.path.join(outdir, 'tnf.npz'), tnfs)
    vambtools.write_npz(os.path.join(
        outdir, 'lengths.npz'), contiglengths)
    vambtools.write_npz(os.path.join(
        outdir, 'contignames.npz'), contignames)

    elapsed = round(time.time() - begintime, 2)
    ncontigs = len(contiglengths)
    nbases = contiglengths.sum()

    print('', file=logfile)
    log('Kept {} bases in {} sequences'.format(nbases, ncontigs), logfile, 1)
    log('Processed TNF in {} seconds'.format(elapsed), logfile, 1)

    return tnfs, contignames, contiglengths


def calc_rpkm(outdir, bampaths, rpkmpath, jgipath, mincontiglength, refhash, ncontigs,
              minalignscore, minid, subprocesses, logfile):
    begintime = time.time()
    log('\nLoading RPKM', logfile)
    # If rpkm is given, we load directly from .npz file
    if rpkmpath is not None:
        log('Loading RPKM from npz array {}'.format(rpkmpath), logfile, 1)
        rpkms = vambtools.read_npz(rpkmpath)

        if not rpkms.dtype == np.float32:
            raise ValueError('RPKMs .npz array must be of float32 dtype')

    else:
        log('Reference hash: {}'.format(
            refhash if refhash is None else refhash.hex()), logfile, 1)

    # Else if JGI is given, we load from that
    if jgipath is not None:
        log('Loading RPKM from JGI file {}'.format(jgipath), logfile, 1)
        with open(jgipath) as file:
            rpkms = vambtools._load_jgi(file, mincontiglength, refhash)

    else:
        log('Parsing {} BAM files with {} subprocesses'.format(len(bampaths), subprocesses),
            logfile, 1)
        log('Min alignment score: {}'.format(minalignscore), logfile, 1)
        log('Min identity: {}'.format(minid), logfile, 1)
        log('Min contig length: {}'.format(mincontiglength), logfile, 1)
        log('\nOrder of columns is:', logfile, 1)
        log('\n\t'.join(bampaths), logfile, 1)
        print('', file=logfile)

        dumpdirectory = os.path.join(outdir, 'tmp')
        rpkms = parsebam.read_bamfiles(bampaths, dumpdirectory=dumpdirectory,
                                            refhash=refhash, minscore=minalignscore,
                                            minlength=mincontiglength, minid=minid,
                                            subprocesses=subprocesses, logfile=logfile)
        print('', file=logfile)
        vambtools.write_npz(os.path.join(outdir, 'rpkm.npz'), rpkms)
        shutil.rmtree(dumpdirectory)

    if len(rpkms) != ncontigs:
        raise ValueError(
            "Length of TNFs and length of RPKM does not match. Verify the inputs")

    elapsed = round(time.time() - begintime, 2)
    log('Processed RPKM in {} seconds'.format(elapsed), logfile, 1)

    return rpkms



def create_contigs_zarr_dataset(
        output_zarr_path: str,
        num_bins: int,
        contigname_attrs: str,
        labels_path: str,
        tnf_attrs: str,
        rpkm_attrs: str,
        # ag_graph_path: str,
        # pe_graph_path: str,
        filter_threshold: int = 1000,
        # long_contig_threshold: int = 1000,
    ):
    """create long contigs zarr dataset based on contigname file and labels file as filters.

    Args:
        output_zarr_path (string): path to save the processed long contigs datasets.
        contigname_path (string): path of contigname file.
        labels_path (string): path of labels file.
        tnf_feature_path (string): path of tnf feature file.
        rpkm_feature_path (string): path of rpkm feature file.
        ag_graph_path (string): path of ag graph file.
        pe_graph_path (string): path of pe graph file.
        filter_threshold (int): threshold of filtering bp length default 1000.
        long_contig_threshold (int): threshold of long contig.

    Returns:
        None.

    Root zarr group stands for the collection of contigs:
        - contig_id_list (attrs -> list): contigs list after filtering.
        - long_contig_id_list (attrs -> list): long contigs list.
    Each zarr group stands for one unique contig, with six properties:
        - id (group -> np.ndarray (1)): contig id.
        - tnf_feat (group -> np.ndarray (103)): tnf feature.
        - rpkm_feat (group -> np.ndarray (1)): rpkm feature.
        - labels (group -> np.ndarray (1)): ground truth bin id.
        - ag_graph_edges (group -> list): relevant edge in ag graph.
        - pe_graph_edges (group -> list): relevant edge in pe graph.
    """
    root = zarr.open(output_zarr_path, mode="w")
    bin_list = summary_bin_list_from_csv(labels_path)
    num_cluster = len(bin_list)


    contig_id_list = []
    tnf_list = []
    rpkm_list = []
    label_list = []
    # long_contig_id_list = []
    for i in trange(len(contigname_attrs), desc="Preprocessing dataset......"):
        props = contigname_attrs[i].split("_")
        contig_id = int(props[1])
        contig_length = int(props[3])
        # if contig_length >= long_contig_threshold:
        #     long_contig_id_list.append(i)
        if contig_length >= filter_threshold:
            have_label = False
            for j in range(num_cluster):
                if contig_id in bin_list[j]:
                    labels = j
                    have_label = True
            if have_label is False:
                labels = -1 # Use -1 to hold unlabeled contig (node)

            contig_id_list.append(contig_id)
            tnf_list.append(list(tnf_attrs[i]))
            rpkm_list.append(list(rpkm_attrs[i]))
            label_list.append(labels)

    root.attrs["contig_id_list"] = contig_id_list
    root.attrs["tnf_list"] = tnf_list
    root.attrs["rpkm_list"] = rpkm_list
    root.attrs["label_list"] = label_list
    root.attrs["num_bins"] = num_bins

def run(outdir, fastapath, bampaths, rpkmpath, jgipath,
        mincontiglength, norefcheck, minalignscore, minid, subprocesses, output_zarr_path, label_path, logfile):

    log('Date and time is ' + str(datetime.datetime.now()), logfile, 1)
    begintime = time.time()

    num_bins = cal_num_bins(fastapath, mincontiglength, 20, output='/tmp') * 7 // 8
    # Get TNFs, save as npz
    tnfs, contignames, contiglengths = calc_tnf(outdir, fastapath, mincontiglength, logfile)

    # Parse BAMs, save as npz
    refhash = None if norefcheck else vambtools._hash_refnames(
        contignames)
    rpkms = calc_rpkm(outdir, bampaths, rpkmpath, jgipath, mincontiglength, refhash,
                      len(tnfs), minalignscore, minid, subprocesses, logfile)

    create_contigs_zarr_dataset(
            output_zarr_path=output_zarr_path,
            num_bins=num_bins,
            contigname_attrs=contignames,
            labels_path=label_path,
            tnf_attrs=tnfs,
            rpkm_attrs=rpkms,
            # ag_graph_path=self.ag_graph_path,
            # pe_graph_path=self.pe_graph_path,
            filter_threshold=mincontiglength,
            # long_contig_threshold=self.long_contig_threshold,
        )
    describe_dataset(processed_zarr_dataset_path=output_zarr_path)

    elapsed = round(time.time() - begintime, 2)


def main():
    doc = """DeepMetaBin: DeepMetaBin for metagenomic binning.

    Default use, good for most datasets:
    python preprocessing --outdir out --fasta my_contigs.fna --bamfiles *.bam """
    parser = argparse.ArgumentParser(
        prog="DeepMetaBin",
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s outdir tnf_input rpkm_input [options]",
        add_help=False)

    # Help
    helpos = parser.add_argument_group(title='Help', description=None)
    helpos.add_argument(
        '-h', '--help', help='print help and exit', action='help')

    # Positional arguments
    reqos = parser.add_argument_group(
        title='Output (required)', description=None)
    reqos.add_argument('--outdir', metavar='', required=True,
                       help='output directory to create')
    reqos.add_argument('--label_path', metavar='', required=True,
                       help='label path')

    # TNF arguments
    tnfos = parser.add_argument_group(
        title='TNF input (either fasta or all .npz files required)')
    tnfos.add_argument('--fasta', metavar='', help='path to fasta file')

    # RPKM arguments
    rpkmos = parser.add_argument_group(
        title='RPKM input (either BAMs, JGI or .npz required)')
    rpkmos.add_argument('--bamfiles', metavar='',
                        help='paths to (multiple) BAM files', nargs='+')
    rpkmos.add_argument('--rpkm', metavar='', help='path to .npz of RPKM')
    rpkmos.add_argument(
        '--jgi', metavar='', help='path to output of jgi_summarize_bam_contig_depths')

    # Optional arguments
    inputos = parser.add_argument_group(title='IO options', description=None)

    inputos.add_argument('-m', dest='minlength', metavar='', type=int, default=1000,
                         help='ignore contigs shorter than this [100]')
    inputos.add_argument('-s', dest='minascore', metavar='', type=int, default=None,
                         help='ignore reads with alignment score below this [None]')
    inputos.add_argument('-z', dest='minid', metavar='', type=float, default=None,
                         help='ignore reads with nucleotide identity below this [None]')
    inputos.add_argument('-p', dest='subprocesses', metavar='', type=int, default=DEFAULT_PROCESSES,
                         help=('number of subprocesses to spawn '
                               '[min(' + str(DEFAULT_PROCESSES) + ', nbamfiles)]'))
    inputos.add_argument('--norefcheck', help='skip reference name hashing check [False]',
                         action='store_true')
    inputos.add_argument('--minfasta', dest='minfasta', metavar='', type=int, default=None,
                         help='minimum bin size to output as fasta [None = no files]')



    ######################### PRINT HELP IF NO ARGUMENTS ###################
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    ###################### SET UP LAST PARAMS ############################

    # This doesn't actually work, but maybe the PyTorch folks will fix it sometime.
    subprocesses = args.subprocesses
    torch.set_num_threads(args.subprocesses)
    if args.bamfiles is not None:
        subprocesses = min(subprocesses, len(args.bamfiles))

    ################### RUN PROGRAM #########################
    try:
        os.mkdir(args.outdir)
    except FileExistsError:
        pass
    except:
        raise
    
    output_zarr_path = os.path.join(args.outdir, 'data.zarr')
    logpath = os.path.join(args.outdir, 'log.txt')

    with open(logpath, 'w') as logfile:
        run(args.outdir,
            args.fasta,
            args.bamfiles,
            args.rpkm,
            args.jgi,
            mincontiglength=args.minlength,
            norefcheck=args.norefcheck,
            minalignscore=args.minascore,
            minid=args.minid,
            subprocesses=subprocesses,
            output_zarr_path=output_zarr_path,
            label_path=args.label_path,
            logfile=logfile)


if __name__ == '__main__':
    main()
