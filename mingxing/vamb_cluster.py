import numpy as np
import vamb
import csv
import argparse



def clustering_vamb(embedding, ids, output_csv_path):
    clustering_result = vamb.cluster.cluster(embedding, destroy=True)
    with open(output_csv_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        cnt = 0
        for i, (_, cluster) in enumerate(clustering_result):
            cnt += 1
            contig_id = ids[i]
            # output_contig_head = "NODE_" + "{}".format(int(contig_id))
            bins = [bin for bin in cluster]
            assert len(bins) != 0
            writer.writerow([contig_id, bins[0]])
    print(cnt)

def cluster(clusterspath, latent, contignames, windowsize, minsuccesses, maxclusters,
            minclustersize, separator, cuda=False):

    it = vamb.cluster.cluster(latent, contignames, destroy=True, windowsize=windowsize,
                              normalized=False, minsuccesses=minsuccesses, cuda=cuda)

    renamed = ((str(i+1), c) for (i, (n,c)) in enumerate(it))

    # Binsplit if given a separator
    if separator is not None:
        renamed = vamb.vambtools.binsplit(renamed, separator)

    with open(clusterspath, 'w') as clustersfile:
        _ = vamb.vambtools.write_clusters(clustersfile, renamed, max_clusters=maxclusters,
                                          min_size=minclustersize, rename=False)
    clusternumber, ncontigs = _



if __name__ == "__main__":
    # ------------------for vamb------------
    # latent_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/vamb_1000/vamb_out/latent.npz"
    # contignames_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/vamb_1000/vamb_out/contignames.npz"
    # output_csv_path = "/tmp/local/zmzhang/DeepMetaBin/mingxing/work_with_wc/vae_2000_result.csv"
    # mask_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/vamb_1000/vamb_out/mask.npz"


    # latent = np.load(latent_path)
    # latent = latent['arr_0']
    # contignames = np.load(contignames_path)
    # contignames = contignames['arr_0']
    # mask = np.load(mask_path)
    # mask = mask['arr_0']

    # contignames = [c for c, m in zip(contignames, mask) if m]

    # --------------for deepmetabin------------
    parser = argparse.ArgumentParser(description='manual to this script')

    parser.add_argument('--output_csv_path', type=str, default='/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/cami1-low-deepmetabin-result/result.csv', 
                        help='Checkm result for deepmetabin')
    parser.add_argument('--latent_path', type=str, default='/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/vamb_1000/vamb_out/latent.npz', 
                        help='Checkm result for deepmetabin')
    parser.add_argument('--contignames_path', type=str, default='/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/vamb_1000/vamb_out/contignames.npz', 
                        help='Checkm result for deepmetabin')
    args = parser.parse_args()
    

    # latent_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/vae/latent_fix_2000.npy"
    # contignames_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/vae/id_2000.npy"
    # output_csv_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/vae/vae_2000_result.csv"
    # mask_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/vamb_2000/vamb_out/mask.npz"


    latent = np.load(args.latent_path)
    contignames = np.load(args.contignames_path)
    # contignames = np.squeeze(contignames)
    contignames = contignames.tolist()
    # contignames = ['NODE_'+str(m) for m in contignames]

    cluster(args.output_csv_path, latent, contignames, 200, 20, None, 1, None, False)
    # clustering_vamb(latent, ids, output_csv_path)
