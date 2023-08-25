
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--checkmPath', type=str, default='/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/gmvae_1000/checkm_result/latent600.tsv', help='Checkm result path')
    parser.add_argument('--clusterResult', type=str, default='/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/gmvae_1000/latent600_result.csv', help='Checkm result path')
    parser.add_argument('--ncResultSave', type=str, default='/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/gmvae_1000/latent600_nc_result.csv', help='Checkm result path')
    args = parser.parse_args()

    keys = ['Bin Id', 'Marker lineage',	'# genomes', '# markers', '# marker sets',
        	'0', '1', '2', '3',	'4', '5+', 'Completeness', 'Contamination',	'Strain heterogeneity']



    # binning_tools = dict()
    # for path in checkm_results:
    checkm_result = []
    with open(args.checkmPath) as f:
        lines = f.readlines()
        for line in lines:
            elems = line.split()
            if len(elems) == 15:
                elems.pop(2)
                checkm_result.append(dict(zip(keys, elems)))

    nc_bins = []
    for record in checkm_result:
        if float(record['Completeness']) > 90 and float(record['Contamination']) < 5:
            nc_bins.append(record['Bin Id'].split('.')[1])
        #         num_nc_bins += 1
        # nc_bin_result[binning_tool] = num_nc_bins
    
    nc_contigs = []
    with open(args.clusterResult) as f:
        lines = f.readlines()
        for line in lines:
            if line.split()[0].strip() in nc_bins:
                nc_contigs.append(line)
    
    with open(args.ncResultSave, 'w') as f:
        for line in nc_contigs:
            f.write(line)
    

        



