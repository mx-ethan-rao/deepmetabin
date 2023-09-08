#!/bin/bash


set -ex

trap clean EXIT SIGTERM
clean(){
    # Unbind the file descriptor and delete the pipe when all is said and done
    exec 4<&-
    exec 4>&-
    rm -f mylist
    kill -9 -$$
}

source /home/comp/zmzhang/software/anaconda3/bin/activate base
export PATH=/home/comp/zmzhang/code:$PATH

cd $(dirname $0)
root_path=/datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI2/binning_results/Oral/deepmetabin

export sample_paths=`ls -d $root_path/S*`
mkdir -p $root_path/post_bins

thread_num=20

# Create a FIFO
mkfifo mylist
# Bind a file descriptor 4 to the FIFO 
exec 4<>mylist
# Write data to the pipeline beforehand, as many times as you want to start a child process.
for ((i=0; i < $thread_num; i++)); do
    echo $i >&4
done

for sample_path in $sample_paths; do
    read p_idx <&4
    # The & here opens a child process to execute
    {
        export num_classes=`python /datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/mingxing/deepmetabin/src/utils/calculate_bin_num.py \
                                --fasta  $sample_path/contigs.fasta \
                                --binned_length 1000 \
                                --output $sample_path`

        latent_path=`realpath $sample_path/latents/latent_*best.npy`
        python /datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/mingxing/deepmetabin/src/utils/gmm.py \
            --latent_path $latent_path \
            --contignames_path $sample_path/contignames.npz \
            --output_csv_path $sample_path/gmm.csv \
            --num_bins $num_classes

        cd $sample_path
        awk '{ t = $1; $1 = $2; $2 = t; print; }' gmm.csv > temp.tsv && get_binning_results.py contigs.fasta temp.tsv gmm_bins && rm -rf temp.tsv
        cd gmm_bins
        checkm lineage_wf -t 100 -x fasta --tab_table -f checkm.tsv ./ ./
        cd ..
        source /home/comp/zmzhang/software/anaconda3/bin/activate SemiBin
        python /datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/mingxing/deepmetabin/secondary_clustering.py \
            --fasta_path gmm_bins \
            --orignal_binning_file gmm.csv \
            --contig_path contignames.npz \
            --output_path  postprocess \
            --must_link_path must_link.csv \
            --latent_path $latent_path \
            --checkm_path gmm_bins/checkm.tsv \
            --binned_length 1000 \
            --mode max
        
        source /home/comp/zmzhang/software/anaconda3/bin/activate base
        cd postprocess
        awk '{ t = $1; $1 = $2; $2 = t; print; }' post_cluster.csv > temp.tsv && get_binning_results.py ../contigs.fasta temp.tsv post_bins && rm -rf temp.tsv
        cd post_bins
        for file in `ls cluster.*.fasta`; do mv $file $(basename $sample_path)$file; done
        cp *cluster.*.fasta $root_path/post_bins
        echo $p_idx >&4
    } &
done
# Use the wait command to block the current process until all child processes have finished
wait
source /home/comp/zmzhang/software/anaconda3/bin/activate base
cd $root_path/post_bins
checkm lineage_wf -t 100 -x fasta --tab_table -f checkm.tsv ./ ./
cd $(dirname $0)
