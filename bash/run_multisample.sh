#!/bin/bash

# $1 is the specific sample to run (e.g. S1)
# Specify a sample to run by sh run_multisample.sh S1:S3:S6

set -ex

trap clean EXIT SIGTERM
clean(){
    # Unbind the file descriptor and delete the pipe when all is said and done
    exec 4<&-
    exec 4>&-
    rm -f mylist_$fifoname
    kill -9 -$$
}


cd $(dirname $0)
multisample_name=$1
concat_tnf=$3
concat_rkpm=$4
concat_contignames=$5
out=$6
# multisample_name=cami2_Oral
# contig_dir=/datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI2/spades/Oral
# concat_tnf=/datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI2/binning_results/Oral/vamb/vamb_out/tnf.npz
# concat_rkpm=/datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI2/binning_results/Oral/vamb/vamb_out/rpkm.npz
# concat_contignames=/datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI2/binning_results/Oral/vamb/vamb_out/contignames.npz
# concat_label=/datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI2/binning_results/Oral/vamb/labels/labels.csv
# out=/datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI2/binning_results/Oral/deepmetabin

# mkdir -p $(dirname $0)/data/$multisample_name


export sample_paths=`ls -d $out/S*`
fifoname=0

# if [ ! $1 ];then
#     python ./src/utils/split_samples.py \
#         $contigs \
#         --concat_tnf $concat_tnf \
#         --concat_rkpm $concat_rkpm \
#         --concat_contignames $concat_contignames \
#         --out $out
#     export sample_paths=`ls -d $out/S*`
#     fifoname=0
# else
#     sample_paths=$out/${1//:/:$out\/}
#     export sample_paths=${sample_paths//:/ }
#     fifoname=$1
# fi
# export sample_paths=`ls -d $out/S*`
thread_num=2

# Create a FIFO
mkfifo mylist_$fifoname
# Bind a file descriptor 4 to the FIFO 
exec 4<>mylist_$fifoname
# Write data to the pipeline beforehand, as many times as you want to start a child process.
for ((i=0; i < $thread_num; i++)); do
    echo $i >&4
done



for sample_path in $sample_paths; do
    read p_idx <&4
    # The & here opens a child process to execute
    {
        mkdir -p $sample_path/latents
        sample_name=`basename $sample_path`
        # python preprocess.py \
        #     --output_zarr_path $(dirname $0)/data/$multisample_name/$sample_name.zarr \
        #     --contigname_path $sample_path/contignames.npz \
        #     --labels_path $sample_path/labels.csv \
        #     --tnf_feature_path $sample_path/tnf.npz \
        #     --rpkm_feature_path $sample_path/rpkm.npz \
        #     --filter_threshold 1000

        export num_classes=`python ./src/utils/calculate_bin_num.py \
                                --fasta  $sample_path/contigs.fasta \
                                --binned_length 1000 \
                                --output $sample_path`
        tmp=`ls -d $out/S*`
        tmp=(${tmp// / })
        input_dim=`expr 103 + ${#tmp[@]}`
        python run.py experiment=train_deepbin \
            name=$multisample_name-$sample_name \
            model.input_size=$input_dim \
            datamodule.zarr_dataset_path=$sample_path/$data.zarr \
            datamodule.multisample=True \
            model.num_classes=$num_classes \
            datamodule.output=$sample_path/deepmetabin_out \
            model.contignames_path=$sample_path/contignames.npz \
            model.contig_path=$sample_path/contigs.fasta
        
        cd $sample_path/deepmetabin_out/pre_bins && checkm lineage_wf -t 100 -x fasta --tab_table -f checkm.tsv ./ ./
        python secondary_clustering.py --primary_out $sample_path/deepmetabin_out --contigname_path $sample_path/deepmetabin_out/contignames.npz --output_path  $sample_path/deepmetabin_out/results/secondary_bins --binned_length 1000 

        echo $p_idx >&4
    } &
done
# Use the wait command to block the current process until all child processes have finished
wait

