#!/bin/bash

# $1 is the specific sample to run (e.g. S1)
# Specify a sample to run by sh run_multisample.sh S1:S3:S6

set -ex

trap clean EXIT SIGTERM
clean(){
    # 全部结束后解绑文件描述符并删除管道
    exec 4<&-
    exec 4>&-
    rm -f mylist_$fifoname
    kill -9 -$$
}


cd $(dirname $0)
multisample_name=cami2_airways
contig_dir=/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI2/spades/Airways
concat_tnf=/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI2/binning_results/vamb/vamb_out/tnf.npz
concat_rkpm=/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI2/binning_results/vamb/vamb_out/rpkm.npz
concat_contignames=/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI2/binning_results/vamb/vamb_out/contignames.npz
concat_label=/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI2/binning_results/vamb/labels/labels.csv
out=/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI2/binning_results/deepmetabin

mkdir -p $(dirname $0)/data/$multisample_name
contigs=`ls $contig_dir/*/contigs.fasta`

source /home/comp/zmzhang/software/anaconda3/bin/activate base

if [ ! $1 ];then
    python /datahome/datasets/ericteam/csmxrao/DeepMetaBin/mingxing/fixed_codes/split_samples.py \
        $contigs \
        --concat_tnf $concat_tnf \
        --concat_rkpm $concat_rkpm \
        --concat_contignames $concat_contignames \
        --concat_label $concat_label \
        --out $out \
    export sample_paths=`ls -d $out/S*`
    fifoname=0
else
    sample_paths=$out/${1//:/:$out\/}
    export sample_paths=${sample_paths//:/ }
    fifoname=$1
fi
# export sample_paths=`ls -d $out/S*`
thread_num=20

# 创建一个管道
mkfifo mylist_$fifoname
# 给管道绑定文件描述符4
exec 4<>mylist_$fifoname
# 事先往管道中写入数据，要开启几个子进程就写入多少条数据
for ((i=0; i < $thread_num; i++)); do
    echo $i >&4
done



for sample_path in $sample_paths; do
    read p_idx <&4
    sleep 1
    # 这里的 & 会开启一个子进程执行
    {
        mkdir -p $sample_path/latents
        sample_name=`basename $sample_path`
        python preprocess.py \
            --output_zarr_path $(dirname $0)/data/$multisample_name/$sample_name.zarr \
            --contigname_path $sample_path/contignames.npz \
            --labels_path $sample_path/labels.csv \
            --tnf_feature_path $sample_path/tnf.npz \
            --rpkm_feature_path $sample_path/rpkm.npz \
            --filter_threshold 1000

        export num_classes=`python /datahome/datasets/ericteam/csmxrao/DeepMetaBin/mingxing/fixed_codes/calculate_bin_num.py \
                                --fasta  $sample_path/contigs.fasta \
                                --binned_length 1000 \
                                --output $sample_path`
        tmp=`ls -d $out/S*`
        tmp=(${tmp// / })
        input_dim=`expr 103 + ${#tmp[@]}`
        python run.py experiment=train_deepbin \
            name=$multisample_name-$sample_name \
            model.gaussian_size=32 \
            model.k=3 \
            model.log_path=$(pwd)/logs \
            model.latent_save_path=$sample_path/latents \
            model.input_size=$input_dim \
            datamodule.k=3 \
            datamodule.must_link_path=$sample_path/must_link.csv \
            datamodule.zarr_dataset_path=$(pwd)/data/$multisample_name/$sample_name.zarr \
            datamodule.multisample=True \
            trainer.check_val_every_n_epoch=3 \
            model.num_classes=$num_classes


        echo $p_idx >&4
    } &
done
# 使用 wait 命令阻塞当前进程，直到所有子进程全部执行完
wait

