#!/bin/bash

set -ex

trap clean EXIT SIGTERM
clean(){
    # 全部结束后解绑文件描述符并删除管道
    exec 4<&-
    exec 4>&-
    rm -f mylist
    kill -9 -$$
}

export PATH=$PATH:/home/comp/zmzhang/code

cd $(dirname $0)
dataset_dir=/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/deepmetabin_best
checkm_dir=/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/deepmetabin_best/checkm_result
fasta_file=/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/metaspades/contigs.fasta
[ ! -d $dataset_dir ] && mkdir $dataset_dir
[ ! -d $checkm_dir ] && mkdir $checkm_dir
latent_epochs=`ls $dataset_dir/hierarchical_*.csv`

source ~/software/anaconda3/bin/activate base



thread_num=20


main() {
    # 创建一个管道
    mkfifo mylist
    # 给管道绑定文件描述符4
    exec 4<>mylist
    # 事先往管道中写入数据，要开启几个子进程就写入多少条数据
    for ((i=0; i < $thread_num; i++)); do
        echo $i >mylist
    done



    for latent_epoch in $latent_epochs; do
        read p_idx <mylist
        # 这里的 & 会开启一个子进程执行
        {
            epoch_name=`basename $latent_epoch`
            if [ ! -f $checkm_dir/${epoch_name/.npy/}.tsv ];then
                # python /tmp/local/zmzhang/DeepMetaBin/mingxing/work_with_wc/Metagenomic-Binning/mingxing/vamb_cluster.py \
                #     --output_csv_path ${latent_epoch/.npy/}_result.csv \
                #     --latent_path $latent_epoch \
                #     --contignames_path $dataset_dir/id.npy
                # awk '{ t = $1; $1 = $2; $2 = t; print; }' $latent_epoch > ${latent_epoch/.csv/}_temp.tsv && get_binning_results.py $fasta_file ${latent_epoch/.csv/}_temp.tsv ${latent_epoch/.csv/}_bins && rm -rf ${latent_epoch/.csv/}_temp.tsv
                get_binning_results.py $fasta_file $latent_epoch ${latent_epoch/.csv/}_bins idonly
                cd ${latent_epoch/.csv/}_bins
                checkm lineage_wf -t 100 -x fasta --tab_table -f checkm.tsv ./ ./
                cp checkm.tsv $checkm_dir/${epoch_name/.csv/}.tsv
            fi
            echo $p_idx >mylist
        } &
    done
    # 使用 wait 命令阻塞当前进程，直到所有子进程全部执行完
    wait
    python /tmp/local/zmzhang/DeepMetaBin/mingxing/work_with_wc/Metagenomic-Binning/mingxing/plot_bins.py --checkmPath $checkm_dir
}


main
