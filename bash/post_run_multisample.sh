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

source /home/comp/zmzhang/software/anaconda3/bin/activate base
export PATH=/home/comp/zmzhang/code:$PATH

cd $(dirname $0)
root_path=/datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI2/binning_results/Oral/deepmetabin

export sample_paths=`ls -d $root_path/S*`
mkdir -p $root_path/post_bins

thread_num=20

# 创建一个管道
mkfifo mylist
# 给管道绑定文件描述符4
exec 4<>mylist
# 事先往管道中写入数据，要开启几个子进程就写入多少条数据
for ((i=0; i < $thread_num; i++)); do
    echo $i >&4
done

for sample_path in $sample_paths; do
    read p_idx <&4
    # 这里的 & 会开启一个子进程执行
    {
        export num_classes=`python /datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/mingxing/deepmetabin/src/utils/calculate_bin_num.py \
                                --fasta  $sample_path/contigs.fasta \
                                --binned_length 1000 \
                                --output $sample_path`

        latent_path=`realpath $sample_path/latents/latent_*best.npy`
        python /datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/mingxing/deepmetabin/src/utils/vamb_the_gmm.py \
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
        python /datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/mingxing/deepmetabin/src/utils/post_cluster_processing.py \
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
# 使用 wait 命令阻塞当前进程，直到所有子进程全部执行完
wait
source /home/comp/zmzhang/software/anaconda3/bin/activate base
cd $root_path/post_bins
checkm lineage_wf -t 100 -x fasta --tab_table -f checkm.tsv ./ ./
cd $(dirname $0)
