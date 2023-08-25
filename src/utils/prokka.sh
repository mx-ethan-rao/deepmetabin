#!/bin/bash


set -ex

trap clean EXIT SIGTERM
clean(){
    # 全部结束后解绑文件描述符并删除管道
    exec 4<&-
    exec 4>&-
    rm -f mylist1
    kill -9 -$$
}

export PATH=/home/comp/zmzhang/code:$PATH
cd $(dirname $0)


root_path=/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/metadecoder_2000
bin_path=/datahome/datasets/ericteam/csmxrao/DeepMetaBin/CAMI1/low/metadecoder_2000/prokka/kraken_bins
prokka_path=$root_path/prokka
suffix=fasta
[ ! -d $prokka_path ] && mkdir $prokka_path

bins=`ls $bin_path/*.$suffix`

source /home/comp/zmzhang/software/anaconda3/bin/activate prokka
thread_num=30

# make FIFO
mkfifo mylist1
# bind file descriptor 4 to mylist1
exec 4<>mylist1
# write the thread number to the fifo
for ((i=0; i < $thread_num; i++)); do
    echo $i >&4
done

touch $prokka_path/prokka_result.tsv

for bin in $bins; do
    read p_idx <&4
    {
        bin_name=`basename $bin`
        mkdir $prokka_path/$bin_name
        prokka --kingdom bacteria \
            --outdir $prokka_path/$bin_name \
            --prefix $bin_name \
            --force \
            --cpu 2 \
            $bin \
            --centre X \
            --compliant
        
        num_gene=`wc -l < $prokka_path/$bin_name/$bin_name.tsv`
        num_gene=`expr $num_gene - 1`
        echo -e "$bin_name\t$num_gene" >> $prokka_path/prokka_result.tsv
        echo p_idx>&4
    } &
done
wait

