
label_origin = '/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/plot.csv'
label_other = '/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/metadecoder/initial_contig_bins.csv'
save_path = '/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/plot_knn_metadecoder.csv'
proces_num = 21

origin_dict = {}
with open(label_origin, 'r') as f:
    for l in f.readlines():
        items = l.split(',')
        origin_dict[items[0].strip()] = items[1].strip()

other_dict = {}
with open(label_other, 'r') as f:
    for l in f.readlines():
        items = l.split(',')
        other_dict[items[0].strip()] = items[1].strip()

with open(save_path, 'w') as f:
    for key in origin_dict.keys():
        if key in other_dict.keys():
            f.write(key + ',' + other_dict[key] + '\n')
        else:
            f.write(key + ',-1' + '\n')




