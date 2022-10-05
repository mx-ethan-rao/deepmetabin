
if __name__ == "__main__":
    nc_bins = [10,11, 12,15]
    path = '/tmp/local/zmzhang/DeepMetaBin/mingxing/DeepBin/data/CAMI1_L/labels.csv'
    savepath = '/tmp/local/zmzhang/DeepMetaBin/mingxing/work_with_wc/Metagenomic-Binning/temp/4_big_clusters.txt'
    cutoff = 1000

    # ----------------load gt---------------
    ground_truth = dict()
    gt_bins = set()
    with open(path, 'r') as f:
        for l in f.readlines():
            items = l.split(',')
            if len(items) == 3:
                continue
            temp = items[0].split('_')
            if int(temp[3]) >= cutoff:
                ground_truth[items[0]] = items[1]
            # ground_truth[temp[1]] = items[1]
            gt_bins.add(items[1])

    temp_list = list(gt_bins)
    temp_list.sort()
    label_dict = dict(zip(temp_list, range(len(gt_bins))))
    with open(savepath, 'w') as f:
        for key, value in ground_truth.items():
            if label_dict[value] not in [10,11, 12,15]:
                f.write(key + '\n')

    # nc_keep = []
    # with open(path, 'r') as f:
    #     for l in f.readlines():
    #         items = l.split()
    #         temp = items[1].split('_')
    #         if int(temp[3]) >= cutoff and items[0].strip() in nc_bins:
    #             nc_keep.append(items[1].strip() + ',' + items[0].strip() + '\n')
    # with open(savepath, 'w') as f:
    #     for line in nc_keep:
    #         f.write(line)