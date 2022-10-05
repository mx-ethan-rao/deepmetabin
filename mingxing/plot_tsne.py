from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import csv

def process_latent(gt_path, contignamepath, cutoff = 1000):
    ground_truth = dict()
    gt_bins = set()
    with open(gt_path, 'r') as f:
        for l in f.readlines():
            items = l.split(',')
            if len(items) == 3:
                continue
            temp = items[0].split('_')
            # if int(temp[3]) >= cutoff:
            # ground_truth[items[0]] = items[1]
            ground_truth[temp[1]] = items[1]
            gt_bins.add(items[1])

    temp_list = list(gt_bins)
    temp_list.sort()
    label_dict = dict(zip(temp_list, range(len(gt_bins))))
    contignames = np.load(contignamepath)
    contignames = contignames['arr_0']

    labels = []
    for contig in contignames:
        try:
            # if label_dict[str(ground_truth[str(int(contig))])] not in [10,11, 12,15]:
            #     labels.append(-1)
            # else:
            #     if label_dict[str(ground_truth[str(int(contig))])] == 10:
            #         labels.append(25)
            #     elif label_dict[str(ground_truth[str(int(contig))])] == 11:
            #         labels.append(20)
            #     elif label_dict[str(ground_truth[str(int(contig))])] == 12:
            #         labels.append(15)
            #     elif label_dict[str(ground_truth[str(int(contig))])] == 15:
            #         labels.append(10)
            # labels.append(label_dict[str(ground_truth[str(int(contig))])])

            labels.append(label_dict[str(ground_truth[str(int(contig.split('_')[1]))])])
            # labels.append(label_dict[str(ground_truth[str(contig)])])
        except KeyError:
            labels.append(-1)

    return np.array(labels)

def vamb_knn_plot(gt_path, contignamepath, latents, cutoff = 1000):
    ground_truth = dict()
    gt_bins = set()
    with open(gt_path, 'r') as f:
        for l in f.readlines():
            items = l.split(',')
            if len(items) == 3:
                continue
            temp = items[0].split('_')
            # if int(temp[3]) >= cutoff:
            # ground_truth[items[0]] = items[1]
            ground_truth[temp[1]] = items[1]
            gt_bins.add(items[1])

    temp_list = list(gt_bins)
    temp_list.sort()
    label_dict = dict(zip(temp_list, range(len(gt_bins))))
    contignames = np.load(contignamepath)
    # contignames = contignames['arr_0']

    labels = []
    latent_with_labels = []
    for contig, latent in zip(contignames, latents):
        try:
            # if label_dict[str(ground_truth[str(int(contig))])] not in [10,11, 12,15]:
            #     labels.append(-1)
            # else:
            #     if label_dict[str(ground_truth[str(int(contig))])] == 10:
            #         labels.append(25)
            #     elif label_dict[str(ground_truth[str(int(contig))])] == 11:
            #         labels.append(20)
            #     elif label_dict[str(ground_truth[str(int(contig))])] == 12:
            #         labels.append(15)
            #     elif label_dict[str(ground_truth[str(int(contig))])] == 15:
            #         labels.append(10)
            # labels.append(label_dict[str(ground_truth[str(int(contig))])])

            labels.append(label_dict[str(ground_truth[str(int(contig.split('_')[1]))])])
            # labels.append(label_dict[str(ground_truth[str(contig)])])
            latent_with_labels.append(latent)
        except KeyError:
            # labels.append(-1)
            pass
    return np.array(labels), np.array(latent_with_labels)


# def load_csv(csv_path):
#     labels = []
#     with open(csv_path) as file:
#         reader = csv.reader(file)
#         for row in reader:
#             labels_id = int(row[0].split("N")[0])
#             labels.append(labels_id)
#     return np.array(labels)

def load_csv(csv_path):
    labels = []
    with open(csv_path) as f:
        lines = f.readlines()
        for line in lines:
            item = line.split()
            if int(item[1].split('_')[3]) >= 3000:
                labels.append(int(item[0]))
            else:
                labels.append(-1)
    return np.array(labels)


if __name__ == "__main__":
    latent_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/gmvae_1000/latent600.npy"
    labels_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/vamb_1000/vamb_out/clusters.tsv"    
    output_path = "/tmp/local/zmzhang/DeepMetaBin/mingxing/work_with_wc/Metagenomic-Binning/graphs/gmvae_knn_labels_metadecoder.png"

    gt_path = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/plot_knn_metadecoder.csv"
    contignamepath = "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/gmvae_1000/id.npy"

    latent = np.load(latent_path) # 8045, 32
    # latent = latent['arr_0']
    # labels = load_csv(labels_path)
    # labels = process_latent(gt_path, contignamepath)
    labels, latent = vamb_knn_plot(gt_path, contignamepath, latent)

    tsne = TSNE(
        n_components=2,
        learning_rate="auto",
    )
    compressed_latent = tsne.fit_transform(latent)
    figure = plt.figure(figsize=(10, 10))
    plt.scatter(
        compressed_latent[:, 0],
        compressed_latent[:, 1],
        c=labels,
        marker=".",
        s=10,
        cmap=plt.cm.get_cmap('jet', 10)
    )
    plt.colorbar()
    figure.savefig(output_path)

