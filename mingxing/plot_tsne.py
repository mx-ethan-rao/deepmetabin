from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import csv


def draw_figure(latent, labels):
    pass


def load_csv(csv_path):
    labels = []
    with open(csv_path) as file:
        reader = csv.reader(file)
        for row in reader:
            labels_id = int(row[0].split("N")[0])
            labels.append(labels_id)
    return np.array(labels)


if __name__ == "__main__":
    latent_path = "/home/eddie/mingxing_temp/latent600.npy"
    labels_path = "/home/eddie/mingxing_temp/gmm_epoch_600.csv"

    output_path = "/home/eddie/mingxing_temp/gmm_600.png"

    latent = np.load(latent_path) # 8045, 32
    labels = load_csv(labels_path)
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
    )
    figure.savefig(output_path)

