# Two tasks: 
# 1. visualize the ground truth contig labels on knn graph.
# 2. visualize the metadecoder nc contig on knn graph.

# To Do Steps:
# - [ ]  load all contigs id, features, labels from zarr dataset into np array.
# - [ ]  construct neighbors id array based on tnf and abundance neighbor features.
# - [ ]  create knn graph based on the contig id list.
# - [ ]  use bin list paired with color dict to visualize the knn graph.

import igraph
from torch import float64
import zarr
import csv
import numpy as np
from tqdm import trange, tqdm
from absl import app, flags
from sklearn.cluster import DBSCAN
from scipy import sparse
from src.utils.plot import update_graph_labels
from src.utils.util import summary_bin_list_from_csv
from mingxing.test_label_propagation import lbp


def write_csv(
    list1,
    list2,
    output_csv_path="/home/eddie/cami1-low-log/output.csv",
):
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for index in list1:
            writer.writerow(["NODE_{}".format(index), 1])
        for index in list2:
            writer.writerow(["NODE_{}".format(index), 2])


def write_contig_csv(
    bin_list,
    plot_contig_list,
    output_csv_path="/home/eddie/cami1-low-log/plot.csv",    
):
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        bin_num = len(bin_list)
        for i in range(bin_num):
            for contig_id in bin_list[i]:
                if contig_id in plot_contig_list:
                    writer.writerow(["NODE_{}".format(contig_id), i])


def create_matrix(data_list, contig_list, option="normal"):
    node_num = len(data_list)
    if option == "normal":
        pre_compute_matrix = np.full((node_num, node_num), 1000.0, dtype=float)
        for i in range(node_num):
            neighbors_array = data_list[i]["neighbors"]
            distances_array = data_list[i]["distances"]
            neighbors_num = neighbors_array.shape[0]
            for j in range(neighbors_num):
                neighbors_id = int(neighbors_array[j])
                neighbors_index = get_index_by_id_from_list(
                    neighbors_id,
                    contig_list,
                )
                pre_compute_matrix[i][neighbors_index] = distances_array[j]
                pre_compute_matrix[neighbors_index][i] = distances_array[j]
            pre_compute_matrix[i][i] = 0.0
    elif option == "sparse":
        pre_compute_matrix = np.zeros((node_num, node_num), dtype=float)
        for i in range(node_num):
            neighbors_array = data_list[i]["neighbors"]
            distances_array = data_list[i]["distances"]
            neighbors_num = neighbors_array.shape[0]
            for j in range(neighbors_num):
                neighbors_id = int(neighbors_array[j])
                neighbors_index = get_index_by_id_from_list(
                    neighbors_id,
                    contig_list
                )
                pre_compute_matrix[i][neighbors_index] = distances_array[j]
                pre_compute_matrix[neighbors_index][i] = distances_array[j]
        pre_compute_matrix = sparse.csr_matrix(pre_compute_matrix)
    return pre_compute_matrix


def summary_bin_list_from_array(
    data_list,
    labels_array,
):
    contig_num = labels_array.shape[0]
    labels_array = labels_array.tolist()
    cluster_list = []
    for i in range(contig_num):
        if labels_array[i] in cluster_list:
            pass
        else:
            cluster_list.append(labels_array[i])
    
    num_cluster = len(cluster_list)
    bin_list = [[] for i in range(num_cluster)]
    for i in range(contig_num):
        cluster_id = labels_array[i]
        contig_id = data_list[i]["id"]
        for index, cluster in enumerate(cluster_list):
            if cluster_id == cluster:
                bin_list[index].append(contig_id)
    print("test for result of bin list number: {}".format(len(bin_list)))
    return bin_list


def plot_dbscan_graph(
    root_path,
    csv_path,
    k,
    graph_type,
    output_path="/datahome/datasets/ericteam/csmxrao/DeepMetaBin/mingxing/Metagenomic-Binning/graphs",
    threshold=0.2,
    compute_method="top_k",
):
    data_list, contig_id_list = load_dataset(root_path)
    dbscan = DBSCAN(
        eps=1.65,
        metric="precomputed",
        min_samples=2,
    )
    """
    dbscan = DBSCAN(
        eps=1.5,
    )
    """
    data_list = create_knn_graph(
        data_list,
        k,
        threshold,
        compute_method,
    )
    
    feature_list = []
    for i in range(len(data_list)):
        feature_list.append(data_list[i]["feature"])
    feature_array = np.array(feature_list)

    plotting_graph_size = 1000
    plotting_contig_list = contig_id_list[:plotting_graph_size]
    pre_compute_matrix = create_matrix(
        data_list=data_list,
        contig_list=contig_id_list,
    )
    cluster_result = dbscan.fit(pre_compute_matrix)
    #cluster_result = dbscan.fit(feature_array)
    labels_array = cluster_result.labels_
    labels_array = label_propagation(labels_array, pre_compute_matrix)

    data_list, labels_array = remove_ambiguous_label(data_list, labels_array, contig_id_list)
    bin_list = summary_bin_list_from_array(
        data_list=data_list,
        labels_array=labels_array,
    )
    
    knn_graph = construct_knn_graph(
        data_list=data_list,
        plotting_graph_size=plotting_graph_size,
        plotting_contig_list=plotting_contig_list,
        k=k,
    )

    plot_knn_graph(
        graph=knn_graph,
        log_path=output_path,
        graph_type=graph_type,
        plotting_contig_list=plotting_contig_list,
        bin_list=bin_list,
    )


def plot_graph(
    root_path,
    csv_path,
    k,
    graph_type,
    output_path="/tmp/local/zmzhang/DeepMetaBin/mingxing/work_with_wc/Metagenomic-Binning/graphs",
    threshold=0.2,
    compute_method="top_k",
):
    data_list, contig_id_list = load_dataset(root_path)
    
    data_list = create_knn_graph(
        data_list,
        k,
        threshold,
        compute_method,
    )

    plotting_graph_size = 1000
    plotting_contig_list = contig_id_list[:plotting_graph_size]
    bin_list = summary_bin_list_from_csv(csv_path)
    # write_contig_csv(
    #     bin_list=bin_list,
    #     plot_contig_list=plotting_contig_list,
    # )
    # write_csv(
    #     list1=bin_list[23],
    #     list2=bin_list[28],
    # )
    knn_graph = construct_knn_graph(
        data_list=data_list,
        plotting_graph_size=plotting_graph_size,
        plotting_contig_list=plotting_contig_list,
        k=k,
    )

    plot_knn_graph(
        graph=knn_graph,
        log_path=output_path,
        graph_type=graph_type,
        plotting_contig_list=plotting_contig_list,
        bin_list=bin_list,
    )


def get_index_by_id_from_list(node_id, node_id_list):
    """Get index by id from id list, used for the labeling the ag and knn graph.

    Args:
        node_id (int): node id to be searched inside the id list.
        node_id_list (list): list of id from zarr attrs contig id list.
    
    Returns:
        index (int): node_id index based on the node id list.
    """
    for index, id in enumerate(node_id_list):
        if node_id == id:
            return index
    raise TypeError("{} is not found in node id list.".format(node_id))


class Gaussian:
    def __init__(self, sigma=1):
        self.sigma = sigma

    def cal_coefficient(self, dis_array):
        """Calculate the coefficient array according to the
        distance array.
        
        Args:
            dis_array (np.ndarray): distance array.

        Returns:
            coefficient_array (np.ndarray): coefficient array
                after gaussian kernel.
        """
        dis_array /= (2 * self.sigma * self.sigma)
        coefficient_array = np.exp(-dis_array)

        return coefficient_array


def softmax(x):
    """Run softmax function based on axis=0 array.
    
    Args:
        x (np.ndarray): coefficient array.
    
    Returns:
        f_x (np.ndarray): score array.
    """
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def zscore(array, axis=None):
    """Normalize feature using zscore.

    Args:
        array (np.ndarray): feature to be normalized.
        axis (int): axis to average.

    Returns:
        normalized_array (np.ndarray): New normalized Numpy-array
    """
    mean = array.mean(axis=axis)
    std = array.std(axis=axis)
    normalized_array = (array - mean) / std
    return normalized_array


def load_dataset(zarr_dataset_path):
    root = zarr.open(zarr_dataset_path, mode="r")
    contig_id_list = root.attrs["contig_id_list"]
    data_list = []
    for i in tqdm(contig_id_list):
        item = {}
        tnf = np.array(root[i]["tnf_feat"])
        normalized_tnf = zscore(tnf, axis=0)
        rpkm = np.array(root[i]["rpkm_feat"])
        feature = np.concatenate((normalized_tnf, rpkm), axis=0)
        labels = np.array(root[i]["labels"])
        contig_id = np.array(root[i]["id"])
        item["feature"] = feature
        item["labels"] = labels
        item["id"] = contig_id
        data_list.append(item)
    return data_list, contig_id_list


def create_knn_graph(data_list, k, threshold=0.2, compute_method="top_k"):
    """Updates the k nearest neighbors for each contig in the dictionary. 
    
    Alerts: knn graph is created by id vector, stores the neightbors 
    and weights for each neighbor.

    Args:
        data_list (list): list format dataset.

    Returns:
        data_list (list): list format dataset.
    """
    Gau = Gaussian()
    id_list = []
    feature_list = []
    for i in range(len(data_list)):
        feature_list.append(data_list[i]["feature"])
        id_list.append(data_list[i]["id"])
    
    feature_array = np.array(feature_list)
    id_array = np.array(id_list)
    # TODO: remove feature_array.shape[0] by using N.
    for i in trange(feature_array.shape[0], desc="Creating KNN graph......."):
        neighbors_array, distance_array = compute_neighbors(
            index=i,
            feature_array=feature_array,
            id_array=id_array,
            k=k,
            threshold=threshold,
            use_gpu=False,
            compute_method=compute_method,
        )
        data_list[i]["neighbors"] = neighbors_array
        data_list[i]["distances"] = distance_array

    return data_list


def compute_neighbors(
        index,
        feature_array,
        id_array,
        k,
        threshold=0.2,
        use_gpu=False,
        compute_method="top_k",
    ):
    """Compute the neighbors of a single contig;
    
    Args:
        index (int): contig index in data list.
        feature_array (np.array): global feature array.
        id_array (np.array): global id array.
        compute_method (str): method to compute the knn neighbors, including "top_k" and "threshold".
        k (int): top k neighbors from global.
        threshold (float): threshold to filter the knn neighbors.
        use_gpu (boolean): whether to use gpu.
        compute_method (str): methods to compute the neighbors for each contig.
    """
    if compute_method == "top_k":
        tar_feature = np.expand_dims(feature_array[index], axis=0)
        dist_array = np.power((feature_array - tar_feature), 2)
        dist_sum_array = np.sum(dist_array, axis=1).reshape((feature_array.shape[0], 1))
        pairs = np.concatenate((dist_sum_array, id_array), axis=1) # track the id.
        sorted_pairs = pairs[pairs[:, 0].argsort()]
        top_k_pairs = sorted_pairs[1: k+1]
        neighbors_array = top_k_pairs[:, 1]
        distance_array = top_k_pairs[:, 0]
        valid_num = np.sum(distance_array < 5)
        return neighbors_array[:valid_num], distance_array[:valid_num]
    elif compute_method == "threshold":
        tar_feature = np.expand_dims(feature_array[index], axis=0)
        dist_array = np.power((feature_array - tar_feature), 2)
        dist_sum_array = np.sum(dist_array, axis=1).reshape((feature_array.shape[0], 1))
        pairs = np.concatenate((dist_sum_array, id_array), axis=1)
        sorted_pairs = pairs[pairs[:, 0].argsort()]
        mask = sorted_pairs[:, 0] < threshold
        mask = mask.tolist()
        neighbors_list = []
        for i in range(feature_array.shape[0]):
            if i == 0:
                continue
            if mask[i] is True:
                neighbors_list.append(sorted_pairs[i, 1])
        neighbors_array = np.array(neighbors_list)
        return neighbors_array
    else:
        raise NotImplementedError("Only support top_k and threshold method currently.")


def construct_knn_graph(data_list, plotting_graph_size, plotting_contig_list, k=1):
    knn_graph = igraph.Graph()
    knn_graph.add_vertices(plotting_graph_size)
    edge_list = []
    node_label_list = []
    node_num = len(data_list)
    for i in trange(node_num, desc="Creating KNN Subgraph for Visualization"):
        id = int(data_list[i]["id"])
        labels = int(data_list[i]["labels"])
        if id in plotting_contig_list:
            neighbors = data_list[i]["neighbors"]
            labels = data_list[i]["labels"]
            node_label_list.append(int(labels))
            if neighbors is None:
                continue
            k = neighbors.shape[0]
            for j in range(k):
                neighbor_id = int(neighbors[j])
                if neighbor_id in plotting_contig_list:
                    new_index_id = get_index_by_id_from_list(id, plotting_contig_list)
                    new_index_neighbor_id = get_index_by_id_from_list(neighbor_id, plotting_contig_list)
                    if (new_index_neighbor_id, new_index_id) in edge_list:
                        pass
                    else:
                        edge_list.append((new_index_id, new_index_neighbor_id))
                else:
                    pass
        else:
            pass
    knn_graph.add_edges(edge_list)
    knn_graph.vs["label"] = node_label_list
    return knn_graph


def plot_knn_graph(graph, log_path, graph_type, plotting_contig_list, bin_list):
    """Plot graph to disk.
    
    Args:
        graph (igraph.Graph): igraph.Graph object created from plotting contig list.
        log_path (string): predefined path to store the visualized image.
        graph_type (string): graph type to store.
        plotting_contig_list (list): list of sampled contig ids.
        bin_list (2D list): 1d stands for the cluster, 2d stands for the node list.

    Returns:
        path (string): output path of plotting grpah.
    """
    update_graph_labels(graph, plotting_contig_list, bin_list)
    relative_path = "/{}.pdf".format(graph_type)
    path = log_path + relative_path
    layout = graph.layout_fruchterman_reingold()
    visual_style = {}
    visual_style["layout"] = layout
    visual_style["vertex_size"] = 10
    visual_style["bbox"] = (1200, 1200)
    igraph.plot(graph, path, **visual_style)


def label_propagation(labels_array, pre_compute_matrix):
    label_prop_model = lbp(n_jobs=10)
    pre_compute_matrix = np.where(pre_compute_matrix == 1000.0, 0.0, pre_compute_matrix)
    pre_compute_matrix = np.where(pre_compute_matrix != 0.0, 1.0, pre_compute_matrix)
    # np.fill_diagonal(pre_compute_matrix, 1.0)
    for idx, elem in enumerate(zip(labels_array, pre_compute_matrix)):
        label, row = elem
        if label != -1:
            row = np.where(row != 0.0, 0.0, row)
            row[idx] = 1.0
            pre_compute_matrix[idx] = row
    divide = np.sum(pre_compute_matrix, axis=1)
    divide = np.where(divide == 0.0, 1e-18, divide)
    divide = np.expand_dims(divide, 1)
    pre_compute_matrix = pre_compute_matrix / divide.repeat(pre_compute_matrix.shape[0], 1)
    
    labels_array = label_prop_model.fit(pre_compute_matrix, labels_array)
    return labels_array


def remove_ambiguous_label(knn_graph, labels_array, contig_id_list):
    node_num = len(knn_graph)
    prev_label_array = np.copy(labels_array)
    unlabeled = set()
    for i in range(node_num):
        neighbors_array = knn_graph[i]["neighbors"]
        neighbors_num = neighbors_array.shape[0]
        num_diff_bins = set()
        for j in range(neighbors_num):
            neighbors_id = int(neighbors_array[j])
            neighbors_index = get_index_by_id_from_list(
                neighbors_id,
                contig_id_list,
            )
            num_diff_bins.add(prev_label_array[neighbors_index])
        if len(num_diff_bins) > 1 or prev_label_array[i] == 0:
            # directly delete the adge
            knn_graph[i]["neighbors"] = np.array([])
            knn_graph[i]["distances"] = np.array([])
            labels_array[i] = 0
            unlabeled.add(int(knn_graph[i]['id']))
    # reversely delete the edge
    for i in range(node_num):
        delete_idx = []
        for idx, neighbor in enumerate(knn_graph[i]["neighbors"]):
            if int(neighbor) in unlabeled:
                delete_idx.append(idx)
        knn_graph[i]["neighbors"] = np.delete(knn_graph[i]["neighbors"], delete_idx)
        knn_graph[i]["distances"] = np.delete(knn_graph[i]["distances"], delete_idx)
    return knn_graph, labels_array


class PlottingManager:
    def __init__(
        self,
        root_path,
        csv_path,
        graph_type,
        k,
        output_path,
        threshold,
        compute_method,
        **kwargs,
    ):
        self.root_path = root_path
        self.csv_path = csv_path
        self.graph_type = graph_type
        self.k = k
        self.output_path = output_path
        self.threshold = threshold
        self.compute_method = compute_method

    def plot(self):
        plot_graph(
            root_path=self.root_path,
            csv_path=self.csv_path,
            graph_type=self.graph_type,
            k=self.k,
            output_path=self.output_path,
            threshold=self.threshold,
            compute_method=self.compute_method,
        )

    def plot_dbscan_knn(self):
        plot_dbscan_graph(
            root_path=self.root_path,
            csv_path=self.csv_path,
            k=self.k,
            graph_type=self.graph_type,
            output_path=self.output_path,
            threshold=self.threshold,
            compute_method=self.compute_method,
        )


def main(argv=None):
    plotting_manager = PlottingManager(
        **FLAGS.flag_values_dict()
    )
    # plotting_manager.plot()
    plotting_manager.plot_dbscan_knn()


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    flags.DEFINE_string("root_path", "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/mingxing/Metagenomic-Binning/data/cami-m1.zarr", "")
    flags.DEFINE_string("csv_path", "/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/metadecoder/nc_result.csv", "")
    flags.DEFINE_string("graph_type", "test", "")
    flags.DEFINE_integer("k", 3, "")
    flags.DEFINE_string("output_path", "/datahome/datasets/ericteam/csmxrao/DeepMetaBin/mingxing/Metagenomic-Binning/graphs", "")
    flags.DEFINE_float("threshold", 0.2, "")
    flags.DEFINE_string("compute_method", "top_k", "")
    """
    root_path = "/home/eddie/cami1-low-long.zarr"
    csv_nc_path = "/home/eddie/cami1-low/nc_result.csv"
    csv_deepmetabin_path = "/home/eddie/csv-result/cami1-sota.csv"
    csv_gt_path = "/home/eddie/cami1-low/labels.csv"
    csv_gmvae_path = "/home/eddie/gmm_epoch_600.csv"
    csv_gmvae_vamb_path = "/home/eddie/latent_600_nc_result.csv"
    plot_graph(root_path, csv_gt_path)
    """
    app.run(main)
