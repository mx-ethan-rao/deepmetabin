import numpy as np
import igraph
import time
import torch
import zarr
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy import sparse
from src.utils.plot import update_graph_labels, COLOUR_DICT
from src.utils.metric import Metric
from sklearn.manifold import TSNE
from mingxing.test_label_propagation import lbp


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


def get_index_by_id_from_array(node_id, id_array):
    """Get index by id from id array inside dataset for refine the sequence of
    graph.
    
    Args:
        node_id (int): node id to be searched inisde the id_array.
        id_array (np.ndarray): id array.

    Returns:
        index (int): node_id index based on the id_array.
    """
    for index, id in enumerate(id_array):
        if node_id == id:
            return index
    raise TypeError("{} is not found in node id list.".format(node_id))


def get_index_by_id_from_tensor(node_id, id_tensor):
    """Get index by id from id tensor inside the batch for sampling
    long contig logits from whole logits.

    Args:
        node_id (int): node id to be searched inside id_tensor.
        id_tensor (tensor): id tensor.
    
    Returns:
        index (int): node_id index based on the id_tensor.
    """
    for index, id in enumerate(id_tensor):
        if node_id == id:
            return index
    raise TypeError("{} is not found in node id list.".format(node_id))


def summary_cluster_list_from_csv(csv_path):
    """Summary cluster list from csv file.

    Args:
        csv_path (string): path of csv file.

    Returns:
        cluster_list (list): list of bin_id.
        num_cluser (int): number of bins in csv file.
    """
    cluster_list = []
    with open(csv_path) as file:
        reader = csv.reader(file)
        for row in reader:
            cluster_id = row[1]
            if cluster_id in cluster_list:
                pass
            else:
                cluster_list.append(cluster_id)
    num_cluster = len(cluster_list)
    
    return cluster_list, num_cluster


def summary_bin_list_from_csv(csv_path):
    """Summary contig id into bin list.

    Args:
        csv_path (string): path of csv file.
    
    Returns:
        final_bin_list (2D list): 1st dimension stands for bin list,
            2nd dimension stands for the contig id each bin.
    """
    cluster_list, num_cluster = summary_cluster_list_from_csv(csv_path)
    bin_list = [[] for i in range(num_cluster)]
    with open(csv_path) as file:
        reader = csv.reader(file)
        for row in reader:
            node_id = int(row[0].split("_")[1])
            cluster_id = row[1]
            for index, cluster in enumerate(cluster_list):
                if cluster_id == cluster:
                    bin_list[index].append(node_id)
    final_bin_list = []
    for i in bin_list:
        if len(i) != 0:
            final_bin_list.append(i)
    print("final bins: {}".format(len(final_bin_list)))
    return final_bin_list


def get_non_labeled_id_list_from_bin_lists(gd_bin_list, result_bin_list):
    """Calculate the non labeled contig id list from result bin list and 
    ground truth bin list.

    Args:
        gd_bin_list (2D list): bins of binning result nodes.
        result_bin_list (2D list): bins of ground truth nodes.
    
    Returns:
        non_labeled_id_list (list): list of non labeled contig id from 
            compared model binning result compared to grounf truth.
    """
    def collect_contig_id_list(bin_list, list_type):
        contig_id_list = []
        bins_num = len(bin_list)
        for i in trange(bins_num, desc="collecting contig id list from {}".format(list_type)):
            for j in bin_list[i]:
                if j in contig_id_list:
                    raise ValueError("Duplicate contig id detected in \
                    {} bin list {}".format(list_type, j))
                else:
                    contig_id_list.append(j)
        return contig_id_list
    
    ground_truth_contig_id_list = collect_contig_id_list(gd_bin_list, "ground truth")
    result_contig_id_list = collect_contig_id_list(result_bin_list, "result")
    non_labeled_id_list = []
    
    for i in tqdm(ground_truth_contig_id_list, desc="comparing bin lists."):
        if i not in result_contig_id_list:
            non_labeled_id_list.append(i)
    return non_labeled_id_list


def load_graph(graph_path: str):
    """Function to load graph output from MetaSpades assembeler to contig_id_list
    and edges list.
    
    Args:
        graph_path (string): path of the input graph.

    Returns:
        contig_id_dict_list (list): list of the contig id, contig_id_dict_list[i]
            paired with edges_list[i].
        edges_list (list): list of edges list every contig, same as above.
    """
    contig_id_dict_list = []
    edges_list = []
    with open(graph_path) as graph_file:
        line = graph_file.readline()
        edges = [] # initialise to avoid non defintion.
        contig_id = -1 # initialise to remove first node issue.
        while line != "":
            line = line.strip()
            strings = line[:-1].split()
            if line[-1] == ":":
                # last contig and edges.
                contig_id_dict_list.append(contig_id)
                edges_list.append(edges)
                # new iteration contig and edges.
                contig_id = int(strings[0].split("_")[1])
                edges = []
            if line[-1] == ";":
                edges_contig_id = int(strings[0].split("_")[1])
                edges.append((contig_id, edges_contig_id))
            line = graph_file.readline()
    
    # remove first placeholder for edges and contig_id when reading graph file.
    contig_id_dict_list.pop(0)
    edges_list.pop(0)
    return contig_id_dict_list, edges_list


def describe_dataset(processed_zarr_dataset_path):
    """Function to describe the processed zarr dataset, which includes the original contig list
    and long contig list.

    Args:
        processed_zarr_dataset_path (string): path of processed zarr dataset.

    Returns:
        None.
    """
    root = zarr.open(processed_zarr_dataset_path, mode="r")
    contig_id_list = root.attrs["contig_id_list"]
    long_contig_id_list = root.attrs["long_contig_id_list"]
    print("Total contig number is {}".format(len(contig_id_list)))
    print("Long contig number is {}".format(len(long_contig_id_list)))


def summary_bin_list_from_batch(batch, bin_tensor):
    """Summary bin list according to the contigs batch and bin_tensor output from
    model. Need to pay attention to the unlabeled contig inside the ground truth 
    bins.

    result_bin_dict (list) holds the dictionary of bin_id.
    result_bin_dist (2D list) holds the list of contig id per bin.

    Args:
        batch (dict): batch of validation dataloader.
        bin_tensor (tensor): tensor of binning result.

    Returns:
        gd_bin_list (2D list): Id stands for the cluster, 2d stands for the node list.
        result_bin_list (2D list): 1d stands for the cluster, 2d stands for the node list.
        non_labeled_id_list (list): list of non ground truth labeled id list.
    """
    result_bin_dict = []
    label_tensor = torch.squeeze(batch["labels"])
    id_tensor = torch.squeeze(batch["id"])
    for i in range(bin_tensor.shape[0]):
        result_bin_id = int(bin_tensor[i])
        if not result_bin_id in result_bin_dict:
            result_bin_dict.append(result_bin_id)
    result_bin_num = len(result_bin_dict)

    gd_bin_dict = []
    non_labeled_id_list = []
    for i in range(bin_tensor.shape[0]):
        bin_id = int(label_tensor[i])
        # remove ground truth bin_id -1 cluster.
        if not bin_id in gd_bin_dict and bin_id != -1:
            gd_bin_dict.append(bin_id)
    gd_bin_num = len(gd_bin_dict)

    result_bin_list = [[] for i in range(result_bin_num)]
    gd_bin_list = [[] for i in range(gd_bin_num)]
    for i in trange(bin_tensor.shape[0]):
        contig_id = int(id_tensor[i])
        i_gd_bin = int(label_tensor[i])

        if i_gd_bin == -1:
            non_labeled_id_list.append(contig_id)

        # remove no labeled contig.
        for index, bin_id in enumerate(gd_bin_dict):
            if i_gd_bin == bin_id:
                gd_bin_list[index].append(contig_id)
        
        result_bin = int(bin_tensor[i])
        for index, bin_id in enumerate(result_bin_dict):
            if result_bin == bin_id:
                result_bin_list[index].append(contig_id)

    return gd_bin_list, result_bin_list, non_labeled_id_list


def get_long_contig_logits_from_batch(batch, logits):
    """Get the long contig logits tensor from whole contig logits tensor, which aims to sample
    the long contig logits from the result, instead of recomputing the long contigs feature
    again.

    Args:
        batch (dict): batch of training dataloader.
        logits (tensor): contig logits tensor from encoder.

    Returns:
        long_contig_logits (tensor): tensor of long contig logits.
    """
    id_tensor = batch["id"]
    long_id_tensor = batch["long_contig_id"]
    long_contig_logits = []
    for i in range(long_id_tensor.shape[0]):
        node_id = int(long_id_tensor[i].item())
        index = get_index_by_id_from_tensor(node_id, id_tensor)
        long_contig_logits.append(logits[index])
    long_contig_logits = torch.stack(long_contig_logits)
    return long_contig_logits


def construct_assessment_matrix(ground_truth_bins, binning_result_bins, non_labeled_id_list):
    """Construct the assessment matrix based on ground truth bin list, model result
    bin list and non labeled id list.

    assessment matrix: (k+1, s+1) size
        rows stand for the model result bins varied in ground truth bins.
        columns stand for the ground truth bins varied in model result bins.
       
    Args:
        ground_truth_bins (2D list): bins of ground truth nodes, reference to the 
    construction of bin_list, already removed non labeled bin.
        binning_result_bins (2D list): bins of binning result nodes, reference to 
    the construction of bin_list.
        non_labeled_id_list (list): list of non labeled id in ground truth bins.

    Returns:
        assessment_matrix (np.ndarray): assessment matrix with shape (k+1, s+1).
    """
    k = len(binning_result_bins)
    s = len(ground_truth_bins)
    matrix_shape = (k + 1, s + 1)
    assessment_matrix = np.zeros(matrix_shape, dtype=int)
    for i in trange(k):
        bin_i = binning_result_bins[i]
        for node in bin_i:                
            for j in range(s):
                bin_j = ground_truth_bins[j]
                if node in bin_j:
                    assessment_matrix[i][j] += 1
                    continue

            # perform for the last column non labeled id.
            if node in non_labeled_id_list:
                assessment_matrix[i][s] += 1
    return assessment_matrix


def create_ag_graph(plotting_graph_size, processed_zarr_dataset_path, plotting_contig_list):
    """Create ag graph based on the sampled contig list, remember to reindex the contig
    id, cause igraph.Graph object indexing from 0 to self.plot_graph_size-1.

    Args:
        plotting_graph_size (int): plotting graph size.
        processed_zarr_dataset_path (string): path of processed zarr dataset.
        plotting_contig_list (list): list of sampled contig ids.

    Returns:
        ag_graph (igraph.Graph): reindexed igraph.Graph object.
    """
    ag_graph = igraph.Graph()
    ag_graph.add_vertices(plotting_graph_size)
    root = zarr.open(processed_zarr_dataset_path, mode="r")
    edge_list = []
    for id in tqdm(root.attrs["contig_id_list"], desc="Creating AG Subgraph for Visualization"):
        if id in plotting_contig_list:
            edges = root[id]["ag_graph_edges"]
            for edge_pair in edges:
                neighbor_id = edge_pair[1]
                if neighbor_id in plotting_contig_list:
                    new_index_id = get_index_by_id_from_list(id, plotting_contig_list)
                    new_index_neighbor_id = get_index_by_id_from_list(neighbor_id, plotting_contig_list)
                    edge_list.append((new_index_id, new_index_neighbor_id))
                else:
                    pass
        else:
            pass
    ag_graph.add_edges(edge_list)
    return ag_graph


def create_knn_graph(plotting_graph_size, plotting_contig_list, k, batch):
    """Create knn graph based on the sampled contig list and its neighbors, 
    remember to reindex the contig id, cause igraph.Graph object indexing 
    from 0 to self.plot_graph_size-1.

    Args:
        plotting_graph_size (int): plotting graph size.
        plotting_contig_list (list): list of sampled contig ids.
        k (int): k nearest neighbors.
        batch (dictionary): batch from datamodule to get the neighbors id.

    Returns:
        knn_graph (igraph.Graph): reindexed igraph.Graph object.
    """
    knn_graph = igraph.Graph()
    knn_graph.add_vertices(plotting_graph_size)
    edge_list = []
    id_tensor = torch.squeeze(batch["id"])
    neighbor_tensor = torch.squeeze(batch["neighbors"])
    node_num = id_tensor.shape[0]
    for i in trange(node_num, desc="Creating KNN Subgraph for Visualization"):
        id = int(id_tensor[i])
        if id in plotting_contig_list:
            neighbors = neighbor_tensor[i]
            for j in range(k):
                neighbor_id = int(neighbors[j])
                if neighbor_id in plotting_contig_list:
                    new_index_id = get_index_by_id_from_list(id, plotting_contig_list)
                    new_index_neighbor_id = get_index_by_id_from_list(neighbor_id, plotting_contig_list)
                    edge_list.append((new_index_id, new_index_neighbor_id))
                else:
                    pass
        else:
            pass
    knn_graph.add_edges(edge_list)
    return knn_graph


def plot_graph(graph, log_path, graph_type, plotting_contig_list, bin_list):
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
    relative_path = "/{}-{}.png".format(graph_type, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    path = log_path + relative_path
    layout = graph.layout_fruchterman_reingold()
    visual_style = {}
    visual_style["layout"] = layout
    visual_style["vertex_size"] = 10
    visual_style["bbox"] = (800, 800)
    igraph.plot(graph, path, **visual_style)
    return path


def log_ag_graph(
    plotting_graph_size,
    processed_zarr_dataset_path,
    plotting_contig_list,
    log_path,
    gd_bin_list,
    result_bin_list,
):
    """Wrapper function inside validation step, plot graph to disk.
    
    Args:
        plotting_graph_size (int): plotting graph size.
        processed_zarr_dataset_path (string): path of processed zarr dataset.
        plotting_contig_list (list): list of sampled contig ids.
        graph (igraph.Graph): igraph.Graph object created from plotting contig list.
        log_path (string): predefined path to store the visualized image.
        gd_bin_list (2D list): ground truth bin list, 1d stands for the cluster, 
            2d stands for the node list.
        result_bin_list (2D list): result bin list, 1d stands for the cluster, 
            2d stands for the node list.

    Returns:
        gd_ag_graph_path (string): output path of plotting gd_ag_grpah.
        result_ag_graph_path (string): output path of plotting result_ag_grpah.
    """
    gd_ag_graph = create_ag_graph(
        plotting_graph_size=plotting_graph_size, 
        processed_zarr_dataset_path=processed_zarr_dataset_path,
        plotting_contig_list=plotting_contig_list,
    )
    result_ag_graph = create_ag_graph(
        plotting_graph_size=plotting_graph_size, 
        processed_zarr_dataset_path=processed_zarr_dataset_path,
        plotting_contig_list=plotting_contig_list,
    )
    gd_ag_graph_path = plot_graph(
        graph=gd_ag_graph,
        log_path=log_path,
        graph_type="gd-ag",
        plotting_contig_list=plotting_contig_list,
        bin_list=gd_bin_list,
    )
    result_ag_graph_path = plot_graph(
        graph=result_ag_graph,
        log_path=log_path,
        graph_type="result-ag",
        plotting_contig_list=plotting_contig_list,
        bin_list=result_bin_list,
    )
    return gd_ag_graph_path, result_ag_graph_path


def log_knn_graph(
    plotting_graph_size,
    plotting_contig_list,
    k,
    batch,
    log_path,
    gd_bin_list,
    result_bin_list,
):
    """Wrapper function inside validation step, plot graph to disk.
    
    Args:
        plotting_graph_size (int): plotting graph size.
        plotting_contig_list (list): list of sampled contig ids.
        k (int): k neighbors used in knn graph.
        batch (dictionary):  batch from datamodule to get the neighbors id.
        log_path (string): predefined path to store the visualized image.
        gd_bin_list (2D list): ground truth bin list, 1d stands for the cluster, 
            2d stands for the node list.
        result_bin_list (2D list): result bin list, 1d stands for the cluster, 
            2d stands for the node list.

    Returns:
        gd_knn_graph_path (string): output path of plotting gd_knn_grpah.
        result_knn_graph_path (string): output path of plotting result_knn_grpah.
    """
    gd_knn_graph = create_knn_graph(
        plotting_graph_size=plotting_graph_size,
        plotting_contig_list=plotting_contig_list,
        k=k,
        batch=batch,
    )
    result_knn_graph = create_knn_graph(
        plotting_graph_size=plotting_graph_size, 
        plotting_contig_list=plotting_contig_list,
        k=k,
        batch=batch,
    )
    gd_knn_graph_path = plot_graph(
        graph=gd_knn_graph,
        log_path=log_path,
        graph_type="gd-knn",
        plotting_contig_list=plotting_contig_list,
        bin_list=gd_bin_list,
    )
    result_knn_graph_path = plot_graph(
        graph=result_knn_graph,
        log_path=log_path,
        graph_type="result-knn",
        plotting_contig_list=plotting_contig_list,
        bin_list=result_bin_list,
    )
    return gd_knn_graph_path, result_knn_graph_path


def evaluate(gd_bin_list, result_bin_list, non_labeled_id_list, unclassified):
    """Evaluate the performance from the ground truth labels and binning result.
    
    Args:
        gd_bin_list (2D list): bins of ground truth nodes.
        result_bin_list (2D list): bins of binning result nodes.
        n (int): num of of nodes.
        unclassified (int): unclassified nodes, 0 for gmgat.
    
    Returns:
        precision (float): precision metric.
        recall (float): recall metric.
        ARI (float): ARI metric.
        F1 (float): F1 metric.
    """
    assessment_matrix = construct_assessment_matrix(gd_bin_list, result_bin_list, non_labeled_id_list)
    metric = Metric()
    precision = metric.GetPrecision(assessment_matrix)
    recall = metric.GetRecall(assessment_matrix, unclassified)
    ARI = metric.GetARI(assessment_matrix)
    F1 = metric.GetF1(precision, recall)
    return precision, recall, ARI, F1


def log_tsne_figure(batch, latent, log_path):
    """Log the tsne figure on latent space z using sklearn tsne function.
    
    Args:
        batch (dictionary): batch from datamodule to get the node labels.
        latent (tensor): z tensor output from the encoder.
        log_path (string): path to store the plotting tsne figure.
    
    Returns:
        result_tsne_figure_path (string): path of plotting tsne figure.
    """
    labels = batch["labels"]
    relative_path = "/{}-{}.png".format("tsne", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    result_tsne_figure_path = log_path + relative_path
    tsne = TSNE(n_components=2, learning_rate='auto')
    compressed_latent = tsne.fit_transform(latent)
    figure = plt.figure(figsize=(10, 10))
    plt.scatter(
        compressed_latent[:, 0],
        compressed_latent[:, 1],
        c=labels,
        marker='.',
        s=10,
    )
    figure.savefig(result_tsne_figure_path)
    return result_tsne_figure_path


def log_similarity_matrix(batch, latent, log_path):
    """Log the simialirty matrix on latent space z using seaborn package.

    Args: 
        batch (dictionary): batch from datamodule to get the node labels.
        latent (tenor): z tensor output from the encoder.
        log_path (string): path to store the plotting similarity matrix figure.

    Retruns:
        result_similarity_matrix_path (string): path of plotting similarity matrix figure.
    """
    pass


def refine_gmm(gmm, batch, latent):
    """Use EM algorithm to further train the latent vector and predict the target cluster.

    Args:
        gmm (sklearn.gmm): GMM module from sklearn.
        batch (dictionary): batch from datamodule to get the node labels.
        latent (tensor): z tensor output from the encoder.

    Returns:
        gmm_precision (float): GMM precision metric.
        gmm_recall (float): GMM recall metric.
        gmm_ARI (float): GMM ARI metric.
        gmm_F1 (float): GMM F1 metric.
    """
    latent = latent.detach().numpy()
    predicts = gmm.fit_predict(latent)
    gmm_gd_bin_list, gmm_result_bin_list, non_labeled_id_list = summary_bin_list_from_batch(
        batch=batch,
        bin_tensor=predicts,
    )
    gmm_precision, gmm_recall, gmm_ARI, gmm_F1 = evaluate(
        gd_bin_list=gmm_gd_bin_list,
        result_bin_list=gmm_result_bin_list,
        non_labeled_id_list=non_labeled_id_list,
        unclassified=0,
    )
    return predicts, gmm_precision, gmm_recall, gmm_ARI, gmm_F1


def generate_csv_from_bin_tensor(output_csv_path, id_tensor, results):
    with open(output_csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(id_tensor.shape[0]):
            contig_id = id_tensor[i].detach().numpy()
            bin_id = int(results[i])
            output_contig_head = "NODE_" + "{}".format(int(contig_id))
            writer.writerow([output_contig_head, bin_id])


def visualize_graph(bin_list, contig_list, save_path):
    """Visualize the graph from model results.

    Args:
        bin_list (list): cluster list including of 
        contig_list (list): list of contig id to log.
        save_path (str): plotting path of graph.
    """
    graph = igraph.Graph()
    graph_size = len(contig_list)
    graph.add_vertices(graph_size)
    node_colour_list = []
    node_label_list = []
    node_weights_list = []
    node_embedding_list = []

    for bin_id, bin in enumerate(bin_list):
        for contig in bin:
            contig_id = contig["contig_id"]
            contig_embedding = contig["contig_embedding"]

            node_colour_list.append(COLOUR_DICT[bin_id])
            node_label_list.append(contig_id)
            node_embedding_list.append(contig_embedding)

    embedding_array = np.array(node_embedding_list)
    for i in trange(len(node_embedding_list)):
        tar_feature = np.expand_dims(embedding_array[i], axis=0)
        dist_array = np.power((embedding_array - tar_feature), 2)
        dist_sum_array = np.sum(dist_array, axis=1).reshape((embedding_array.shape[0], 1))
        node_weights_list.extend(dist_sum_array[i] for i in range(embedding_array.shape[0]))
    
    layout = graph.layout_fruchterman_reingold()
    graph.vs["color"] = node_colour_list
    graph.vs["label"] = node_label_list
    graph.es["label"] = node_weights_list
    visual_style = {}
    visual_style["layout"] = layout
    visual_style["vertex_size"] = 10
    visual_style["bbox"] = (800, 800)
    igraph.plot(graph, save_path, **visual_style)


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
        valid_num = np.sum(distance_array < 10)
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
