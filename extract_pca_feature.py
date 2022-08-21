#TODO: update the neighbor features and id mask directly into the zarr dataset,


import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from scipy.linalg import fractional_matrix_power
import zarr
from tqdm import tqdm
from src.utils.util import get_index_by_id_from_array
from absl import app, flags


def cal_laplacian_matrix(ag_adj_matrix):
    """Calculate the laplacian matrix from the adjacency matrix.

    Args:
        ag_adj_matrix (np.ndarray): ag graph based adjacency matrix.
    
    Returns:
        laplacian_matrix (np.ndarray): laplacian matrix of adjacency matrix.
    """
    I = np.identity(ag_adj_matrix.shape[0])
    degree_vector = np.sum(ag_adj_matrix, axis=1)
    degree_vector = np.squeeze(np.asarray(degree_vector))
    D = np.diag(degree_vector)
    D_half_norm = fractional_matrix_power(D, -0.5)
    adj_normalized_matrix = D_half_norm.dot(ag_adj_matrix.toarray()).dot(D_half_norm)
    laplacian_matrix = I - adj_normalized_matrix
    return laplacian_matrix


def create_adj_matrix(root_path):
    """Create the adjacency matrix directly from the processed dataset.

    Args:
        root_path (str): root path of processed directory.

    Returns:
        ag_adj_matrix (np.ndarray): whole adjacency matrix from the original
            graph.
    """
    ag_graph = nx.Graph()
    root = zarr.open(root_path, "r")
    contig_id_list = root.attrs["contig_id_list"]
    id_list = []

    # create the id attrs according to the codebase.
    for i in contig_id_list:
        contig_id = np.array(root[i]["id"])
        contig_id_int = int(contig_id[0])
        id_list.append(contig_id_int)
        ag_graph.add_node(contig_id_int)
    id_vector = np.array(id_list)

    for i in tqdm(contig_id_list, desc="Creating ag and pe graph....."):
        # get the edges id from processed zarr group.
        ag_graph_edges = root[i]["ag_graph_edges"]
        contig_id = np.array(root[i]["id"])
        contig_id_int = int(contig_id[0])

        # update ag graph edges.
        for edge_pair in ag_graph_edges:
            neighbor_id = edge_pair[1]
            if neighbor_id in contig_id_list:
                neighbor_index = get_index_by_id_from_array(neighbor_id, id_vector)
                neighbor_id_int = id_vector[neighbor_index]
                ag_graph.add_edge(contig_id_int, neighbor_id_int)

    ag_adj_matrix = nx.adjacency_matrix(ag_graph)
    return ag_adj_matrix, id_vector


def filter_dead_ends_graph(root_path, ag_adj_matrix, id_vector):
    """Filter the dead ends from the graph.

    Args:
        root_path (str): root path of processed directory.
        ag_adj_matrix (np.ndarray): whole adjacency matrix from the original 
            graph.
        id_vector (np.ndarray): id vector of cotnigs.
    """
    root = zarr.open(root_path, "a")
    contig_id_list = root.attrs["contig_id_list"]
    non_dead_ends_contig_list = []
    id_mask_list = []
    degree_tensor = np.sum(ag_adj_matrix, axis=1)
    N = degree_tensor.shape[0]
    non_dead_ends_ag_graph = nx.Graph()
    for i in range(N):
        if degree_tensor[i] != 0:
            non_dead_ends_contig_list.append(int(id_vector[i]))
            id_mask_list.append(1)
            non_dead_ends_ag_graph.add_node(int(id_vector[i]))
        else:
            id_mask_list.append(0)
    
    print("length of non dead ends contig id list: {}".format(len(non_dead_ends_contig_list)))
    root.attrs["non_dead_ends_contig_id_list"] = non_dead_ends_contig_list

    # recreate the ag graph of non dead ends.
    for i in range(len(non_dead_ends_contig_list)):
        # get the contig id from processed zarr group.
        contig_id_int = non_dead_ends_contig_list[i]
        ag_graph_edges = root[contig_id_int]["ag_graph_edges"]

        # filter the ag graph edges of long contig and non dead ends contig.
        for edge_pair in ag_graph_edges:
            neighbor_id = edge_pair[1]
            # filter the long contig.
            if neighbor_id in contig_id_list:
                neighbor_index = get_index_by_id_from_array(neighbor_id, id_vector)
                neighbor_id_int = id_vector[neighbor_index]
                # filter the non dead ends condition.
                if neighbor_id_int in non_dead_ends_contig_list:
                    non_dead_ends_ag_graph.add_edge(contig_id_int, neighbor_id_int)

    ag_adj_matrix = nx.adjacency_matrix(non_dead_ends_ag_graph)
    lap_ag_adj_matrix = cal_laplacian_matrix(ag_adj_matrix)
    id_mask = np.array(id_mask_list)
    return lap_ag_adj_matrix, id_mask


class ExtractManager:
    def __init__(
        self,
        dataset_path,
        save_feature_path,
        save_mask_path,
        pca_shape=5,
        **kwargs,
    ):
        self.dataset_path = dataset_path
        self.pca_shape = pca_shape
        self.save_feature_path = save_feature_path
        self.save_mask_path = save_mask_path

    def extract_pca_feature(self):
        ag_adj_matrix, id_vector = create_adj_matrix(
            self.dataset_path        
        )
        print("adj matrix already created, shape is {}".format(
            id_vector.shape    
        ))
        lap_ag_adj_matrix, id_mask = filter_dead_ends_graph(
            self.dataset_path,
            ag_adj_matrix,
            id_vector,
        )
        print("lap matrix already created, shape is {}".format(
            lap_ag_adj_matrix.shape
        ))
        pca = PCA(n_components=self.pca_shape)
        out_feat = pca.fit_transform(lap_ag_adj_matrix)
        np.save(self.save_feature_path, out_feat)
        np.save(self.save_mask_path, id_mask)


def main(argv=None):
    extract_manager = ExtractManager(**FLAGS.flag_values_dict())
    extract_manager.extract_pca_feature()


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    flags.DEFINE_string("dataset_path", "", "")
    flags.DEFINE_string("save_feature_path", "", "")
    flags.DEFINE_string("save_mask_path", "", "")
    app.run(main)
