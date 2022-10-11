import scipy
import zarr
import torch
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power
import networkx as nx
import numpy as np
from tqdm import tqdm, trange
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset
from src.utils.util import (
    Gaussian,
    softmax,
    get_index_by_id_from_array,
    compute_neighbors,
    create_matrix,
    label_propagation,
    remove_ambiguous_label,
)


class GraphDataset(Dataset):
    """Graph dataset object, loading dataset in a batch-wise manner.

    Args:
        zarr_dataset_path (string): path of zarr dataset root.
        k (int): k nearest neighbor used in neighbors.
        sigma (float): Gaussian variance when computing neighbors coefficient.
        use_neighbor_feature (boolean): whether to use the reconstructing 
            neighbors strategy.

    Each contig is stored into dictionary, with following keys:
        - feature: concated tnf and rpkm feature dim (104, 1);

    """
    # TODO: add feature per batch as comments.
    def __init__(
        self,
        zarr_dataset_path: str = "",
        k: int = 5,
        sigma: int =1,
        use_neighbor_feature=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.zarr_dataset_path = zarr_dataset_path
        self.k = k
        self.use_neighbor_feature = use_neighbor_feature
        self.Gaussian = Gaussian(sigma=sigma)
        self.data = []

        self.data = self.load_dataset(zarr_dataset_path)

    def load_dataset(self, zarr_dataset_path):
        data_list = self._load_graph_attrs(zarr_dataset_path)
        if self.use_neighbor_feature:
            data_list = self.create_knn_graph(data_list)
        return data_list

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _load_graph_attrs(self, zarr_dataset_path: str):
        root = zarr.open(zarr_dataset_path, mode="r")
        contig_id_list = root.attrs["contig_id_list"]
        data_list = []
        for i in tqdm(contig_id_list):
            item = {}
            tnf = np.array(root[i]["tnf_feat"])
            normalized_tnf = self.zscore(tnf, axis=0)
            rpkm = np.array(root[i]["rpkm_feat"])
            feature = np.concatenate((normalized_tnf, rpkm), axis=0)
            labels = np.array(root[i]["labels"])
            contig_id = np.array(root[i]["id"])
            item["feature"] = feature
            item["labels"] = labels
            item["id"] = contig_id
            data_list.append(item)
        return data_list

    def create_knn_graph(self, data_list):
        """Updates the k nearest neighbors for each contig in the dictionary. 
        
        Alerts: knn graph is created by id vector, stores the neightbors 
        and weights for each neighbor.

        Args:
            data_list (list): list format dataset.

        Returns:
            data_list (list): list format dataset.
        """
        id_list = []
        feature_list = []
        for i in range(len(data_list)):
            feature_list.append(data_list[i]["feature"])
            id_list.append(data_list[i]["id"])
        
        feature_array = np.array(feature_list)
        id_array = np.array(id_list)
        # TODO: remove feature_array.shape[0] by using N.
        for i in trange(feature_array.shape[0], desc="Creating KNN graph......."):
            tar_feature = np.expand_dims(feature_array[i], axis=0)
            dist_array = np.power((feature_array - tar_feature), 2)
            dist_sum_array = np.sum(dist_array, axis=1).reshape((feature_array.shape[0], 1))
            pairs = np.concatenate((dist_sum_array, id_array), axis=1) # track the id.
            sorted_pairs = pairs[pairs[:, 0].argsort()]
            top_k_pairs = sorted_pairs[1: self.k+1]
            k_nearest_dis_array = self.Gaussian.cal_coefficient(top_k_pairs[:, 0])
            normalized_k_dis_array = softmax(k_nearest_dis_array)
            neighbors_array = top_k_pairs[:, 1]
            data_list[i]["neighbors"] = neighbors_array
            data_list[i]["weights"] = normalized_k_dis_array
            if self.use_neighbor_feature:
                neighbor_feature = []
                for index in range(self.k):
                    neighbor_id = int(neighbors_array[index])
                    # search the feature by the id array tensor.
                    for j in range(feature_array.shape[0]):
                        if int(id_array[j]) == neighbor_id:
                            neighbor_feature.append(feature_array[j])
                neighbor_feature = np.stack(neighbor_feature, axis=0)
                data_list[i]["neighbors_feature"] = neighbor_feature

        return data_list

    @staticmethod
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


class KNNGraphDataset(Dataset):
    """Graph dataset object, loading dataset in a batch-wise manner.

    Args:
        zarr_dataset_path (string): path of zarr dataset root.
        k (int): k nearest neighbor used in neighbors.
        sigma (float): Gaussian variance when computing neighbors coefficient.
        use_neighbor_feature (boolean): whether to use the reconstructing 
            neighbors strategy.

    Each contig is stored into dictionary, with following keys:
        - feature: concated tnf and rpkm feature dim (104, 1);

    """
    # TODO: add feature per batch as comments.
    def __init__(
        self,
        zarr_dataset_path: str = "",
        k: int = 5,
        sigma: int =1,
        use_neighbor_feature=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.zarr_dataset_path = zarr_dataset_path
        self.k = k
        self.use_neighbor_feature = use_neighbor_feature
        self.Gaussian = Gaussian(sigma=sigma)
        self.dbscan = DBSCAN(
            eps=1.65,
            metric="precomputed",
            min_samples=2,
        )
        self.data = []

        self.data = self.load_dataset(zarr_dataset_path)

    def load_dataset(self, zarr_dataset_path):
        data_list, contig_id_list = self._load_graph_attrs(zarr_dataset_path)
        if self.use_neighbor_feature:
            data_list = self.create_knn_graph(
                data_list=data_list,
                k=self.k,
                threshold=0.2,
                compute_method="top_k",
            )
        pre_compute_matrix = create_matrix(
            data_list=data_list,
            contig_list=contig_id_list,
        )
        
        # Get DBSCAN clustering result.
        cluster_result = self.dbscan.fit(pre_compute_matrix)
        labels_array = cluster_result.labels_
        labels_array = label_propagation(labels_array, pre_compute_matrix)
        data_list, labels_array = remove_ambiguous_label(data_list, labels_array, contig_id_list)
        data_list = self.filter_knn_graph(data_list)

        return data_list

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _load_graph_attrs(self, zarr_dataset_path: str):
        root = zarr.open(zarr_dataset_path, mode="r")
        contig_id_list = root.attrs["contig_id_list"]
        data_list = []
        for i in tqdm(contig_id_list):
            item = {}
            tnf = np.array(root[i]["tnf_feat"])
            normalized_tnf = self.zscore(tnf, axis=0)
            rpkm = np.array(root[i]["rpkm_feat"])
            feature = np.concatenate((normalized_tnf, rpkm), axis=0)
            labels = np.array(root[i]["labels"])
            contig_id = np.array(root[i]["id"])
            item["feature"] = feature
            item["labels"] = labels
            item["id"] = contig_id
            data_list.append(item)
        return data_list, contig_id_list

    def create_knn_graph(self, data_list, k, threshold=0.2, compute_method="top_k"):
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
        for i in trange(feature_array.shape[0], desc="Creating KNN graph using DBSCAN......."):
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

    def filter_knn_graph(self, data_list):
        # filter the knn graph and create neighbor features and neighbors mask instead.
        id_list = []
        feature_list = []
        for i in range(len(data_list)):
            feature_list.append(data_list[i]["feature"])
            id_list.append(data_list[i]["id"])

        feature_array = np.array(feature_list)
        id_array = np.array(id_list)
        node_num = len(data_list)
        for i in range(node_num):
            neighbors_array = data_list[i]["neighbors"]
            weights_array = data_list[i]["distances"]
            del data_list[i]["neighbors"]
            del data_list[i]["distances"]
            neighbors_num = neighbors_array.shape[0]
            neighbor_feature_list = []
            neighbor_weight_list = []
            # hardcode k = 3 here.
            if neighbors_num == 3:
                for index in range(3):
                    neighbor_id = int(neighbors_array[index])
                    for j in range(feature_array.shape[0]):
                        if int(id_array[j]) == neighbor_id:
                            neighbor_feature_list.append(feature_array[j])
                neighbors_feature = np.stack(neighbor_feature_list, axis=0)
                neighbors_feature_mask = np.array([1, 1, 1])
                neighbors_weight = weights_array
            elif neighbors_num == 2:
                for index in range(2):
                    neighbor_id = int(neighbors_array[index])
                    for j in range(feature_array.shape[0]):
                        if int(id_array[j]) == neighbor_id:
                            neighbor_feature_list.append(feature_array[j])
                    neighbor_weight_list.append(weights_array[index])
                neighbor_feature_list.append(neighbor_feature_list[0])
                neighbor_weight_list.append(100000.0)
                neighbors_feature = np.stack(neighbor_feature_list, axis=0)
                neighbors_feature_mask = np.array([1, 1, 0])
                neighbors_weight = np.stack(neighbor_weight_list, axis=0)

            elif neighbors_num == 1:
                for index in range(1):
                    neighbor_id = int(neighbors_array[index])
                    for j in range(feature_array.shape[0]):
                        if int(id_array[j]) == neighbor_id:
                            neighbor_feature_list.append(feature_array[j])
                    neighbor_weight_list.append(weights_array[index])
                neighbor_feature_list.append(neighbor_feature_list[0])
                neighbor_feature_list.append(neighbor_feature_list[0])
                neighbor_weight_list.append(100000.0)
                neighbor_weight_list.append(100000.0)
                neighbors_feature = np.stack(neighbor_feature_list, axis=0)
                neighbors_feature_mask = np.array([1, 0, 0])        
                neighbors_weight = np.stack(neighbor_weight_list, axis=0)
            else:
                for index in range(3):
                    neighbor_feature_list.append(feature_array[0])
                neighbors_feature = np.stack(neighbor_feature_list, axis=0)
                neighbors_feature_mask = np.array([0, 0, 0])
                neighbors_weight = np.array([0.0, 0.0, 0.0])
            
            gau = Gaussian(sigma=1.0) 
            neighbor_weight = gau.cal_coefficient(neighbors_weight)
            updated_neighbor_weight = softmax(neighbor_weight)
            data_list[i]["neighbors_feature"] = neighbors_feature
            data_list[i]["neighbors_feature_mask"] = neighbors_feature_mask
            data_list[i]["neighbors_weight"] = updated_neighbor_weight
        return data_list

    @staticmethod
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


class GMGATSingleGraphDataset(Dataset):
    """Graph dataset object, loading dataset in a whole graph manner.

    Alerts: Different from GraphDataSet above, GMGATGraphDataset has only
    one batch, cause only one graph per type each dataset.

    Feature:
        batch["id"]: id vector with shape (N, 1);
        batch["labels"]: labels vector with shape (N, 1);
        batch["feature"]: input feature vector with shape (N, 104);

    Args:
        zarr_dataset_path (string): path of zarr dataset root.
        k (int): k nearest neighbor used in neighbors.
        sigma (float): Gaussian variance when computing neighbors coefficient.
    """
    def __init__(
        self,
        zarr_dataset_path: str = "",
        U_feature_path: str = "",
        k: int = 15,
        sigma: int = 1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.zarr_dataset_path = zarr_dataset_path
        self.U_feature_path = U_feature_path
        self.k = k
        self.sigma = sigma
        self.data = self.load_dataset(zarr_dataset_path)
        self.Gaussian = Gaussian(sigma=sigma)

    def load_dataset(self, zarr_dataset_path):
        data = []
        data.append(self.generate_batch_item(zarr_dataset_path))
        return data

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def generate_batch_item(self, zarr_dataset_path: str):
        ag_graph = nx.Graph()
        pe_graph = nx.Graph()
        root = zarr.open(zarr_dataset_path, mode="r")
        item = {}

        # update U feature, knn feature, id, label, ag and pe graph.
        item, ag_graph, pe_graph = self.load_graph_attrs(
            root=root,
            item=item,
            ag_graph=ag_graph,
            pe_graph=pe_graph,
        )
        
        # update the graph edges based on id array sequence.
        ag_graph, pe_graph = self.create_ag_pe_graph(
            root=root,
            item=item,
            ag_graph=ag_graph,
            pe_graph=pe_graph,
        )
        ag_adj_unnormalized_matrix = nx.adjacency_matrix(ag_graph)
        pe_adj_unnormalized_matrix = nx.adjacency_matrix(pe_graph)
        ag_adj_normalized_matrix = self._normalize_adjacency_matrix(ag_adj_unnormalized_matrix)
        pe_adj_normalized_matrix = self._normalize_adjacency_matrix(pe_adj_unnormalized_matrix)
        ag_mask_matrix = self._get_mask_matrix(ag_adj_unnormalized_matrix)
        pe_mask_matrix = self._get_mask_matrix(pe_adj_unnormalized_matrix)
        item["ag_mask_matrix"] = ag_mask_matrix.toarray().astype(np.single)
        item["ag_adj_matrix"] = ag_adj_normalized_matrix.toarray().astype(np.single)
        item["pe_mask_matrix"] = pe_mask_matrix.toarray().astype(np.single)
        item["pe_adj_matrix"] = pe_adj_normalized_matrix.toarray().astype(np.single)
        item = self.create_knn_graph(item)
        return item

    def create_knn_graph(self, item):
        """Updates the k nearest neighbors for each contig in dictionary.
        Includes the neightbors, weights for each neighbor and knn graph.

        Args:
            item (dictionary): list format dataset.

        Returns:
            data_list (list): list format dataset.
        """
        knn_graph = nx.Graph()
        neighbors_attrs = []
        feature_array = item["knn_feature"]
        id_array = item["id"]
        for i in trange(feature_array.shape[0], desc="Creating KNN graph......."):
            id = int(id_array[i])
            tar_feature = np.expand_dims(feature_array[i], axis=0)
            dist_array = np.power((feature_array - tar_feature), 2)
            dist_sum_array = np.sum(dist_array, axis=1).reshape((feature_array.shape[0], 1))
            pairs = np.concatenate((dist_sum_array, id_array), axis=1)
            sorted_pairs = pairs[pairs[:, 0].argsort()]
            top_k_pairs = sorted_pairs[1: self.k+1]
            k_nearest_dis_array = self.Gaussian.cal_coefficient(sorted_pairs[:, ][0])
            normalized_k_dis_array = softmax(k_nearest_dis_array)
            neighbors_array = top_k_pairs[:, 1]
            k_dis_array = top_k_pairs[:, 0]
            neighbors_attrs.append(neighbors_array)
            for j in range(self.k):
                neighbor_index = get_index_by_id_from_array(neighbors_array[j], item["id"])
                neighbor_id_int = item["id"][neighbor_index][0]
                knn_graph.add_edge(id, neighbor_id_int)
        self._describe_graph("knn_grpah", knn_graph)
        knn_adj_matrix = nx.adjacency_matrix(knn_graph).toarray()
        normalized_knn_adj_matrix = self._normalize_adjacency_matrix(knn_adj_matrix)
        item["knn_adj_matrix"] = normalized_knn_adj_matrix.astype(np.single)
        item["neighbors"] = np.array(neighbors_attrs)
        return item

    def load_U_matrix(self):
        U = torch.load(self.U_feature_path)
        U = U.detach().numpy()
        return U

    def load_graph_attrs(self, root, item, ag_graph, pe_graph):
        """Load graph attrs from dataset, includes knn_feature, ag_pe_feature,
        contig_id tensor, contig_label tensor.

        Args:
            root (zarr.Group): root of processed zarr dataset.
            item (dict): dictionary to store the attributes.
            ag_graph (nx.Graph): ag graph, add node by contig id sequence.
            pe_graph (nx.Graph): pe graph, add node by contig id sequence.

        Returns:
            item (dict): updated dictionary to store the attributes.
            ag_graph (nx.Graph): ag graph with id sequence nodes.
            pe_graph (nx.Graph): pe graph with id sequence nodes.
        """
        contig_id_list = root.attrs["contig_id_list"]
        graph_attrs = []
        id_attrs = []
        label_attrs = []
        for i in tqdm(contig_id_list, desc="Loading graph attributes......."):
            # graph feature attrs
            tnf = np.array(root[i]["tnf_feat"])
            normalized_tnf = self.zscore(tnf, axis=0)
            rpkm = np.array(root[i]["rpkm_feat"])
            feature = np.concatenate((normalized_tnf, rpkm), axis=0)
            graph_attrs.append(feature)

            # contig labels 
            labels = np.array(root[i]["labels"])
            label_attrs.append(labels)

            # contig ids
            contig_id = np.array(root[i]["id"])
            id_attrs.append(contig_id)
            contig_id_int = int(contig_id[0])

            # creation of ag, pe graph.
            ag_graph.add_node(contig_id_int)
            pe_graph.add_node(contig_id_int)

        item["knn_feature"] = np.array(graph_attrs)
        item["ag_pe_feature"] = self.load_U_matrix()
        item["id"] = np.array(id_attrs)
        item["labels"] = np.array(label_attrs)

        # add long contig id tensor to further perform cross entropy.
        long_contig_id_list = root.attrs["long_contig_id_list"]
        item["long_contig_id"] = np.array(long_contig_id_list)
        return item, ag_graph, pe_graph

    def create_ag_pe_graph(self, root, item, ag_graph, pe_graph):
        """Add edegs to ag and pe graph according to the new index.

        Args:
            root (zarr.Group): root of processed zarr dataset.
            item (dict): dictionary to store the attributes.
            ag_graph (nx.Graph): created ag graph.
            pe_graph (nx.Graph): created pe graph.
        
        Returns:
            ag_graph (nx.Graph): ag graph with edges.
            pe_graph (nx.Graph): pe graph with edges.
        """
        contig_id_list = root.attrs["contig_id_list"]
        for i in tqdm(contig_id_list, desc="Creating ag and pe graph......."):
            # get the edges id from proessed zarr group.
            ag_graph_edges = root[i]["ag_graph_edges"]
            pe_graph_edges = root[i]["pe_graph_edges"]
            contig_id = np.array(root[i]["id"])
            contig_id_int = int(contig_id[0])
            
            # update ag graph edges.
            for edge_pair in ag_graph_edges:
                neighbor_id = edge_pair[1]
                if neighbor_id in contig_id_list:
                    neighbor_index = get_index_by_id_from_array(neighbor_id, item["id"])
                    neighbor_id_int = item["id"][neighbor_index][0]
                    ag_graph.add_edge(contig_id_int, neighbor_id_int)

            # update pe graph edges.
            for edge_pair in pe_graph_edges:
                neighbor_id = edge_pair[1]
                if neighbor_id in contig_id_list:
                    neighbor_index = get_index_by_id_from_array(neighbor_id, item["id"])
                    neighbor_id_int = item["id"][neighbor_index][0]
                    pe_graph.add_edge(contig_id_int, neighbor_id_int)

        self._describe_graph("ag_graph", ag_graph)
        self._describe_graph("pe_graph", pe_graph)
        return ag_graph, pe_graph

    @staticmethod
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

    @staticmethod
    def _describe_graph(graph_type: str, graph: nx.Graph):
        graph_nodes = graph.number_of_nodes()
        graph_edges = graph.number_of_edges()
        print("{} graph details: nodes: {}; edges: {}".format(
            graph_type, graph_nodes, graph_edges
        ))

    @staticmethod
    def _normalize_adjacency_matrix(adj_matrix):
        """Normalize adjacency matrix using the formulation from GCN;
        A_norm = D(-1/2) @ A @ D(-1/2).
        
        Args:
            adj_matrix (scipy.csr_matrix): adjacency matrix from graph.

        Returns:
            adj_normalized_matrix (np.array single precision): normalized adjacency
                matrix could cause out of memory issue.
        """
        I = np.identity(adj_matrix.shape[0])
        adj_matrix_hat = adj_matrix + I # add self-loop to A
        D = np.diag(np.sum(adj_matrix_hat, axis=0))
        D_half_norm = fractional_matrix_power(D, -0.5)
        adj_normalized_matrix = D_half_norm.dot(adj_matrix_hat).dot(D_half_norm)
        adj_normalized_matrix = adj_normalized_matrix.astype(np.single)

        return adj_normalized_matrix

    @staticmethod
    def _get_mask_matrix(adj_matrix):
        """Get mask matrix from adjacency matrix;
        M_ij = {
            -inf if A_ij = 0;
            0 if A_ij != 0;
        }
        
        Args:
            adj_matrix (np.array): adjacency matrix from graph.

        Returns:
            mask_matrix (np.array): mask matrix.
        """
        mask_matrix = adj_matrix.copy()
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i][j] == 0:
                    mask_matrix[i][j] = np.NINF
                else:
                    mask_matrix[i][j] = 0
        return mask_matrix


class MatrixFactorizationDataset(Dataset):
    """Matrix Factorization Dataset object, loading ag and pe graph from 
    zarr dataset and return the row_id(user_id), column_id(item_id) and 
    value each item.

    Args:
        zarr_dataset_path (str): path of zarr dataset root.
    """
    def __init__(
        self,
        zarr_dataset_path: str = "",
        *args,
        **kwargs,
    ):
        self.zarr_dataset_path = zarr_dataset_path
        self.data = []
        self.node_num = 0
        self.load_dataset(self.zarr_dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_graph(self, zarr_dataset_path: str):
        ag_graph = nx.Graph()
        pe_graph = nx.Graph()
        root = zarr.open(zarr_dataset_path, mode="r")
        contig_id_list = root.attrs["contig_id_list"]
        id_attrs = []
        item = {}

        for i in tqdm(contig_id_list, desc="Loading graph attributes......"):
            # creation of id array.
            contig_id = np.array(root[i]["id"])
            id_attrs.append(contig_id)
            contig_id_int = int(contig_id[0])

            # creation of ag, pe graph.
            ag_graph.add_node(contig_id_int)
            pe_graph.add_node(contig_id_int)
        
        item["id"] = np.array(id_attrs)

        # update the graph edges based on id array sequence.
        for i in tqdm(contig_id_list, desc="Creating ag and pe graph......."):
            ag_graph_edges = root[i]["ag_graph_edges"]
            pe_graph_edges = root[i]["pe_graph_edges"]
            contig_id = np.array(root[i]["id"])
            contig_id_int = int(contig_id[0])
            
            # update ag graph edges.
            for edge_pair in ag_graph_edges:
                neighbor_id = edge_pair[1]
                if neighbor_id in contig_id_list:
                    neighbor_index = get_index_by_id_from_array(neighbor_id, item["id"])
                    neighbor_id_int = item["id"][neighbor_index][0]
                    ag_graph.add_edge(contig_id_int, neighbor_id_int)

            # update pe graph edges.
            for edge_pair in pe_graph_edges:
                neighbor_id = edge_pair[1]
                if neighbor_id in contig_id_list:
                    neighbor_index = get_index_by_id_from_array(neighbor_id, item["id"])
                    neighbor_id_int = item["id"][neighbor_index][0]
                    pe_graph.add_edge(contig_id_int, neighbor_id_int)

        ag_adj_matrix = nx.adjacency_matrix(ag_graph).toarray()
        pe_adj_matrix = nx.adjacency_matrix(pe_graph).toarray()
        return ag_adj_matrix, pe_adj_matrix

    def load_dataset(self, zarr_dataset_path: str):
        ag_adj_matrix, pe_adj_matrix = self.load_graph(zarr_dataset_path)
        assert ag_adj_matrix.shape == pe_adj_matrix.shape
        N = ag_adj_matrix.shape[0]
        self.node_num = N
        for i in trange(N, desc="Create ag and pe graph as user-score dataset....."):
            for j in range(N):
                item = {}
                ag_value = ag_adj_matrix[i][j]
                pe_value = pe_adj_matrix[i][j]
                item["user_id"] = i
                item["item_id"] = j
                item["ag_value"] = ag_value
                item["pe_value"] = pe_value
                self.data.append(item)


class GMGATTransformerDataset(Dataset):
    """GMGAT Transformer Dataset object, which includes the graph embedding 
    compared to Graph Dataset and can be treated as pure transformer.

    """
    def __init__(
        self,
        zarr_dataset_path: str = "",
    ):
        #TODO: add the graph embedding module.
        #TODO: add the concatenation codes.
        pass


class MultiGraphDataset(Dataset):
    """Graph dataset object, loading dataset in a batch-wise manner.

    Args:
        zarr_dataset_path (string): path of zarr dataset root.
        k (int): k nearest neighbor used in neighbors.
        sigma (float): Gaussian variance when computing neighbors coefficient.
        use_neighbor_feature (boolean): whether to use the reconstructing 
            neighbors strategy.

    Each contig is stored into dictionary, with following keys:
        - feature: concated tnf and rpkm feature dim (104, 1);

    """
    # TODO: add feature per batch as comments.
    def __init__(
        self,
        zarr_dataset_path: str = "",
        k: int = 5,
        sigma: int = 1,
        use_neighbor_feature=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.zarr_dataset_path = zarr_dataset_path
        self.k = k
        self.use_neighbor_feature = use_neighbor_feature
        self.Gaussian = Gaussian(sigma=sigma)
        self.data = []

        self.data = self.load_dataset(zarr_dataset_path)

    def load_dataset(self, zarr_dataset_path):
        data_list = self._load_graph_attrs(zarr_dataset_path)
        if self.use_neighbor_feature:
            data_list = self.create_knn_graph(data_list)
        return data_list

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _load_graph_attrs(self, zarr_dataset_path: str):
        root = zarr.open(zarr_dataset_path, mode="r")
        contig_id_list = root.attrs["contig_id_list"]
        data_list = []
        for i in tqdm(contig_id_list):
            item = {}
            tnf = np.array(root[i]["tnf_feat"])
            normalized_tnf = self.zscore(tnf, axis=0)
            rpkm = np.array(root[i]["rpkm_feat"])
            feature = np.concatenate((normalized_tnf, rpkm), axis=0)
            labels = np.array(root[i]["labels"])
            contig_id = np.array(root[i]["id"])
            item["feature"] = feature
            item["labels"] = labels
            item["id"] = contig_id
            data_list.append(item)
        return data_list

    def create_knn_graph(self, data_list):
        """Updates the k nearest neighbors for each contig in dictionary. 
        
        Alerts: knn graph is created by id vector, stores the neightbors 
        and weights for each neighbor.

        Args:
            data_list (list): list format dataset.

        Returns:
            data_list (list): list format dataset.
        """
        id_list = []
        feature_list = []
        for i in range(len(data_list)):
            feature_list.append(data_list[i]["feature"])
            id_list.append(data_list[i]["id"])
        
        feature_array = np.array(feature_list)
        id_array = np.array(id_list)
        # TODO: remove feature_array.shape[0] by using N.
        for i in trange(feature_array.shape[0], desc="Creating KNN graph......."):
            tar_feature = np.expand_dims(feature_array[i], axis=0)
            dist_array = np.power((feature_array - tar_feature), 2)
            dist_sum_array = np.sum(dist_array, axis=1).reshape((feature_array.shape[0], 1))
            pairs = np.concatenate((dist_sum_array, id_array), axis=1) # track the id.
            sorted_pairs = pairs[pairs[:, 0].argsort()]
            top_k_pairs = sorted_pairs[1: self.k+1]
            k_nearest_dis_array = self.Gaussian.cal_coefficient(top_k_pairs[:, 0])
            normalized_k_dis_array = softmax(k_nearest_dis_array)
            neighbors_array = top_k_pairs[:, 1]
            data_list[i]["neighbors"] = neighbors_array
            data_list[i]["weights"] = normalized_k_dis_array
            if self.use_neighbor_feature:
                neighbor_feature = []
                for index in range(self.k):
                    neighbor_id = int(neighbors_array[index])
                    # search the feature by the id array tensor.
                    for j in range(feature_array.shape[0]):
                        if int(id_array[j]) == neighbor_id:
                            neighbor_feature.append(feature_array[j])
                neighbor_feature = np.stack(neighbor_feature, axis=0)
                data_list[i]["neighbors_feature"] = neighbor_feature

        return data_list

    @staticmethod
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


class AgFeatGraphDataset(Dataset):
    """Graph dataset object, loading dataset in a batch-wise manner.

    Args:
        zarr_dataset_path (string): path of zarr dataset root.
        k (int): k nearest neighbor used in neighbors.
        sigma (float): Gaussian variance when computing neighbors coefficient.
        use_neighbor_feature (boolean): whether to use the reconstructing 
            neighbors strategy.

    Each contig is stored into dictionary, with following keys:
        - feature: concated tnf and rpkm feature dim (104, 1);

    """
    def __init__(
        self,
        zarr_dataset_path: str = "/home/eddie/cami1-low-large.zarr",
        ag_feature_path: str = "/home/eddie/cami1-low/ag-feat.npy",
        id_mask_path: str = "/home/eddie/cami1-low/ag-mask.npy",
        k: int = 5,
        sigma: int = 1,
        use_neighbor_feature: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.zarr_dataset_path = zarr_dataset_path
        self.ag_feature_path = ag_feature_path
        self.id_mask_path = id_mask_path
        self.k = k
        self.use_neighbor_feature = use_neighbor_feature
        self.Gaussian = Gaussian(sigma=sigma)
        self.data = []

        self.data = self.load_dataset(zarr_dataset_path)

    def load_dataset(self, zarr_dataset_path):
        data_list = self._load_graph_attrs(zarr_dataset_path)
        if self.use_neighbor_feature:
            data_list = self.create_knn_graph(data_list)
            data_list = self.create_ag_based_knn_graph(data_list) 
        return data_list

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _load_graph_attrs(self, zarr_dataset_path: str):
        root = zarr.open(zarr_dataset_path, mode="r")
        contig_id_list = root.attrs["contig_id_list"]
        id_mask = np.load(self.id_mask_path)
        id_mask_list = [item for item in id_mask]
        data_list = []
        for i, c in enumerate(tqdm(contig_id_list)):
            item = {}
            tnf = np.array(root[c]["tnf_feat"])
            normalized_tnf = self.zscore(tnf, axis=0)
            rpkm = np.array(root[c]["rpkm_feat"])
            feature = np.concatenate((normalized_tnf, rpkm), axis=0)
            labels = np.array(root[c]["labels"])
            contig_id = np.array(root[c]["id"])
            item["feature"] = feature
            item["labels"] = labels
            item["id"] = contig_id
            item["id_mask"] = id_mask_list[i]
            data_list.append(item)
        return data_list

    def create_ag_based_knn_graph(self, data_list):
        root = zarr.open(self.zarr_dataset_path, "r")
        non_dead_ends_id_list = root.attrs["non_dead_ends_contig_id_list"]
        feature_list = []
        id_list = []
        for i in range(len(data_list)):
            feature_list.append(data_list[i]["feature"])
            id_list.append(data_list[i]["id"])
        original_feature_array = np.array(feature_list)
        original_id_array = np.array(id_list)
        pca_feature_array = np.load(self.ag_feature_path)
        non_dead_ends_id_array = np.array(non_dead_ends_id_list).reshape(-1, 1)
        
        # Non-dead-ends neighbors feature.
        for i in trange(len(non_dead_ends_id_list), desc="Creating AG based KNN graph"):
            tar_feat = np.expand_dims(pca_feature_array[i], axis=0)
            dist_array = np.power((pca_feature_array - tar_feat), 2)
            dist_sum_array = np.sum(dist_array, axis=1).reshape((pca_feature_array.shape[0], 1))
            pairs = np.concatenate((dist_sum_array, non_dead_ends_id_array), axis=1) # track the id.
            sorted_pairs = pairs[pairs[:, 0].argsort()]
            top_k_pairs = sorted_pairs[1: self.k+1]
            k_nearest_dis_array = self.Gaussian.cal_coefficient(top_k_pairs[:, 0])
            normalized_k_dis_array = softmax(k_nearest_dis_array)
            neighbors_array = top_k_pairs[:, 1]

            # data list i is not paired with the outer looping i.
            origin_index = get_index_by_id_from_array(non_dead_ends_id_list[i], original_id_array)
            #data_list[origin_index]["ag_based_knn_neighbors"] = neighbors_array
            data_list[origin_index]["ag_based_knn_weights"] = normalized_k_dis_array
            if self.use_neighbor_feature:
                neighbor_feature = []
                for index in range(self.k):
                    neighbor_id = int(neighbors_array[index])
                    # search the feature by the id array tensor.
                    for j in range(original_feature_array.shape[0]):
                        if int(original_id_array[j]) == neighbor_id:
                            neighbor_feature.append(original_feature_array[j])
                neighbor_feature = np.stack(neighbor_feature, axis=0)
                data_list[origin_index]["ag_based_neighbors_feature"] = neighbor_feature
        
        # Dead-ends neighbors feature.
        for item in data_list:
            if item["id_mask"] == 0:
                neighbor_feature_list = []
                for index in range(self.k):
                    neighbor_feature_list.append(item["feature"])
                neighbor_feature = np.stack(neighbor_feature_list)
                item["ag_based_neighbors_feature"] = neighbor_feature
                item["ag_based_knn_weights"] = np.array([1/self.k for i in range(self.k)])
        
        return data_list

    def create_knn_graph(self, data_list):
        """Updates the k nearest neighbors for each contig in the dictionary. 
        
        Alerts: knn graph is created by id vector, stores the neightbors 
        and weights for each neighbor.

        Args:
            data_list (list): list format dataset.

        Returns:
            data_list (list): list format dataset.
        """
        id_list = []
        feature_list = []
        for i in range(len(data_list)):
            feature_list.append(data_list[i]["feature"])
            id_list.append(data_list[i]["id"])
        
        feature_array = np.array(feature_list)
        id_array = np.array(id_list)
        # TODO: remove feature_array.shape[0] by using N.
        for i in trange(feature_array.shape[0], desc="Creating KNN graph......."):
            tar_feature = np.expand_dims(feature_array[i], axis=0)
            dist_array = np.power((feature_array - tar_feature), 2)
            dist_sum_array = np.sum(dist_array, axis=1).reshape((feature_array.shape[0], 1))
            pairs = np.concatenate((dist_sum_array, id_array), axis=1) # track the id.
            sorted_pairs = pairs[pairs[:, 0].argsort()]
            top_k_pairs = sorted_pairs[1: self.k+1]
            k_nearest_dis_array = self.Gaussian.cal_coefficient(top_k_pairs[:, 0])
            normalized_k_dis_array = softmax(k_nearest_dis_array)
            neighbors_array = top_k_pairs[:, 1]
            data_list[i]["neighbors"] = neighbors_array
            data_list[i]["weights"] = normalized_k_dis_array
            if self.use_neighbor_feature:
                neighbor_feature = []
                for index in range(self.k):
                    neighbor_id = int(neighbors_array[index])
                    # search the feature by the id array tensor.
                    for j in range(feature_array.shape[0]):
                        if int(id_array[j]) == neighbor_id:
                            neighbor_feature.append(feature_array[j])
                neighbor_feature = np.stack(neighbor_feature, axis=0)
                data_list[i]["neighbors_feature"] = neighbor_feature

        return data_list

    @staticmethod
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
