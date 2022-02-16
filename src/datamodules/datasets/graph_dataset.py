import torch
import scipy
import scipy.sparse as sp
import networkx as nx
import numpy as np
from numpy import load
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    """Graph dataset object, loading graph object and iterate 
    graph attributes for training.

    Args:
        graph_dataset_roots (list): list of the graph datasets path.
        graph_attrs_dataset_roots (list): list of the graph attrs 
            datasets path (bam file).
    """
    def __init__(
        self,
        graph_dataset_roots=[],
        graph_attrs_dataset_roots=[],
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.graph_dataset_roots = graph_dataset_roots
        self.graph_attrs_dataset_roots = graph_attrs_dataset_roots
        self.data = []
        assert len(self.graph_dataset_roots) == len(self.graph_attrs_dataset_roots)

        for graph_path, graph_attrs_path in zip(
            self.graph_dataset_roots, self.graph_attrs_dataset_roots
        ):
            self.data.append(self.load_dataset(graph_path, graph_attrs_path))

    def load_dataset(self, graph_path, graph_attrs_path):
        graph = self._load_graph(graph_path)
        self._describe_grpah(graph)
        self.generate_batch_item(graph, graph_attrs_path)
        
    def generate_batch_item(self, graph, graph_attrs):
        adj_unnormalized_matrix = nx.adjacency_matrix(graph)
        adj_normalized_matrix = self._normalize_adjacency_matrix(adj_unnormalized_matrix)
        mask_matrix = self._get_mask_matrix(adj_normalized_matrix)
        graph_attrs = self._load_graph_attrs(graph_attrs)
        
        # define and update data per item
        item = {}
        item["adj_matrix"] = adj_normalized_matrix
        item["mask_matrix"] = mask_matrix
        item["graph_attrs"] = graph_attrs
        return item

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _load_graph(graph_path: str):
        """Load graph from file.
        
        Args:
            graph_path (str): path of the graph file.
        
        Returns:
            graph (nx.Graph): networkx Graph object.
        """
        graph = nx.Graph()
        with open(graph_path, "r") as graph_file:
            line = graph_file.readline()
            while line != "":
                line = line.strip()
                strings = line[:-1].split()
                if line[-1] == ":":
                    contig = strings[0]
                    graph.add_node(contig)
                elif line[-1] == ";":
                    graph.add_edge(contig, strings[0])
                line = graph_file.readline()
        return graph

    @staticmethod
    def _load_graph_attrs(graph_attrs_path: str):
        attrs_file = load(graph_attrs_path)
        attrs_name = attrs_file.files[0] # keys 
        attrs = attrs_file[attrs_name]
        return attrs

    @staticmethod
    def _describe_grpah(graph: nx.Graph):
        graph_nodes = graph.number_of_nodes()
        graph_edges = graph.number_of_edges()
        print("graph details: nodes: {}; edges: {}".format(
            graph_nodes, graph_edges
        ))

    @staticmethod
    def _normalize_adjacency_matrix(adj_matrix):
        """Normalize adjacency matrix using the formulation from GCN;
        A_norm = D(-1/2) @ A @ D(-1/2).
        
        Args:
            adj_matrix (scipy.csr_matrix): adjacency matrix from graph.

        Returns:
            adj_normalized_matrix (np.array half precision): normalized adjacency
                matrix could cause out of memory issue.
        """
        diags = adj_matrix.sum(axis=1).A1 # degree vector.
        with scipy.errstate(divide="ignore"):
            sqrt = 1.0 / np.sqrt(diags)
        sqrt[np.isinf(sqrt)] = 0 # remove infinte value as zero.
        D = sp.diags(sqrt, format='csr')
        adj_normalized_matrix = D @ adj_matrix @ D
        adj_normalized_matrix = self.csr2tensor(adj_normalized_matrix) # to sparse tensor.
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

    @staticmethod
    def csr2tensor(matrix):
        """ Transform csr format sparse matrix to torch.tensor.
        To save space use torch.half to save the values of sparse
        matrix.

        Args:
            matrix (csr): input csr matrix.

        Returns:
            sprase_matrix (tensor): matrix torch sparse format tensor.
        """
        coo = matrix.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.HalfTensor(values)
        shape = coo.shape

        sparse_matrix = torch.sparse.HalfTensor(i, v, torch.Size(shape))
        return sparse_matrix

    @staticmethod
    def to_sparse(x):
        """Transform dense tensor to sparse tensor.
        
        Args:
            x (tensor): dense tensor.

        Returns:
            x_sparse (sparse.tensor): sparse tensor.
        """
        x_typename = torch.typename(x).split(".")[-1]
        sparse_tensortype = getattr(torch.sparse, x_typename)

        indices = torch.nonzero(x)
        if len(indices.shape) == 0:  # if all elements are zeros
            return sparse_tensortype(*x.shape)
        indices = indices.t()
        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        x_sparse = sparse_tensortype(indices, values, x.size())
        return x_sparse
