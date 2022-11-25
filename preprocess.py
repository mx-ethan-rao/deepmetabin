import numpy as np
import zarr
from tqdm import trange
from absl import app, flags
from src.utils.util import (
    summary_bin_list_from_csv,
    load_graph,
    describe_dataset,
)
# import os


def create_contigs_zarr_dataset(
        output_zarr_path: str,
        contigname_path: str,
        labels_path: str,
        tnf_feature_path: str,
        rpkm_feature_path: str,
        # ag_graph_path: str,
        # pe_graph_path: str,
        filter_threshold: int = 1000,
        # long_contig_threshold: int = 1000,
    ):
    """create long contigs zarr dataset based on contigname file and labels file as filters.

    Args:
        output_zarr_path (string): path to save the processed long contigs datasets.
        contigname_path (string): path of contigname file.
        labels_path (string): path of labels file.
        tnf_feature_path (string): path of tnf feature file.
        rpkm_feature_path (string): path of rpkm feature file.
        ag_graph_path (string): path of ag graph file.
        pe_graph_path (string): path of pe graph file.
        filter_threshold (int): threshold of filtering bp length default 1000.
        long_contig_threshold (int): threshold of long contig.

    Returns:
        None.

    Root zarr group stands for the collection of contigs:
        - contig_id_list (attrs -> list): contigs list after filtering.
        - long_contig_id_list (attrs -> list): long contigs list.
    Each zarr group stands for one unique contig, with six properties:
        - id (group -> np.ndarray (1)): contig id.
        - tnf_feat (group -> np.ndarray (103)): tnf feature.
        - rpkm_feat (group -> np.ndarray (1)): rpkm feature.
        - labels (group -> np.ndarray (1)): ground truth bin id.
        - ag_graph_edges (group -> list): relevant edge in ag graph.
        - pe_graph_edges (group -> list): relevant edge in pe graph.
    """
    root = zarr.open(output_zarr_path, mode="w")
    bin_list = summary_bin_list_from_csv(labels_path)
    num_cluster = len(bin_list)

    # ag_graph_contig_id_dict_list, ag_graph_edges_list = load_graph(ag_graph_path)
    # pe_graph_contig_id_dict_list, pe_graph_edges_list = load_graph(pe_graph_path)
    # load contig name file:
    contigname_file = np.load(contigname_path)
    contigname_attrs_name = contigname_file.files[0]
    contigname_attrs = contigname_file[contigname_attrs_name]
    # load tnf file:
    tnf_file = np.load(tnf_feature_path)
    tnf_attrs_name = tnf_file.files[0]
    tnf_attrs = tnf_file[tnf_attrs_name]
    # load rpkm file:
    rpkm_file = np.load(rpkm_feature_path)
    rpkm_attrs_name = rpkm_file.files[0]
    rpkm_attrs = rpkm_file[rpkm_attrs_name]

    contig_id_list = []
    tnf_list = []
    rpkm_list = []
    label_list = []
    # long_contig_id_list = []
    for i in trange(contigname_attrs.shape[0], desc="Preprocessing dataset......"):
        props = contigname_attrs[i].split("_")
        contig_id = int(props[1])
        contig_length = int(props[3])
        # if contig_length >= long_contig_threshold:
        #     long_contig_id_list.append(i)
        if contig_length >= filter_threshold:
            have_label = False
            for j in range(num_cluster):
                if contig_id in bin_list[j]:
                    labels = j
                    have_label = True
            if have_label is False:
                labels = -1 # Use -1 to hold unlabeled contig (node)

            contig_id_list.append(contig_id)
            tnf_list.append(list(tnf_attrs[i]))
            rpkm_list.append(list(rpkm_attrs[i]))
            label_list.append(labels)
            # root.create_group(contig_id)
            # root[contig_id]["id"] = np.array([contig_id])
            # root[contig_id]["tnf_feat"] = tnf_attrs[i]
            # root[contig_id]["rpkm_feat"] = rpkm_attrs[i]
            # root[contig_id]["labels"] = labels
            # # add ag graph edges.
            # for index, id in enumerate(ag_graph_contig_id_dict_list):
            #     if contig_id == id:
            #         root[contig_id]["ag_graph_edges"] = ag_graph_edges_list[index]
            #         break
            # # add pe graph edges.
            # for index, id in enumerate(pe_graph_contig_id_dict_list):
            #     if contig_id == id:
            #         root[contig_id]["pe_graph_edges"] = pe_graph_edges_list[index]
            #         break

    root.attrs["contig_id_list"] = contig_id_list
    root.attrs["tnf_list"] = tnf_list
    root.attrs["rpkm_list"] = rpkm_list
    root.attrs["label_list"] = label_list

    # root.attrs["long_contig_id_list"] = long_contig_id_list


class PreprocessManager:
    def __init__(
        self,
        output_zarr_path: str,
        contigname_path: str,
        labels_path: str,
        tnf_feature_path: str,
        rpkm_feature_path: str,
        # ag_graph_path: str,
        # pe_graph_path: str,
        filter_threshold: int = 1000,
        # long_contig_threshold: int = 1000,
        **kwargs,
    ):
        self.output_zarr_path = output_zarr_path
        self.contigname_path = contigname_path
        self.labels_path = labels_path
        self.tnf_feature_path = tnf_feature_path
        self.rpkm_feature_path = rpkm_feature_path
        # self.ag_graph_path = ag_graph_path
        # self.pe_graph_path = pe_graph_path
        self.filter_threshold = filter_threshold
        # self.long_contig_threshold = long_contig_threshold

    def preprocess(self):
        # original_contignames = np.load(self.contigname_path)['arr_0']
        # contignames = [ f'NODE_{i + 1}_' + '_'.join(str(contigname).split('_')[2:]) for i, contigname in enumerate(original_contignames)]
        # contignames = np.array(contignames)
        # np.savez('tmp_contignames.npz', contignames)

        # contig_dict = dict(zip(original_contignames.tolist(), contignames))
        # with open(self.labels_path, 'r') as f1, open('tmp_labels.csv', 'w') as f2:
        #     for line in f1.readlines():
        #         item = line.split(',')
        #         f2.write('{},{}\n'.format(contig_dict[item[0].strip()], item[1].strip()))

        create_contigs_zarr_dataset(
            output_zarr_path=self.output_zarr_path,
            contigname_path=self.contigname_path,
            labels_path=self.labels_path,
            tnf_feature_path=self.tnf_feature_path,
            rpkm_feature_path=self.rpkm_feature_path,
            # ag_graph_path=self.ag_graph_path,
            # pe_graph_path=self.pe_graph_path,
            filter_threshold=self.filter_threshold,
            # long_contig_threshold=self.long_contig_threshold,
        )
        describe_dataset(processed_zarr_dataset_path=self.output_zarr_path)
        # os.remove('tmp_contignames.npz')
        # os.remove('tmp_labels.csv')



def main(argv=None):
    preprocess_manager = PreprocessManager(**FLAGS.flag_values_dict())
    preprocess_manager.preprocess()


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    # Details: Pls check function create_contigs_zarr_dataset arguments.
    flags.DEFINE_string("output_zarr_path", "", "")
    flags.DEFINE_string("contigname_path", "", "")
    flags.DEFINE_string("labels_path", "", "")
    flags.DEFINE_string("tnf_feature_path", "", "")
    flags.DEFINE_string("rpkm_feature_path", "", "")
    # flags.DEFINE_string("ag_graph_path", "", "")
    # flags.DEFINE_string("pe_graph_path", "", "")
    flags.DEFINE_integer("filter_threshold", 1000, "")
    # flags.DEFINE_integer("long_contig_threshold", 1000, "")
    app.run(main)
