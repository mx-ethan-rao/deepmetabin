
import argparse
import numpy as np
import os
import igraph
from collections import defaultdict
from igraph import plot, Graph
import time
from tqdm import tqdm, trange

COLOUR_DICT = [
    "#e6194b", 
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231", 
    "#911eb4", 
    "#46f0f0", 
    "#f032e6", 
    "#fabebe", 
    "#008080", 
    "#e6beff", 
    "#9a6324", 
    "#fffac8", 
    "#800000", 
    "#aaffc3", 
    "#808000", 
    "#000075", 
    "#ffffff", 
    "#ffd8b1", 
    "#000000", 
    "#808080", 
    "grey", 
    "purple", 
    "yellow", 
    "dark gray", 
    "dark orchid", 
    "light pink", 
    "light grey", 
    "light green", 
    "light sky blue", 
    "navy", 
    "orange", 
    "orchid", 
    "silver", 
    "snow", 
    "sea green", 
    "wheat", 
    "peru", 
    "medium blue", 
    "lime", 
    "magenta", 
    "indigo", 
    "ivory", 
    "honeydew", 
    "green yellow", 
    "fuchsia", 
    "gold", 
    "dark orange", 
    "dark green", 
    "dark red", 
    "dark magenta", 
    "dark salmon", 
    "dark khaki", 
    "cyan", 
    "coral", 
    "azure", 
    "aqua", 
    "aliceblue", 
    "light salmon", 
    "light cyan", 
    "light steel blue", 
    "lightskyblue", 
    "lightyellow",
    "#e6194b", 
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231", 
    "#911eb4", 
    "#46f0f0", 
    "#f032e6", 
    "#fabebe", 
    "#008080", 
    "#e6beff", 
    "#9a6324", 
    "#fffac8", 
    "#800000", 
    "#aaffc3", 
    "#808000", 
    "#000075", 
    "#ffffff", 
    "#ffd8b1", 
    "#000000", 
    "#808080", 
    "grey", 
    "purple", 
    "yellow", 
    "dark gray", 
    "dark orchid", 
    "light pink", 
    "light grey", 
    "light green", 
    "light sky blue", 
    "navy", 
    "orange", 
    "orchid", 
    "silver", 
    "snow", 
    "sea green", 
    "wheat", 
    "peru", 
    "medium blue", 
    "lime", 
    "magenta", 
    "indigo", 
    "ivory", 
    "honeydew", 
    "green yellow", 
    "fuchsia", 
    "gold", 
    "dark orange", 
    "dark green", 
    "dark red", 
    "dark magenta", 
    "dark salmon", 
    "dark khaki", 
    "cyan", 
    "coral", 
    "azure", 
    "aqua", 
    "aliceblue", 
    "light salmon", 
    "light cyan", 
    "light steel blue", 
    "lightskyblue", 
    "lightyellow",
    "#e6194b", 
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231", 
    "#911eb4", 
    "#46f0f0", 
    "#f032e6", 
    "#fabebe", 
    "#008080", 
    "#e6beff", 
    "#9a6324", 
    "#fffac8", 
    "#800000", 
    "#aaffc3", 
    "#808000", 
    "#000075", 
    "#ffffff", 
    "#ffd8b1", 
    "#000000", 
    "#808080", 
    "grey", 
    "purple", 
    "yellow", 
    "dark gray", 
    "dark orchid", 
    "light pink", 
    "light grey", 
    "light green", 
    "light sky blue", 
    "navy", 
    "orange", 
    "orchid", 
    "silver", 
    "snow", 
    "sea green", 
    "wheat", 
    "peru", 
    "medium blue", 
    "lime", 
    "magenta", 
    "indigo", 
    "ivory", 
    "honeydew", 
    "green yellow", 
    "fuchsia", 
    "gold", 
    "dark orange", 
    "dark green", 
    "dark red", 
    "dark magenta", 
    "dark salmon", 
    "dark khaki", 
    "cyan", 
    "coral", 
    "azure", 
    "aqua", 
    "aliceblue", 
    "light salmon", 
    "light cyan", 
    "light steel blue", 
    "lightskyblue", 
    "lightyellow",
]




def visualize_graph(bin_list, num_contigs, save_path):
    """Visualize the graph from model results.
    Args:
        bin_list (list): cluster list including of 
        contig_list (list): list of contig id to log.
        save_path (str): plotting path of graph.
    """
    graph = igraph.Graph()
    graph_size = num_contigs
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--vamb_result', type=str, default='/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/vamb_2000/vamb_out/clusters.tsv', 
                        help='Checkm result for vamb')
    parser.add_argument('--vamb_embedding', type=str, default='/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/vamb_2000/vamb_out/latent.npz', 
                        help='Checkm result for vamb')
    parser.add_argument('--vamb_conigname', type=str, default='/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/vamb_2000/vamb_out/contignames.npz', 
                        help='Checkm result for vamb')
    parser.add_argument('--deepmetabin_result', type=str, default='/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/cami1-csv-metadecoder-refine/latent_epoch_600.0_result.csv', 
                        help='Checkm result for deepmetabin')
    parser.add_argument('--deepmetabin_embedding', type=str, default='/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/cami1-csv-metadecoder-refine/latent_epoch_600.0.npy', 
                        help='Checkm result for deepmetabin')
    parser.add_argument('--deepmetabin_conigname', type=str, default='/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/cami1-csv-metadecoder-refine/id.npy', 
                        help='Checkm result for deepmetabin')

    parser.add_argument('--num_contigs', type=int, default=10, 
                        help='Top number of contigs to compare')
    args = parser.parse_args()

    # handle for vamb
    vamb_res = {}
    vamb_contignames = np.load(args.vamb_conigname)
    vamb_contignames = vamb_contignames['arr_0']
    valid_indice = []
    valid_contignames = []


    for idx, contigname in enumerate(vamb_contignames):
        if int(contigname.split('_')[1]) <= args.num_contigs:
            valid_indice.append(idx)

            valid_contignames.append(contigname.split('_')[0].strip() + '_' + contigname.split('_')[1].strip())

    vamb_embeddings = np.load(args.vamb_embedding)
    vamb_embeddings = vamb_embeddings['arr_0'][valid_indice]

    for embedding, contigname in zip(vamb_embeddings, valid_contignames):
        vamb_res[contigname] = embedding

    vamb_result = defaultdict(list)
    with open(args.vamb_result, 'r') as f:
        for l in f.readlines():
            items = l.split()
            if len(items) == 3:
                continue
            temp = items[1].split('_')
            if int(temp[1]) <= args.num_contigs:
                vamb_result[int(items[0])].append({"contig_id": temp[0].strip() + '_' + temp[1].strip(), "contig_embedding": vamb_res[temp[0].strip() + '_' + temp[1].strip()]})
    vamb_result = dict(vamb_result)

    # handle for deepmetabin
    deepmetabin_res = {}
    deepmetabin_contignames = np.load(args.deepmetabin_conigname)
    deepmetabin_embeddings = np.load(args.deepmetabin_embedding)
    valid_indice = []
    valid_contignames = []


    for idx, contigname in enumerate(deepmetabin_contignames):
        if contigname <= args.num_contigs:
            valid_indice.append(idx)
            valid_contignames.append('NODE_' + str(int(contigname)))


    for embedding, contigname in zip(deepmetabin_embeddings, valid_contignames):
        deepmetabin_res[contigname] = embedding
    
    deepmetabin_result = defaultdict(list)
    with open(args.deepmetabin_result) as f:
        for l in f.readlines():
            items = l.split()
            temp = items[0]
            items[0] = items[1]
            items[1] = temp
            if len(items) == 3:
                continue
            if int(items[0].split('_')[1].strip()) <= args.num_contigs:
                deepmetabin_result[int(items[1])].append({"contig_id": items[0], "contig_embedding": deepmetabin_res[items[0]]})
    deepmetabin_result = dict(deepmetabin_result)

    visualize_graph(vamb_result.values(), args.num_contigs, '/tmp/local/zmzhang/DeepMetaBin/mingxing/work_with_wc/vamb.png')
    visualize_graph(deepmetabin_result.values(), args.num_contigs, '/tmp/local/zmzhang/DeepMetaBin/mingxing/work_with_wc/deepmetabin_2000.png')



    


