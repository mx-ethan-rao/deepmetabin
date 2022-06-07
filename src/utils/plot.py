from igraph import plot, Graph
from tqdm import tqdm


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
    "lightyellow"
]

def update_graph_labels(graph, node_list, bin_list):
    """Update labels(bin_id) to graph and add colours list to 
    graph object.
    
    Args:
        graph (igraph.graph): graph object.
        node_list (list): node id list.
        bin_list (2D list): 2 dimension list, 1d stands for bin list;
            2d stores node list per bin.

    Returns:
        image (np.ndarray): image to logging.
    """
    node_colour_list = []
    bin_num = len(bin_list)
    assert bin_num <= len(COLOUR_DICT)
    for node in tqdm(node_list, desc="Colouring graph..."):
        have_label = False # indicator if node has label.
        for j in range(bin_num):
            if node in bin_list[j]:
                node_colour_list.append(COLOUR_DICT[j])
                have_label = True
        
        if have_label is False:
            node_colour_list.append("white")
    graph.vs["color"] = node_colour_list
