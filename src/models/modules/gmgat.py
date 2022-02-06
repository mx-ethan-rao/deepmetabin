import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np


class AttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        use_bias=True,
    ):
        """
        to be added.
        """
        super().__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.Q = nn.Linear(input_dim, output_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(input_dim, output_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(input_dim, output_dim * num_heads, bias=use_bias)

    def propagate_graph_attention(self, g):
        """
        Args:
            g (dgl graph object): graph g to compute attention.
        """
        g.apply_edges('K_h', 'Q_h', 'score')
        g.apply_edges('score', np.sqrt(self.output_dim))

        edges = g.edges()
        g.send_and_recv(edges, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))


    def forward(self, g, h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        g.ndata['Q_h'] = Q_h.view(-1, self.num_haeds, self.output_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.output_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.output_dim)

        self.propagate_graph_attention(g)
        output = g.ndata['wV']

        return output



class GraphTransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        use_bias=True,
    ):
        # TODO: could further test up the resdual strategy here.
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.attention = AttentionBlock(input_dim, output_dim//num_heads, num_heads, use_bias)

        # optional
        self.projection_layer = nn.Linear(output_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, g, h):
        attention_output = self.attention(g, h).view(-1, self.output_dim)
        attention_output = F.relu(attention)
        attention_output = self.batch_norm(attention_output)
        return attention_output
