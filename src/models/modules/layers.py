import math
import torch
import torch.nn.init as init
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class GumbelSoftmax(nn.Module):

  def __init__(self, f_dim, c_dim):
    super(GumbelSoftmax, self).__init__()
    self.logits = nn.Linear(f_dim, c_dim)
    self.f_dim = f_dim
    self.c_dim = c_dim
     
  def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
      U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self, logits, temperature):
    y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
    return F.softmax(y / temperature, dim=-1)

  def gumbel_softmax(self, logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    #categorical_dim = 10
    y = self.gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard 
  
  def forward(self, x, temperature=1.0, hard=False):
    logits = self.logits(x).view(-1, self.c_dim)
    prob = F.softmax(logits, dim=-1)
    y = self.gumbel_softmax(logits, temperature, hard)
    return logits, prob, y


class Gaussian(nn.Module):
  def __init__(self, in_dim, z_dim):
    super(Gaussian, self).__init__()
    self.mu = nn.Linear(in_dim, z_dim)
    self.var = nn.Linear(in_dim, z_dim)

  def reparameterize(self, mu, var):
    std = torch.sqrt(var + 1e-10)
    noise = torch.randn_like(std)
    z = mu + noise * std
    return z      

  def forward(self, x):
    mu = self.mu(x)
    var = F.softplus(self.var(x))
    z = self.reparameterize(mu, var)
    return mu, var, z


class GraphAttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        use_bias=True,
    ):
        """ Attention block performed on graph.
        
        Args:
            input_dim (int): input dimension of embedding.
            output_dim (int): output dimension of embedding.
            num_heads (int): number of heads.
            use_bias (boolean): whether to use bias in attention block.

        Attrs:
            Q (nn.Linear): Query matrix.
            K (nn.Linear): Keys matrix.
            V (nn.Linear): Values matrix.
        """
        super(GraphAttentionBlock, self).__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dim_per_head = output_dim // num_heads

        self.Q = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.K = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.V = nn.Linear(input_dim, output_dim, bias=use_bias)

    def propagate_graph_attention(self, Q_h, K_h, V_h, adj_matrix, mask_matrix):
        """ This function performs for the aggregation of graph network based on the
        output of single head Q_h, K_h, V_h, adjacent matrix, mask matrix.

        Args:
            Q_h (tensor): Querys, dimention is (B, N, D);
            K_h (tensor): Keys, dimention is (B, N, D);
            V_h (tensor): Values, dimention is (B, N, D);
            adj_matrix (tensor): normalized adjacent matirx, dimension is (B, N, N);
            mask_matrix (tensor): masked matrix, dimension is (B, N, N);

        Returns:
            h (tensor): calculation results of self-attention mechanisms.
        """
        K_h_transpose = torch.transpose(K_h, 1, 2) # dim B, D, N
        # score = (adj_matrix * torch.matmul(Q_h, K_h_transpose) + mask_matrix) / np.sqrt(self.dim_per_head)
        score = (adj_matrix * torch.matmul(Q_h, K_h_transpose)) / np.sqrt(self.dim_per_head)
        attention_score = nn.Softmax(dim=2)(score) # dim B, N, N
        attention_output = torch.matmul(attention_score, V_h) # dim B, N, D
        return attention_output

    def forward(self, h, adj_matrix, mask_matrix):
        """forward function of Attention block.

        Args:
            h (tensor): input of graph tensor, dimension is (B, N, D);
            adj_matrix (tensor): normalized adjacent matrix;
            mask_matrix (tensor): masked matrix;

        Returns:
            output (tensor): result from the multi-head attention block.
        """
        B, N, D = h.shape
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        Q_multi_head_h = Q_h.view(B, N, self.num_heads, self.dim_per_head) # dim B, N, H, D
        K_multi_head_h = K_h.view(B, N, self.num_heads, self.dim_per_head) # dim B, N, H, D
        V_multi_head_h = V_h.view(B, N, self.num_heads, self.dim_per_head) # dim B, N, H, D
        
        heads_output_list = []
        for i in range(self.num_heads):
            single_head_output = self.propagate_graph_attention(
                Q_h=Q_multi_head_h[:, :, i],
                K_h=K_multi_head_h[:, :, i],
                V_h=V_multi_head_h[:, :, i],
                adj_matrix=adj_matrix,
                mask_matrix=mask_matrix,
            ) # dim B, N, D
            heads_output_list.append(single_head_output)
        output = torch.cat(heads_output_list, dim=2) # dim B, N, D*H
        return output


class GraphTransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        use_bias=True,
    ):
        """ GraphTransformer block integrated attention.
        TODO: could further add resdual strategy in transformer block.

        Args:
            input_dim (int): input dimension of embedding.
            output_dim (int): output dimension of embedding.
            num_heads (int): number of heads.
            use_bias (boolean): whether to use bias in attention block.

        Attrs:
            attention (GraphAttentionBlock): compute the graph embedding using
                attention mechanism.
            batch_norm (nn.BatchNorm1d): normalization.
        """
        super(GraphTransformerBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.attention = GraphAttentionBlock(input_dim, output_dim, num_heads, use_bias)

        # optional
        self.projection_layer = nn.Linear(output_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, h, adj_matrix, mask_matrix):
        """forward function of transformer block, support one block of attention for now.
        
        Args:
            h (tensor): input of graph tensor, dimension is (B, N, D);
            adj_matrix (tensor): normalized adjacent matrix;
            mask_matrix (tensor): masked matrix;

        Returns:
            attention_out (tensor): result from transformer block.
        """
        attention_output = self.attention(h, adj_matrix, mask_matrix)
        attention_output = F.relu(attention_output)
        # attention_output = self.batch_norm(attention_output) # one graph per batch.
        return attention_output


class GraphConvolutionBlock(nn.Module):
    """GraphConvolution Block which performs graph convolution when encoding.

    Args:
        in_features (int): input feature dimension.
        out_features (int): output feature dimension.
        use_bias (boolean): whether use bias in when calculating.
    """
    def __init__(
        self,
        in_features,
        out_features,
        use_bias=True
    ):
        super(GraphConvolutionBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # torch mm only perform on 2 dimension tensor, squeeze firstly.
        input = torch.squeeze(input)
        adj = torch.squeeze(adj)
        output = torch.mm(input, self.weight)
        output = torch.spmm(adj, output)
        if self.bias is not None:
            output = output + self.bias
        else:
            pass
        output = F.relu(output)
        output = torch.unsqueeze(output, 0)
        return output
