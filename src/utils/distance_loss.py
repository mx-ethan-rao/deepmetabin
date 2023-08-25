import numpy as np
import torch


class KMeansClusteringLoss(torch.nn.Module):
    def __init__(self):
        super(KMeansClusteringLoss,self).__init__()

    def forward(self, encode_output, centroids, mask_matrix):
        return self.intra_class_distance(encode_output, centroids, mask_matrix) - self.inter_class_distance(centroids)

    def intra_class_distance(self, encode_output, centroids, mask_matrix):
        """Calculate the intra_class distance from known results.
        Args:
            encode_output (torch.float32): n x d matrix, where d is the hidden size
            centroids (torch.float32): k x d where d is the hiddent size
            mask_matrix (torch.bool): n x k matrix, plotting path of graph. 
                                    e.g. mask_matrix[i][j] = True means data sample i belongs to cluster j
        """
        assert (encode_output.shape[1] == centroids.shape[1]),"Dimensions Mismatch"
        n = encode_output.shape[0]
        d = encode_output.shape[1]
        k = centroids.shape[0]

        z = encode_output.reshape(n, 1, d)
        z = z.repeat(1, k, 1)

        mu = centroids.reshape(1, k, d)
        mu = mu.repeat(n, 1, 1)

        mask_matrix = ~mask_matrix

        z = z.masked_fill(mask_matrix.unsqueeze(dim=-1), value=torch.tensor(0.0))
        mu = mu.masked_fill(mask_matrix.unsqueeze(dim=-1), value=torch.tensor(0.0))

        loss = torch.pow(z - mu, 2).sum(dim=-1).reshape(n, k)

        return loss.mean()
    
    def inter_class_distance(self, centroids):
        """Calculate the inter_class distance from known results.
        Args:
            centroids (torch.float32): k x d where d is the hiddent size
        """
        d = centroids.shape[1]
        k = centroids.shape[0]
        mu1 = centroids.reshape(1, k, d)
        mu1 = mu1.repeat(k, 1, 1)
        mu2 = centroids.reshape(k, 1, d)
        mu2 = mu2.repeat(1, k, 1)
        
        loss = torch.pow(mu1 - mu2, 2).sum(dim=-1).reshape(k, k)

        return loss.mean()
        


# sample
encode_output = torch.randn(4,2)
centroids = torch.randn(3,2)
mask = torch.zeros((4,3), dtype=torch.bool)
mask[1,2] = True
mask[3,1] = True
loss_cal = KMeansClusteringLoss()
print(loss_cal(encode_output, centroids, mask))