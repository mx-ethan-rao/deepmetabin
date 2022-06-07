import numpy as np
import scipy.special


class Metric:
    def __init__(self):
        pass

    def GetPrecision(self, assessment_matrix):
        """Compute precision of binning result from assessment matrix.
        
        Args:
            assessment_matrix (np.ndarray): assessment matrix of binning results.

        Returns:
            precision (float): final precision of the binned result.
        """    
        k = assessment_matrix.shape[0] - 1
        s = assessment_matrix.shape[1] - 1
        n = self.compute_n_from_assessment_matrix(assessment_matrix)
        sum_k = 0
        for i in range(k):
            max_s = 0
            for j in range(s):
                if assessment_matrix[i][j] > max_s:
                    max_s = assessment_matrix[i][j]
            sum_k += max_s

        precision = sum_k / n
        return precision
    
    def GetRecall(self, assessment_matrix, unclassified):
        """Compute recall of binning result, k + 1 row stands for the 
        unbinned contigs.

        Args:
            assessment_matrix (np.ndarray): assessment matrix of binning results.
            unclassified (int): precomputed unclassified of samples.

        Returns:
            recall (float): final recall of the binned result.
        """
        k = assessment_matrix.shape[0] - 1
        s = assessment_matrix.shape[1] - 1
        n = self.compute_n_from_assessment_matrix(assessment_matrix)
        sum_s = 0
        for i in range(s):
            max_k = 0
            for j in range(k):
                if assessment_matrix[j][i] > max_k:
                    max_k = assessment_matrix[j][i]
            sum_s += max_k
        recall = sum_s / (n + unclassified)
        return recall

    def GetARI(self, assessment_matrix):
        """Compute ARI of binning result, k + 1 row stands for the 
        unbinned contigs, use scipy.special.binom as refererce.

        Args:
            assessment_matrix (np.ndarray): assessment matrix of binning results.
            k (int): binning result group number.
            s (int): ground truth group number.
            n (int): total number.

        Returns:
            ari (float): final ARI of the binned result.
        """
        k = assessment_matrix.shape[0] - 1
        s = assessment_matrix.shape[1] - 1
        n = self.compute_n_from_assessment_matrix(assessment_matrix)
        t1 = 0
        t2 = 0
        for i in range(k):
            sum_k = 0
            for j in range(s):
                sum_k += assessment_matrix[i][j]
            t1 += scipy.special.binom(sum_k, 2)
            
        for i in range(s):
            sum_s = 0
            for j in range(k):
                sum_s += assessment_matrix[j][i]
            t2 += scipy.special.binom(sum_s, 2)
        t3 = t1 * t2 / scipy.special.binom(n, 2)
        t = 0
        
        for i in range(k):
            for j in range(s):
                t += scipy.special.binom(assessment_matrix[i][j], 2)
        ari = (t - t3) / ((t1 + t2) / 2 - t3)
        return ari

    def GetF1(self, precision, recall):
        """Compute F1 of binning result, k + 1 row stands for the 
        unbinned contigs.

        Args:
            precision (float): precison computed above.
            recall (float): recall computed above.

        Returns:
            F1 (float): final F1 of the binned result.
        """
        if precision == 0.0 or recall == 0.0:
            return 0.0
        else:
            return 2 * precision * recall / (precision + recall)
    
    def compute_n_from_assessment_matrix(self, assessment_matrix):
        """Compute n from assessment matrix.

        Args:
            assessment_matrix (np.ndarray): assessment matrix.

        Returns:
            n (int): number of contigs both labeled from model and ground
                truth.
        """
        k = assessment_matrix.shape[0] - 1
        s = assessment_matrix.shape[1] - 1
        n = 0
        for i in range(k):
            for j in range(s):
                n += assessment_matrix[i][j]
        return n
