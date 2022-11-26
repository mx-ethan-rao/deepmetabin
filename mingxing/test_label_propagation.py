import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot
import warnings
from sklearn.exceptions import ConvergenceWarning

class lbp(LabelPropagation):
    def __init__(self, kernel='rbf', *, gamma=20, n_neighbors=3, max_iter=1000, tol=0.001, n_jobs=None):
        super().__init__(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=max_iter, tol=tol, n_jobs=n_jobs)

    def fit(self, X, y):
        """Fit a semi-supervised label propagation model based

        All the input data is provided matrix X (labeled and unlabeled)
        and corresponding label matrix y with a dedicated marker value for
        unlabeled samples.

        Parameters
        ----------
        X : array-like of pre-computed matrix (n_samples, n_samples)

        y : array-like of shape (n_samples,)
            `n_labeled_samples` (unlabeled points are marked as -1)
            All unlabeled samples will be transductively assigned labels.

        Returns
        -------
        self : object
        """
        # X, y = self._validate_data(X, y)
        # self.X_ = X
        # check_classification_targets(y)

        # actual graph construction (implementations should override this)
        graph_matrix = X
        # graph_matrix = self._build_graph()

        # label construction
        # construct a categorical distribution for classification only
        classes = np.unique(y)
        classes = (classes[classes != -1])
        self.classes_ = classes

        n_samples, n_classes = len(y), len(classes)

        alpha = self.alpha
        if self._variant == 'spreading' and \
                (alpha is None or alpha <= 0.0 or alpha >= 1.0):
            raise ValueError('alpha=%s is invalid: it must be inside '
                             'the open interval (0, 1)' % alpha)
        y = np.asarray(y)
        unlabeled = y == -1

        # initialize distributions
        self.label_distributions_ = np.zeros((n_samples, n_classes))
        for label in classes:
            self.label_distributions_[y == label, classes == label] = 1

        y_static = np.copy(self.label_distributions_)
        if self._variant == 'propagation':
            # LabelPropagation
            y_static[unlabeled] = 0
        else:
            # LabelSpreading
            y_static *= 1 - alpha

        l_previous = np.zeros((X.shape[0], n_classes))

        unlabeled = unlabeled[:, np.newaxis]
        # if sparse.isspmatrix(graph_matrix):
        #     graph_matrix = graph_matrix.tocsr()

        for self.n_iter_ in range(self.max_iter):
            if np.abs(self.label_distributions_ - l_previous).sum() < self.tol:
                break
            print(f"iter{self.n_iter_}")
            l_previous = self.label_distributions_
            self.label_distributions_ = sparse.csr_matrix(self.label_distributions_)
            self.label_distributions_ = safe_sparse_dot(
                graph_matrix, self.label_distributions_)
            self.label_distributions_ = self.label_distributions_.toarray()
            print(f"iter{self.n_iter_} 1")
            if self._variant == 'propagation':
                normalizer = np.sum(
                    self.label_distributions_, axis=1)[:, np.newaxis]
                print(f"iter{self.n_iter_} 2")
                normalizer = np.where(normalizer == 0.0, 1e-18, normalizer)
                print(f"iter{self.n_iter_} 3")
                self.label_distributions_ /= normalizer
                print(f"iter{self.n_iter_} 4")
                self.label_distributions_ = np.where(unlabeled,
                                                     self.label_distributions_,
                                                     y_static)
                print(f"iter{self.n_iter_} 5")

            else:
                # clamp
                self.label_distributions_ = np.multiply(
                    alpha, self.label_distributions_) + y_static
        else:
            warnings.warn(
                'max_iter=%d was reached without convergence.' % self.max_iter,
                category=ConvergenceWarning
            )
            self.n_iter_ += 1

        normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
        normalizer[normalizer == 0] = 1
        self.label_distributions_ /= normalizer

        # set the transduction item
        transduction = self.classes_[np.argmax(self.label_distributions_,
                                               axis=1)]
        self.transduction_ = transduction.ravel()
        return self.transduction_

# label_prop_model = lbp()
# iris = datasets.load_iris()
# rng = np.random.RandomState(42)
# random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
# labels = np.copy(iris.target)
# labels[random_unlabeled_points] = -1
# print(labels)
# label_prop_model.fit(iris.data, labels)
# print(label_prop_model.predict(iris.data))