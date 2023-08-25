from scipy.spatial.distance import cdist
from sklearn.cluster import kmeans_plusplus
import gurobipy as gb
import numpy as np
import time


def update_centers(X, centers, n_clusters, labels):
    """Update positions of cluster centers
    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        n_clusters (int): predefined number of clusters
        labels (np.array): current cluster assignments of objects
    Returns:
        np.array: the updated positions of cluster centers
    """
    for i in range(n_clusters):
        centers[i] = X[labels == i, :].mean(axis=0)
    return centers


def assign_objects(X, centers, ml, cl, p, assignment_time_limit):
    """Assigns objects to clusters
    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        ml (list): must-link pairs as a list of tuples
        cl (list): cannot-link pairs as a list of tuples
        p (float): control parameter for penalty
        assignment_time_limit (float): solver time limit
    Returns:
        np.array: cluster labels for objects
    """

    # Compute model input
    n = X.shape[0]
    k = centers.shape[0]
    distances = cdist(X, centers)
    big_m = distances.max()
    assignments = {(i, j): distances[i, j] for i in range(n) for j in range(k)}

    # Create model
    m = gb.Model()

    # Add binary decision variables (with obj-argument we already define the coefficients of the objective function)
    x = m.addVars(assignments, obj=distances, vtype=gb.GRB.BINARY)
    y = m.addVars(cl, obj=big_m * p, lb=0)
    z = m.addVars(ml, obj=big_m * p, lb=0)

    # Add constraints
    m.addConstrs(x.sum(i, '*') == 1 for i in range(n))
    m.addConstrs(x.sum('*', j) >= 1 for j in range(k))
    m.addConstrs(x[i, j] + x[i_, j] <= 1 + y[i, i_] for i, i_ in cl for j in range(k))
    m.addConstrs(x[i, j] - x[i_, j] <= z[i, i_] for i, i_ in ml for j in range(k))

    # Set solver time limit
    m.setParam('TimeLimit', assignment_time_limit)

    # Determine optimal solution
    m.optimize()

    # Get labels from optimal assignment
    labels = np.array([j for i, j in x.keys() if x[i, j].X > 0.5])

    return labels


def get_total_distance(X, centers, labels):
    """Computes total distance between objects and cluster centers
    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        labels (np.array): current cluster assignments of objects
    Returns:
        float: total distance
    """
    dist = np.sqrt(((X - centers[labels, :]) ** 2).sum(axis=1)).sum()
    return dist


def bh_kmeans(X, n_clusters, ml=None, cl=None, p=1, random_state=None, max_iter=100, time_limit=None,
              assignment_time_limit=None):
    """Finds partition of X subject to must-link and cannot-link constraints
    Args:
        X (np.array): feature vectors of objects
        n_clusters (int): predefined number of clusters
        ml (list): must-link pairs as a list of tuples
        cl (list): cannot-link pairs as a list of tuples
        p (float): control parameter for penalty
        random_state (int, RandomState instance): random state
        max_iter (int): maximum number of iterations of bh_kmeans algorithm
        time_limit (int): algorithm time limit
        assignment_time_limit (int): solver time limit
    Returns:
        np.array: cluster labels of objects
    """

    # Set time limits
    if time_limit is None:
        time_limit = 1e7
    if assignment_time_limit is None:
        assignment_time_limit = time_limit
    else:
        assignment_time_limit = min(time_limit, assignment_time_limit)

    # Initialize ml and cl with emtpy list if not provided
    if ml is None:
        ml = []
    if cl is None:
        cl = []

    # Start stopwatch
    tic = time.perf_counter()

    # Choose initial cluster centers using the k-means++ algorithm
    centers, _ = kmeans_plusplus(X, n_clusters=n_clusters, random_state=random_state)

    # Assign objects
    labels = assign_objects(X, centers, ml, cl, p, assignment_time_limit)

    # Initialize best labels
    best_labels = labels

    # Update centers
    centers = update_centers(X, centers, n_clusters, labels)

    # Compute total distance
    best_total_distance = get_total_distance(X, centers, labels)

    n_iter = 1
    elapsed_time = time.perf_counter() - tic
    while (n_iter < max_iter) and (elapsed_time < time_limit):

        # Assign objects
        labels = assign_objects(X, centers, ml, cl, p, assignment_time_limit)

        # Update centers
        centers = update_centers(X, centers, n_clusters, labels)

        # Compute total distance
        total_distance = get_total_distance(X, centers, labels)

        # Check stopping criterion
        if total_distance >= best_total_distance:
            break
        else:
            # Update best labels and best total distance
            best_labels = labels
            best_total_distance = total_distance

        # Increase iteration counter
        n_iter += 1
        elapsed_time = time.perf_counter() - tic


    return best_labels