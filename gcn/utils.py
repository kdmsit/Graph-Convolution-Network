import numpy as np
import pickle as pkl
import networkx as nx
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# region Create Mask
def sample_mask(idx, l):
    """Create mask.
    This creates an array of size l with first idx terms = "True".
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
# endregion


# region Load Dataset
def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    # region Load Data Set
    '''
    We will load data for the dataset passed through parameter here.Used files(for cora dataset) :
    ind.cora.x  ind.cora.y  ind.cora.tx    ind.cora.ty    ind.cora.allx    ind.cora.ally    ind.cora.graph    ind.cora.test.index  
    '''
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)                                          #sort test data sets as per index
    # endregion

    # region Special Work for Citeseer(PENDING)
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    # endregion

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # Generate Adjacency Matrix from Graph

    nodelist=list(range(np.shape(adj)[0]))

    features = sp.vstack((allx, tx)).tolil()                                #stack all features(train+test) together(Vertically Row wise).tolil() coverts the matrix to Linked list format.
    features[test_idx_reorder, :] = features[test_idx_range, :]             #sort test data features (NOT SURE)


    labels = np.vstack((ally, ty))                                          #Stack All labels together (Vertically Row wise).2708 one hot vectors each of dimension 7.
    labels[test_idx_reorder, :] = labels[test_idx_range, :]                 #sort test data Labels (NOT SURE)

    idx_train, idx_test, tr_y, tst_y = train_test_split(nodelist, labels, test_size=0.25, random_state=42)
    idx_train, idx_val, tr_y, val_y = train_test_split(idx_train, tr_y, test_size=0.1, random_state=42)

    # idx_test = test_idx_range.tolist()                                      #Index of Test data
    # idx_train = range(len(y))                                               #idx_train:range of length of labelled data
    # idx_val = range(len(y), len(y)+500)                                     #idx_val:range of length of labelled data to length of labelled data+500.This is validation data item's index.

    # region Create Mask
    '''
    Created Mask using sample_mask() function defined above.
    train_musk:Create an np.array of size |v|=2708(cora) and make idx_train entry True and others false.
    val_mask:Create an np.array of size |v|=2708(cora) and make idx_val entry True and others false.(Validation)
    test_mask:Create an np.array of size |v|=2708(cora) and make idx_test entry True and others false.
    '''
    train_mask = sample_mask(idx_train, labels.shape[0])                    #Out of 2708 first 140 has true,rest all false.
    val_mask = sample_mask(idx_val, labels.shape[0])                        #Form 140-640 true rest all false(For Validation)
    test_mask = sample_mask(idx_test, labels.shape[0])                      #Out of 2708 last 1000 has true,rest all false.
    # endregion

    # region Create True Label
    '''
    Create True Label for Train,Validation and Test set.
    '''
    y_train = np.zeros(labels.shape)                                        #numpy.zeros of size (2708,7)
    y_val = np.zeros(labels.shape)                                          #numpy.zeros of size (2708,7)
    y_test = np.zeros(labels.shape)                                         #numpy.zeros of size (2708,7)
    y_train[train_mask, :] = labels[train_mask, :]                          #First 140(train_mask has true) will have one hot of true labels
    y_val[val_mask, :] = labels[val_mask, :]                                #range 140-640(val_mask has true) will have one hot of true labels
    y_test[test_mask, :] = labels[test_mask, :]                             #last 1000(test_mask has true) will have one hot of true labels
    # endregion
    print(y_train)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
# endregion


# region Preprocess Features
def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation.
       Divide each row by total number of 1's in that row.
       Like First Row has 9 1's ,then make entry-0.1111111 in those 9 entries
       Second Row has 23 1's, then make entry-0.04347 in those 23 entries.
       So on..
    """
    rowsum = np.array(features.sum(1))                      #Sum features for each Node.return np.ndarray with size (2708,1)[Sum of each 2708 nodes]
    r_inv = np.power(rowsum, -1).flatten()                  #(1/sum) for each node and flatten it.ndarray with size (2708,)
    r_inv[np.isinf(r_inv)] = 0.                             #Check for positive or negative infinity and make those entries as 0.
    r_mat_inv = sp.diags(r_inv)                             #Diagonal matrix of shape (2708,2708) with (i,i) contains (1/sum) for ith node.
    features = r_mat_inv.dot(features)                      #Dot product between Diagonal Matrix and features matrx.
    return sparse_to_tuple(features)                        #Return sparse matrix as tuple.
# endregion


# region Normalise Adjacency Matrix.
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix.
    Input to this procedure is : Adj_hat=Adj+Identity_Matrix
    It will return: Normalised_Adj=D^(-0.5).Adj_hat.D^(-0.5) [D-Degree Matrx]
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))                                                   #Dergees of each node.
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()                                   #degree^(-0.5) and flatten
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)                                           #D^(-0.5):diagonal matrix with (i,i) element contains degree(nodei)^(-0.5)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()          # Adj.D^(-0.5).Tr.D^(-0.5)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
       Adj_hat=Adj+Identity_Matrix
       Normalised_Adj=D^(-0.5).Adj_hat.D^(-0.5) [D-Degree Matrx]
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)
# endregion


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
