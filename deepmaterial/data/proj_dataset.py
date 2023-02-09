import numpy as np
from torch.utils import data as data
from deepmaterial.utils.registry import DATASET_REGISTRY

def get_graph_elements(graph_, n_epochs):
    """
    gets elements of graphs, weights, and number of epochs per edge

    Parameters
    ----------
    graph_ : scipy.sparse.csr.csr_matrix
        umap graph of probabilities
    n_epochs : int
        maximum number of epochs per edge

    Returns
    -------
    graph scipy.sparse.csr.csr_matrix
        umap graph
    epochs_per_sample np.array
        number of epochs to train each sample for
    head np.array
        edge head
    tail np.array
        edge tail
    weight np.array
        edge weight
    n_vertices int
        number of verticies in graph
    """
    ### should we remove redundancies () here??
    # graph_ = remove_redundant_edges(graph_)

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = n_epochs * graph.data

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices




@DATASET_REGISTRY.register()
class projDataset(data.Dataset):
    """Paired projection dataset reader. Return the high-dimensional data point and confidence pair.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            hd_path (str): Data root path for high-dimensional data point.
            y_path (str): Data root path for confidence.
            
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(projDataset, self).__init__()
        self.opt = opt

        hd_path = opt['hd_path']
        y_path = opt['y_path']
        X = np.load(hd_path)
        self.confidence = np.load(y_path)
        graph = self.__get_neighbors(X)
        self.edge_to_exp, self.edge_from_exp = self.__get_full_data(graph)
        print('Intialization data done!')

    def __get_neighbors(self, X):
        from pynndescent import NNDescent
        from sklearn.utils import check_random_state
        from umap.umap_ import fuzzy_simplicial_set
        
        # number of trees in random projection forest
        n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(X.shape[0]))))
        # distance metric
        metric="euclidean"
        # number of neighbors for computing k-neighbor graph
        n_neighbors = 10

        # get nearest neighbors
        nnd = NNDescent(
            X.reshape((len(X), np.product(np.shape(X)[1:]))),
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        # self.sampler = torch.distributions.Uniform(-1.0, 1.0)
        # get indices and distances
        knn_indices, knn_dists = nnd.neighbor_graph
        random_state = check_random_state(None)
        # build fuzzy_simplicial_set
        umap_graph, sigmas, rhos = fuzzy_simplicial_set(
            X = X,
            n_neighbors = n_neighbors,
            metric = metric,
            random_state = random_state,
            knn_indices= knn_indices,
            knn_dists = knn_dists,
        )
        return umap_graph

    def __get_full_data(self, graph_):
        # get data from graph
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
            graph_, 1
        )

        edges_to_exp, edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )

        # shuffle edges
        edges_to_exp = edges_to_exp.astype(np.int64)
        edges_from_exp = edges_from_exp.astype(np.int64)

        return edges_to_exp, edges_from_exp
    def __getitem__(self, index):

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        if self.opt['phase'] == 'train':
            img_path = self.train_paths[index]
        else:
            img_path = self.test_paths[index]

        sample_edge_to_x, sample_edge_from_x = self.edge_to_exp[index], self.edge_from_exp[index]
        confidence =  self.confidence[index]

        return {'edge_to_x': sample_edge_to_x, 'edge_from_x': sample_edge_from_x, 'confidence': confidence}

    def __len__(self):
        return self.confidence.shape[0]
