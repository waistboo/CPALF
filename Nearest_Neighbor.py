import pickle
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import torch
from torch_geometric.data import Data

def create_graph_from_nearest_neighbors(points):
    """
    Create an undirected graph where each node connects to its nearest neighbor.
    Returns a networkx Graph with node attribute 'pos' and weighted edges.
    """
    G = nx.Graph()
    for i in range(len(points)):
        G.add_node(i, pos=points[i])

    tree = KDTree(points)
    for i in range(len(points)):
        distances, indices = tree.query(points[i], k=2)
        if i < indices[1]:
            G.add_edge(i, indices[1], weight=distances[1])
    return G


def find_connected_components(G):
    """
    Return list of connected components (as sets of node indices).
    """
    components = list(nx.connected_components(G))
    return components


# Example usage (small random example)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import copy
    import time
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy.spatial.distance import euclidean
    from sklearn.datasets import make_classification
    from sklearn.metrics import adjusted_rand_score
    from sklearn.neighbors import NearestNeighbors
    from networkx.drawing.nx_pydot import graphviz_layout

    np.random.seed(0)
    points = np.random.rand(10, 2)
    G = create_graph_from_nearest_neighbors(points)
    components = find_connected_components(G)


def nearest_neighbor_cal(feature_space):
    """
    Calculate nearest neighbors for each sample and return edges list (u, v, weight).
    Uses sklearn NearestNeighbors with n_neighbors=3 and excludes self (index 0).
    """
    neighbors = NearestNeighbors(n_neighbors=3).fit(feature_space)
    distances, nearest_neighbors = neighbors.kneighbors(feature_space, return_distance=True)
    edges = []
    for i in range(len(nearest_neighbors)):
        for j in range(1, len(nearest_neighbors[i])):
            u = i
            v = nearest_neighbors[i][j]
            weight = distances[i][j]
            edges.append((u, v, weight))
    return edges


def data_preprocess(data):
    """
    Small random jitter to feature vectors to avoid identical rows.
    """
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.0001
    data = data + random_matrix
    return data


def clustering_loop(feature_space):
    """
    Build a graph from nearest neighbors and return it. Placeholder wrapper.
    """
    Graph = nx.Graph()
    edges = nearest_neighbor_cal(feature_space)
    Graph.add_weighted_edges_from(edges)
    return Graph


def calculate_representativeness(G):
    """
    Compute representativeness score for each node:
    degree(node) + (sum of inverted edge weights connected to node) / total_weight
    """
    total_weight = sum(weight for _, _, weight in G.edges(data='weight'))
    representativeness_scores = {}
    for node in G.nodes():
        node_degree = G.degree(node)
        weighted_degree_sum = sum(1 / weight for _, _, weight in G.edges(node, data='weight') if weight > 0)
        representativeness = node_degree + (weighted_degree_sum / total_weight if total_weight > 0 else 0)
        representativeness_scores[node] = representativeness
    return representativeness_scores


def save_graph_for_gcn(Graph, node_features, filepath):
    """
    Convert a networkx graph and node features into a PyG Data object and save it with torch.save.
    """
    num_nodes = Graph.number_of_nodes()
    edge_index = torch.tensor(list(Graph.edges)).t().contiguous()
    edge_weight = []
    for u, v in Graph.edges():
        weight = Graph[u][v].get('weight', 1.0)
        edge_weight.append(weight)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    torch.save(data, filepath)
    print(f"Graph saved to {filepath}")


def save_data(filepath, data, center_vec_sorted):
    with open(filepath, 'wb') as file:
        pickle.dump((data, center_vec_sorted), file)


if __name__ == '__main__':
    datasets = ['Aviation']
    for dataset in datasets:
        # Example pipeline: load CSV, build NN graph, compute representativeness, save
        with open(f"datasets/{dataset}.csv", 'r') as file:
            data = np.loadtxt(file, delimiter=',')
        data = data_preprocess(data)
        Graph = nx.Graph()
        edges = nearest_neighbor_cal(data)
        Graph.add_weighted_edges_from(edges)
        representativeness_scores = calculate_representativeness(Graph)
        sorted_scores = sorted(representativeness_scores.items(), key=lambda x: x[1], reverse=True)
        print(sorted_scores)
        print(len(sorted_scores))
        save_data("datasets/{dataset}.pkl", data, sorted_scores)
        save_graph_for_gcn(Graph, data, "datasets/{dataset}.pt")