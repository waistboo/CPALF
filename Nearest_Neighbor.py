import pickle
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

import torch
from torch_geometric.data import Data


def create_graph_from_nearest_neighbors(points):
    # Create an empty graph
    G = nx.Graph()

    # Create a node for each point
    for i in range(len(points)):
        G.add_node(i, pos=points[i])

    # Create KDTree for fast nearest neighbor search
    tree = KDTree(points)

    # Find nearest neighbor for each point and create an edge
    for i in range(len(points)):
        # Query the two nearest points (including itself)
        distances, indices = tree.query(points[i], k=2)
        # The nearest non-self point is indices[1], since indices[0] is itself
        # Add edge only once (i < indices[1] prevents bidirectional addition)
        if i < indices[1]:
            G.add_edge(i, indices[1], weight=distances[1])
    return G


def find_connected_components(G):
    # Find connected components
    components = list(nx.connected_components(G))
    return components


# Example data
np.random.seed(0)
points = np.random.rand(10, 2)  # Generate a 10x2 random array

# Create graph and get connected components
G = create_graph_from_nearest_neighbors(points)
components = find_connected_components(G)


# Optional: Use matplotlib to visualize the graph
import matplotlib.pyplot as plt
import copy
import time

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_classification
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from networkx.drawing.nx_pydot import graphviz_layout

# from dataset.iris.iris_generate import generate_iris_data

np.random.seed(0)
# draw_graph function visualizes graph G using matplotlib
def draw_graph(G):
    pos = graphviz_layout(G, prog="twopi")
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, alpha=0.5, node_color="blue", with_labels=True, font_size=20, node_size=30)
    plt.axis("equal")
    plt.show()


def nearest_neighbor_cal(feature_space):
    neighbors = NearestNeighbors(n_neighbors=3).fit(feature_space)
    distances, nearest_neighbors = neighbors.kneighbors(feature_space, return_distance=True)
    edges = []
    for i in range(len(nearest_neighbors)):  # Number of nodes
        for j in range(1, len(nearest_neighbors[i])):
            u = i  # Index of current point
            v = nearest_neighbors[i][j]  # Index of nearest neighbor point
            weight = distances[i][j]  # Distance to nearest neighbor
            edges.append((u, v, weight))  # Add edge
    return edges


def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.0001
    data = data + random_matrix
    return data

def clustering_loop(feature_space):
    Graph = nx.Graph()
    edges = nearest_neighbor_cal(feature_space)
    Graph.add_weighted_edges_from(edges)
    return Graph


def graph_initialization(data):
    feature_space = copy.deepcopy(data)  # Deep copy: changes to feature_space will not affect data
    dict_mapping = {}
    skeleton = nx.Graph()
    representatives, skeleton, dict_mapping = clustering_loop(feature_space, dict_mapping, skeleton)
    return skeleton


def calculate_representativeness(G):
    """
    Calculate the representativeness of each node in the graph.
    Representativeness formula: degree(x_i) + (sum of weights of edges connected to x_i) / total weight of the graph.
    Returns a dictionary with nodes as keys and representativeness scores as values.
    """
    # Calculate total weight of edges in the graph
    total_weight = sum(weight for _, _, weight in G.edges(data='weight'))
    # Initialize dictionary to store representativeness scores for each node
    representativeness_scores = {}
    # Iterate through each node to calculate its representativeness
    for node in G.nodes():
        # Degree of the node
        node_degree = G.degree(node)
        # print(f"node_degree is {node_degree}")
        weighted_degree_sum = sum(1 / weight for _, _, weight in G.edges(node, data='weight') if weight > 0)
        representativeness = node_degree + (weighted_degree_sum / total_weight if total_weight > 0 else 0)
        representativeness_scores[node] = representativeness
    return representativeness_scores


def save_graph_for_gcn(Graph, node_features, filepath):
    # Number of nodes
    num_nodes = Graph.number_of_nodes()

    # Build edge index (2, num_edges)
    # Note: networkx edges are unordered, but PyG generally uses ordered tensor format
    edge_index = torch.tensor(list(Graph.edges)).t().contiguous()  # shape [2, num_edges]

    # Build edge weight tensor
    edge_weight = []
    for u, v in Graph.edges():
        weight = Graph[u][v].get('weight', 1.0)  # Default to 1.0 if no weight exists
        edge_weight.append(weight)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # Convert node features to tensor
    x = torch.tensor(node_features, dtype=torch.float)

    # Node labels, initialized to -1 for all
    y = torch.full((num_nodes,), -1, dtype=torch.long)

    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

    # Save Data object using torch.save
    torch.save(data, filepath)
    print(f"new graph{data}")
    print(f"Graph saved to {filepath}")


def save_data(filepath, data, center_vec_sorted):
    with open(filepath, 'wb') as file:
        pickle.dump((data, center_vec_sorted), file)


if __name__ == '__main__':
    datasets = ['dblp']
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
        save_data(f"datasets/{dataset}.pkl", data, sorted_scores)
        # Visualize the graph
        # draw_graph(Graph)
        save_graph_for_gcn(Graph, data, f"datasets/{dataset}.pt")