import pickle
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

import torch
from torch_geometric.data import Data


def create_graph_from_nearest_neighbors(points):
    # 创建一个空的图
    G = nx.Graph()

    # 为每个点创建一个节点
    for i in range(len(points)):
        G.add_node(i, pos=points[i])

    # 创建KDTree用于快速查找最近邻
    tree = KDTree(points)

    # 为每个点找到最近邻并创建一条边
    for i in range(len(points)):
        # 查询最近的两个点（包括自身）
        distances, indices = tree.query(points[i], k=2)
        # 最近的非自身的点是indices[1]，因为indices[0]是自身
        # 只添加一次边（i < indices[1]防止双向添加）
        if i < indices[1]:
            G.add_edge(i, indices[1], weight=distances[1])
    return G


def find_connected_components(G):
    # 找到连通分量
    components = list(nx.connected_components(G))
    return components


# 示例数据
np.random.seed(0)
points = np.random.rand(10, 2)  # 生成一个10x2的随机数组

# 创建图并获取连通分量
G = create_graph_from_nearest_neighbors(points)
components = find_connected_components(G)



# 可以选择使用matplotlib来可视化图
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
# draw_graph 函数使用 matplotlib 可视化图 G
def draw_graph(G):
    pos = graphviz_layout(G, prog="twopi")
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, alpha=0.5, node_color="blue", with_labels=True, font_size=20, node_size=30)
    plt.axis("equal")
    plt.show()

# def nearest_neighbor_cal(feature_space):
#     neighbors=NearestNeighbors(n_neighbors=2).fit(feature_space)
#     distance,nearest_neighbors= neighbors.kneighbors(feature_space,return_distance=True)
#     distance=distance[:,1]
#     nearest_neighbors=nearest_neighbors.tolist()
#     for i in range(len(nearest_neighbors)):
#         nearest_neighbors[i].append(distance[i])
#     return nearest_neighbors

def nearest_neighbor_cal(feature_space):
    neighbors = NearestNeighbors(n_neighbors=3).fit(feature_space)  # 设置为3，实际只选了除自身的第二近和第三近的邻结点
    distances, nearest_neighbors = neighbors.kneighbors(feature_space, return_distance=True)
    # distances是一个二维数组，其中每一行是对应点到其最近邻的距离。
    # nearest_neighbors也是一个二维数组，每一行是对应点的最近邻的索引
    edges = []
    for i in range(len(nearest_neighbors)):  # 结点个数
        for j in range(1, len(nearest_neighbors[i])):  # 结点i的最近邻结点个数，除开自身从1开始
            u = i  # 当前点的索引
            v = nearest_neighbors[i][j]  # 最近邻点的索引
            weight = distances[i][j]  # 与最近邻的距离
            edges.append((u, v, weight))  # 添加边
    return edges


def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.0001
    data = data+random_matrix
    return data


# def representative_cal(sub_S):  # 传入一个子图
#     # 找最大度数的节点
#     degree_dict = dict(sub_S.degree())
#     max_degree = max(degree_dict.values())
#     nodes_with_max_degree = [node for node, degree in degree_dict.items() if degree == max_degree]  #
#
#     # 找最大度数的节点中最小加权度数的节点
#     min_weighted_degree_sum = float('inf')  # 初始化为正无穷大,记录最小加权度数和
#     min_weighted_degree_node = None  # 初始化为 None，用来记录对应的节点
#     for node in nodes_with_max_degree:
#         weighted_degree_sum = sum(weight for _, _, weight in sub_S.edges(data='weight', nbunch=node))  # 对于每个拥有最大度数的节点，计算它的加权度数和即所有邻居边的权重的总和
#         if weighted_degree_sum < min_weighted_degree_sum:
#             min_weighted_degree_sum = weighted_degree_sum
#             min_weighted_degree_node = node
#     representative = min_weighted_degree_node
#     return representative

def clustering_loop(feature_space):
    Graph = nx.Graph()
    edges = nearest_neighbor_cal(feature_space)
    Graph.add_weighted_edges_from(edges)
    return Graph


def graph_initialization(data):
    feature_space = copy.deepcopy(data)  # 深拷贝:对feature_space的更改不会影响到data
    dict_mapping = {}
    skeleton = nx.Graph()
    representatives, skeleton, dict_mapping = clustering_loop(feature_space, dict_mapping, skeleton)
    return skeleton


def calculate_representativeness(G):
    """
    计算图中每个节点的代表性。
    代表性公式为: degree(x_i) + (sum of weights of edges connected to x_i) / total weight of the graph.
    返回一个字典，键为节点，值为代表性得分。
    """
    # 计算整个图的边权重总和
    total_weight = sum(weight for _, _, weight in G.edges(data='weight'))
    # 初始化字典来存储每个节点的代表性得分
    representativeness_scores = {}
    # 遍历每个节点，计算其代表性
    for node in G.nodes():
        # 节点的度数
        node_degree = G.degree(node)
        # print(f"node_degree is {node_degree}")
        # # 该节点所有相连边的权重和
        # weighted_degree_sum = sum(weight for _, _, weight in G.edges(node, data='weight'))
        # 该节点所有相连边的倒数权重之和
        weighted_degree_sum = sum(1 / weight for _, _, weight in G.edges(node, data='weight') if weight > 0)
        # 计算代表性得分
        representativeness = node_degree + (weighted_degree_sum / total_weight if total_weight > 0 else 0)
        # 存储每个节点的代表性得分
        representativeness_scores[node] = representativeness
    return representativeness_scores


def save_graph_for_gnn(Graph, node_features, filepath):
    # 节点数量
    num_nodes = Graph.number_of_nodes()

    # 构建边索引（2, num_edges）
    # 注意networkx的边是无序的，但PyG一般用有序张量格式
    edge_index = torch.tensor(list(Graph.edges)).t().contiguous()  # shape [2, num_edges]

    # 构建边权重张量
    edge_weight = []
    for u, v in Graph.edges():
        weight = Graph[u][v].get('weight', 1.0)  # 若无权重默认1.0
        edge_weight.append(weight)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # 节点特征转为tensor
    x = torch.tensor(node_features, dtype=torch.float)

    # 节点标签，全部初始化为-1
    y = torch.full((num_nodes,), -1, dtype=torch.long)

    # 创建Data对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

    # 保存Data对象，可以用torch.save
    torch.save(data, filepath)
    print(f"new graph{data}")
    print(f"Graph saved to {filepath}")


def save_data(filepath, data, center_vec_sorted):
    with open(filepath, 'wb') as file:
        pickle.dump((data, center_vec_sorted), file)


if __name__ == '__main__':
    with open("my_test_data/3000/Aviation_1.csv", 'r') as file:
    # with open("D:/Python/learn_pytorch/pythonProject/hf_test_data/Youtube_1.csv", 'r') as file:
        data = np.loadtxt(file, delimiter=',')
    data = data_preprocess(data)
    Graph = nx.Graph()
    edges = nearest_neighbor_cal(data)
    Graph.add_weighted_edges_from(edges)
    representativeness_scores = calculate_representativeness(Graph)
    # 按照分数降序排序
    sorted_scores = sorted(representativeness_scores.items(), key=lambda x: x[1], reverse=True)
    print(sorted_scores)
    print(len(sorted_scores))
    save_data("my_test_data/3000/Aviation_1.pkl", data, sorted_scores)
    # 可视化图
    # draw_graph(Graph)

    save_graph_for_gnn(Graph, data, "my_test_data/3000/Aviation_1.pt")














