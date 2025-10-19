import csv
from collections import defaultdict

import sys  # 导入sys模块
sys.setrecursionlimit(300000)  # 将默认的递归深度修改为3000

# -*- coding: utf-8 -*-
class Graph:
    def __init__(self):
        self.vertices = {}
        self.edges = defaultdict(list)

    #将给定顶点和标签添加到图形中
    def add_vertex(self, vertex_id, label):
        self.vertices[vertex_id] = int(label)

    #在两个顶点之间添加具有标签的边
    def add_edge(self, src, dst, label):
        self.edges[src].append((dst, int(label)))
        self.edges[dst].append((src, int(label)))

    #深度优先遍历生成最小DFS代码
    def generate_min_dfs_code(self):
        min_vertex = min(self.vertices, key=int)
        visited = set()
        parent_map = {}
        dfs_code = []

        def visit(vertex, parent):
            visited.add(vertex)
            parent_map[vertex] = parent
            outgoing_edges = []

            # 前向边和后向边
            for neighbor, label in self.edges[vertex]:
                if neighbor not in visited:
                    outgoing_edges.append((neighbor, label))  # 前向边
                elif neighbor != parent:  # 后向边
                    # Record the actual label of the back edge
                    dfs_code.append((vertex, neighbor, self.vertices[vertex], label, self.vertices[neighbor]))
                    #  v1,v2,l1,e,l2

            outgoing_edges.sort(key=lambda x: (self.vertices[x[0]], x[1]))

            # Recursively visit all unvisited neighbors
            for neighbor, label in outgoing_edges:
                if neighbor not in visited:
                    dfs_code.append((vertex, neighbor, self.vertices[vertex], label, self.vertices[neighbor]))
                    visit(neighbor, vertex)

        visit(min_vertex, None)
        return dfs_code


def compute_pattern_sizes(filename):
    pattern_sizes = defaultdict(int)
    current_id = None
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # 检测样本ID行（如"0:"）
            if line.endswith(':'):
                # 保存上一个模式的记录
                current_id = line[:-1]  # 去掉冒号
                pattern_sizes[current_id] = 0  # 初始化计数器
            else:
                # 统计非ID行（顶点或边）
                pattern_sizes[current_id] += 1

    # 处理最后一个模式
    if current_id is not None:
        pattern_sizes[current_id] += 1
    pattern_sizes_dict = dict(pattern_sizes)
    return pattern_sizes_dict


def parse_graphs(filename):
    graphs = []
    current_graph = Graph()
    existing_vertex_ids = set()  # To track existing vertex IDs
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertex_id = parts[1]
                if vertex_id in existing_vertex_ids:
                    # Save the current graph and start a new one if the current is not empty
                    if current_graph.vertices:
                        graphs.append(current_graph)
                    current_graph = Graph()
                    existing_vertex_ids.clear()

                current_graph.add_vertex(vertex_id, parts[2])
                existing_vertex_ids.add(vertex_id)

            elif parts[0] == 'e':
                if parts[1] in current_graph.vertices and parts[2] in current_graph.vertices:
                    current_graph.add_edge(parts[1], parts[2], parts[3])

    if current_graph.vertices:
        graphs.append(current_graph)
    return graphs


def load_label_encodings(encoding_file):
    label_encodings = {}
    with open(encoding_file, 'r') as file:
        for line in file:
            if line.strip():
                key, value = line.strip().split(':')
                label_encodings[int(key)] = value.strip()  # 删除任何前后空格
    return label_encodings


def main(input_file, output_file, encoding_file):
    graphs = parse_graphs(input_file)
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar=' ')
        for graph in graphs:
            dfs_code = graph.generate_min_dfs_code()
            label_encodings = load_label_encodings(encoding_file)
            binary_vector = []
            print(dfs_code)
            print(len(dfs_code))
            for src, dst, src_lbl, lbl, dst_lbl in dfs_code:
                encoded_src_lbl = label_encodings.get(int(src_lbl))
                encoded_lbl = label_encodings.get(int(lbl))
                encoded_dst_lbl = label_encodings.get(int(dst_lbl))
                # binary_vector.append(f'{src},{dst},{encoded_src_lbl},{encoded_lbl},{encoded_dst_lbl}')
                binary_vector.append(f'{src},{dst},{",".join(encoded_src_lbl)},{",".join(encoded_lbl)},{",".join(encoded_dst_lbl)}')

            # 将 binary_vector 统一到固定维数
            lad = 800
            print(binary_vector)
            vector_length = len(','.join(binary_vector).split(','))
            print(vector_length)
            if vector_length < lad:
                padding = ['0'] * (lad - vector_length)
                binary_vector.extend(padding)
            if vector_length > lad:
                print("溢出了")
            writer.writerow([','.join(binary_vector).replace(' ', '')])
            # Formatting the output to match the desired format
            # formatted_output = '[' + ','.join(f"({src},{dst},{src_lbl},{lbl},{dst_lbl})" for src, dst, src_lbl, lbl, dst_lbl in dfs_code) + ']'
            # file.write(f"{formatted_output}\n")

if __name__ == '__main__':
    input_file = "my_test_data/1000/Aviation_1.txt"
    output_file = "my_test_data/1000/Aviation_1.csv"
    # input_file = "data/simpleData/input/new0.6/patent_no.txt"
    # output_file = "data/simpleData/dfscode/new0.6/patent-0.600.csv"
    encoding_file = "all_data/Aviation-l.txt"
    main(input_file, output_file, encoding_file)
