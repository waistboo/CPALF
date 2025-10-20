import csv
from collections import defaultdict

import sys
sys.setrecursionlimit(300000)  # increase recursion limit

# -*- coding: utf-8 -*-
class Graph:
    def __init__(self):
        self.vertices = {}
        self.edges = defaultdict(list)

    # add a vertex with label to the graph
    def add_vertex(self, vertex_id, label):
        self.vertices[vertex_id] = int(label)

    # add an edge with label between two vertices (undirected)
    def add_edge(self, src, dst, label):
        self.edges[src].append((dst, int(label)))
        self.edges[dst].append((src, int(label)))

    # perform DFS to generate a DFS code (sequence) starting from smallest vertex id
    def generate_min_dfs_code(self):
        min_vertex = min(self.vertices, key=int)
        visited = set()
        parent_map = {}
        dfs_code = []

        def visit(vertex, parent):
            visited.add(vertex)
            parent_map[vertex] = parent
            outgoing_edges = []

            # forward edges and back edges
            for neighbor, label in self.edges[vertex]:
                if neighbor not in visited:
                    outgoing_edges.append((neighbor, label))  # forward edge
                elif neighbor != parent:
                    # record the actual label of the back edge
                    dfs_code.append((vertex, neighbor, self.vertices[vertex], label, self.vertices[neighbor]))
                    # appended tuple: (v1, v2, label_v1, edge_label, label_v2)

            # sort outgoing edges by (neighbor_label, edge_label) to have deterministic order
            outgoing_edges.sort(key=lambda x: (self.vertices[x[0]], x[1]))

            # recursively visit all unvisited neighbors
            for neighbor, label in outgoing_edges:
                if neighbor not in visited:
                    dfs_code.append((vertex, neighbor, self.vertices[vertex], label, self.vertices[neighbor]))
                    visit(neighbor, vertex)

        visit(min_vertex, None)
        return dfs_code


def compute_pattern_sizes(filename):
    """
    Parse a file that contains pattern entries and compute pattern sizes.
    Expected format: lines that end with ':' indicate a pattern id line (e.g., "0:"),
    subsequent lines belong to that pattern and are counted as its size.
    """
    pattern_sizes = defaultdict(int)
    current_id = None
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # detect pattern id line (ending with ':')
            if line.endswith(':'):
                current_id = line[:-1]
                pattern_sizes[current_id] = 0
            else:
                # count non-id lines as pattern content
                pattern_sizes[current_id] += 1

    # account for last pattern (already counted)
    pattern_sizes_dict = dict(pattern_sizes)
    return pattern_sizes_dict


def parse_graphs(filename):
    """
    Parse graphs from a text file with vertex (v) and edge (e) lines.
    - 'v id label' defines a vertex
    - 'e src dst label' defines an edge (only added if both endpoints exist in current graph)
    If a vertex id repeats, start a new graph.
    """
    graphs = []
    current_graph = Graph()
    existing_vertex_ids = set()
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertex_id = parts[1]
                # if vertex id repeats, treat as start of a new graph
                if vertex_id in existing_vertex_ids:
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
    """
    Load label encoding mapping from a file where each line has "int_key:encoded_value".
    Returns a dict mapping int -> encoded string (or list).
    """
    label_encodings = {}
    with open(encoding_file, 'r') as file:
        for line in file:
            if line.strip():
                key, value = line.strip().split(':')
                label_encodings[int(key)] = value.strip()
    return label_encodings


def main(input_file, output_file, encoding_file):
    graphs = parse_graphs(input_file)
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar=' ')
        for graph in graphs:
            dfs_code = graph.generate_min_dfs_code()
            label_encodings = load_label_encodings(encoding_file)
            binary_vector = []
            # print debug information (DFS code and its length)
            print(dfs_code)
            print(len(dfs_code))
            for src, dst, src_lbl, lbl, dst_lbl in dfs_code:
                encoded_src_lbl = label_encodings.get(int(src_lbl))
                encoded_lbl = label_encodings.get(int(lbl))
                encoded_dst_lbl = label_encodings.get(int(dst_lbl))
                # join encoded label strings; encoding values may represent lists in string form
                binary_vector.append(f'{src},{dst},{",".join(encoded_src_lbl)},{",".join(encoded_lbl)},{",".join(encoded_dst_lbl)}')

            # unify binary_vector to fixed dimension
            target_length = 800
            print(binary_vector)
            vector_length = len(','.join(binary_vector).split(','))
            print(vector_length)
            if vector_length < target_length:
                padding = ['0'] * (target_length - vector_length)
                binary_vector.extend(padding)
            if vector_length > target_length:
                print("overflow: code length exceeds target vector length")
            writer.writerow([','.join(binary_vector).replace(' ', '')])


if __name__ == '__main__':
    datasets = ['Aviation']
    for dataset in datasets:
        input_file = f"datasets/pattern/{dataset}.txt"
        output_file = f"datasets/{dataset}.csv"
        encoding_file = "datasets/Initial Graph/Aviation-l.txt"
        main(input_file, output_file, encoding_file)
