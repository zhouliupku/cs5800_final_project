"""
Travelling Sales Man Problem
Solve by Christofides-Serdyukov Algorithm
Reference:
Christofides N. Worst-case analysis of a new heuristic for the travelling
salesman problem[R]. Carnegie-Mellon Univ Pittsburgh Pa Management Sciences
Research Group, 1976.
Time Complexity: O(n^3)

Author: Jing Ming
Date: Augest 14, 2022
"""
import math
import networkx as nx

from networkx.algorithms.matching import max_weight_matching
from FibonacciHeap import FibonacciHeap

class Graph:
    """
    Use a list of list to represent graph by adjacency matrix.
    """
    def __init__(self):
        self.graph = []
        self.num_nodes = 0

    def build_graph(self, location):
        self.num_nodes = len(location)
        for i in range(self.num_nodes):
            self.graph.append([])
            for j in range(self.num_nodes):
                if i == j:
                    self.graph[i].append(0)
                else:
                    weight = self.euclidean_distance(location[i], location[j])
                    self.graph[i].append(weight)

    def euclidean_distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1.0 / 2.0)

    def print_graph(self):
        print(f"Graph represented by matrix: {self.graph}")

def generate_hamiltonian_circuit(eulerian_circuit, graph):
    start = eulerian_circuit[0]
    hamiltonian_circuit = [start]
    visited = [False] * len(eulerian_circuit)
    visited[start] = True
    cost = 0

    for v in eulerian_circuit:
        if not visited[v]:
            previous = hamiltonian_circuit[-1]
            hamiltonian_circuit.append(v)
            visited[v] = True
            cost += graph.graph[previous][v]

    previous = hamiltonian_circuit[-1]
    hamiltonian_circuit.append(start)
    cost += graph.graph[previous][start]

    return hamiltonian_circuit, cost


def remove_edge_from_multigraph(multigraph, v, w):
    for idx, edge in enumerate(multigraph):
        if (edge[0] == v and edge[1] == w) or (edge[1] == v and edge[0] == w):
            del multigraph[idx]


def generate_eulerian_circuit(multigraph, graph):
    # Find neighbors
    neighbors = {}
    for edge in multigraph:
        if edge[0] not in neighbors:
            neighbors[edge[0]] = []
        neighbors[edge[0]].append(edge[1])
        if edge[1] not in neighbors:
            neighbors[edge[1]] = []
        neighbors[edge[1]].append(edge[0])

    # Generate eulerian circuit
    start = multigraph[0][0]
    eulerian_circuit = [neighbors[start][0]]

    while len(multigraph) > 0:
        for idx, v in enumerate(eulerian_circuit):
            if len(neighbors[v]) > 0:
                break

        while len(neighbors[v]) > 0:
            w = neighbors[v][0]

            remove_edge_from_multigraph(multigraph, v, w)

            del neighbors[v][(neighbors[v].index(w))]
            del neighbors[w][(neighbors[w].index(v))]

            idx += 1
            eulerian_circuit.insert(idx, w)

            v = w

    return eulerian_circuit

def generate_minimum_weight_perfect_mathcing(graph, odd_vertices):
    n = len(odd_vertices)
    G = nx.Graph()
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            u = odd_vertices[i]
            v = odd_vertices[j]
            edges.append([u, v, graph.graph[u][v] * -1])
    print(f"{edges}")
    G.add_weighted_edges_from(edges)

    matching = nx.max_weight_matching(G, maxcardinality=True)
    print(f"{matching}")
    mwpm = []

    for edge in matching:
        u = edge[0]
        v = edge[1]
        mwpm.append([u, v, graph.graph[u][v]])

    return mwpm

def find_odd_vertices(mst):
    vertex_degree = {}
    odd_vertices = []

    for edge in mst:
        vertex_degree[edge[0]] = vertex_degree.get(edge[0], 0) + 1
        vertex_degree[edge[1]] = vertex_degree.get(edge[1], 0) + 1

    for key, value in vertex_degree.items():
        if value % 2 == 1:
            odd_vertices.append(key)

    return odd_vertices

def generate_minimum_spanning_tree_with_prim(graph, source):
    mst_prim = []
    visited = set()
    visited.add(source)
    parent = [None] * graph.num_nodes
    key = [math.inf] * graph.num_nodes

    key[source] = 0

    node_list = []

    fibonacci_heap = FibonacciHeap()
    for dest in range(graph.num_nodes):
        if dest == source:
            continue
        else:
            key[dest] = graph.graph[source][dest]
            # store the node in fibonacci_heap, used later for decrease key
            node_list.append(fibonacci_heap.insert(key[dest], dest))
            parent[dest] = source

    while node_list and len(visited) != graph.num_nodes:
            node = fibonacci_heap.extract_min()
            distance = node.key
            dest = node.value
            node_list.remove(node)
            if dest not in visited:
                visited.add(dest)
                mst_prim.append([parent[dest], dest, distance])
                for node in node_list:
                    neighbor = node.value
                    if neighbor not in visited and graph.graph[dest][neighbor] < key[neighbor]:
                        parent[neighbor] = dest
                        key[neighbor] = graph.graph[dest][neighbor]
                        fibonacci_heap.decrease_key(node, key[neighbor])

    return mst_prim


def read_input(input_file):
    city_name = []
    city_location = []
    # `data.txt` in format: [city_name],[city_location_x],[city_location_y]
    with open(input_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")
            city_name.append(line[0])
            city_location.append([float(line[1]), float(line[2])])

    return city_name, city_location


def create_graph(input_file):
    city_name, city_location = read_input(input_file)
    graph = Graph()
    graph.build_graph(city_location)

    return graph


if __name__ == "__main__":
    # Create a graph
    graph = create_graph("data.txt")

    # Find its minimum spanning tree
    mst_prim = generate_minimum_spanning_tree_with_prim(graph, 0)
    print(f"mst prim: {mst_prim}")

    # Find odd_vertices
    odd_vertices = find_odd_vertices(mst_prim)
    print(f"odd vertices: {odd_vertices}")

    # Find minimum weight perfect mathcing
    mwpm = generate_minimum_weight_perfect_mathcing(graph, odd_vertices)
    print(f"mwpm: {mwpm}")

    # Combine mst and mwpm result
    mst_prim.extend(mwpm)
    multigraph = mst_prim
    print(f"multigraph: {multigraph}")

    # Create Eulerian circuit
    eulerian_circuit = generate_eulerian_circuit(multigraph, graph)
    print(f"eulerian_circuit: {eulerian_circuit}")

    # Generate Hamiltonian circuit
    hamiltonian_circuit, cost = generate_hamiltonian_circuit(eulerian_circuit, graph)
    print(f"hamiltonian_circuit: {hamiltonian_circuit}\ntotal cost: {cost}")



