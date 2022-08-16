"""
Travelling Sales Man Problem
Solve by Christofides-Serdyukov Algorithm
Reference:
Christofides N. Worst-case analysis of a new heuristic for the travelling
salesman problem[R]. Carnegie-Mellon Univ Pittsburgh Pa Management Sciences
Research Group, 1976.
Time Complexity: O(n^3)

Minimum Spanning Tree is generated using Prim's Algorithm with binary heap
implementation. Using python embeded library heapq.

Minimum Weight Perfect Matching is generated using .max_weight_matching function
in NetworkX python library.

Author: Jing Ming
Date: Augest 16, 2022
"""
import math
import networkx as nx
import heapq
import time

from networkx.algorithms.matching import max_weight_matching
from collections import defaultdict, deque

class Graph:
    """
    Use a list of list to represent graph by adjacency matrix.
    """
    def __init__(self):
        self.graph = []
        self.num_nodes = 0

    # Build graph from location (x, y).
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

    # Calculate euclidean distance between the two points.
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
    # Mapping every vertex in multigraph to its neighbors.
    neighbors = defaultdict(list)
    for edge in multigraph:
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])

    # Generate eulerian circuit.
    # Start from the neighbors of first vertex of multigraph's first edge.
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
    """
    Minimum Weight Perfect Matching using NetworkX library.
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html
    Args:
        graph:  adjacency matrix represented Graph
        odd_vertices: the vertices of odd degree in the MST
    Returns:
        a list of edges (u, v, weight(u, v)) which represents a minimum weight
        perfect matching of complete graph.
    """

    mwpm = []
    edges = []
    n = len(odd_vertices)
    G = nx.Graph() # a NetworkX graph
    # Gathering the edges between every odd_vertices pair in the graph. Invert
    # edge weights to use max_weight_matching later.
    for i in range(n):
        for j in range(i + 1, n):
            u = odd_vertices[i]
            v = odd_vertices[j]
            edges.append([u, v, graph.graph[u][v] * -1])
    # Add edges to NetworkX graph.
    G.add_weighted_edges_from(edges)

    # Set `maxcardinality` to true in order to find perfect matching on even
    # vertices complete graph. `matching` is a set of (u, v)
    matching = nx.max_weight_matching(G, maxcardinality=True)

    for edge in matching:
        u = edge[0]
        v = edge[1]
        mwpm.append([u, v, graph.graph[u][v]])

    return mwpm


# Find the vertices of odd degree in the minimum spanning tree
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
    """
    Minimum Spanning Tree using Prim's Algorithm
    CLRS[23.2]
    Args:
        graph:  adjacency matrix represented Graph
        source: the root vertex used to generate Prim tree
    Returns:
        a list of edges (u, v, weight(u, v)) which represents a minimum
        spanning tree.
    """

    mst_prim = []
    visited = set()
    visited.add(source)
    # `parent` array stores the parent of each vertex in the MST
    parent = [None] * graph.num_nodes
    # `key` array stores the minimum weight of any edge connecting v to a vertex in the tree
    key = [math.inf] * graph.num_nodes

    key[source] = 0

    heap = []
    for dest in range(graph.num_nodes):
        if dest == source:
            continue
        else:
            parent[dest] = source
            key[dest] = graph.graph[source][dest]
            heap.append((key[dest], dest))
    # Heapify the heap to get a binary min-heap
    heapq.heapify(heap)

    while heap and len(visited) != graph.num_nodes:
            distance, dest = heapq.heappop(heap)
            if dest not in visited:
                visited.add(dest)
                mst_prim.append([parent[dest], dest, distance])
                for neighbor in range(graph.num_nodes):
                    if neighbor not in visited and graph.graph[dest][neighbor] < key[neighbor]:
                        parent[neighbor] = dest
                        key[neighbor] = graph.graph[dest][neighbor]
                        heapq.heappush(heap, (key[neighbor], neighbor))

    return mst_prim


# Read lines from the input_file. Generate name list and location list.
def read_input(input_file):
    city_name = []
    city_location = []
    # input file in format: [city_name],[city_location_x],[city_location_y]
    with open(input_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")
            city_name.append(line[0])
            city_location.append([float(line[1]), float(line[2])])

    return city_name, city_location


# Create undirected weighted graph from the input file.
def create_graph(input_file):
    city_name, city_location = read_input(input_file)
    graph = Graph()
    graph.build_graph(city_location)

    return graph


def tsp_christofides(graph):
    """
    An approximation algorithm to solve the metric travelling sales man problem.
    Args:
        graph: adjacency matrix represented Graph.
    Returns:
        a list of city name which is the tour generated by christofides
        algorithm and its cost.
    """

    # Find its minimum spanning tree.
    mst_prim = generate_minimum_spanning_tree_with_prim(graph, 0)
    print(f"mst prim: {mst_prim}")

    # Find odd_vertices.
    odd_vertices = find_odd_vertices(mst_prim)
    print(f"odd vertices: {odd_vertices}")

    # Find minimum weight perfect mathcing.
    mwpm = generate_minimum_weight_perfect_mathcing(graph, odd_vertices)
    print(f"mwpm: {mwpm}")

    # Combine mst and mwpm result to creat a multigraph.
    mst_prim.extend(mwpm)
    multigraph = mst_prim
    print(f"multigraph: {multigraph}")

    # Create Eulerian circuit.
    eulerian_circuit = generate_eulerian_circuit(multigraph, graph)
    print(f"eulerian_circuit: {eulerian_circuit}")

    # Generate Hamiltonian circuit.
    hamiltonian_circuit, cost = generate_hamiltonian_circuit(eulerian_circuit, graph)
    print(f"hamiltonian_circuit: {hamiltonian_circuit}\ntotal cost: {cost}")

    return hamiltonian_circuit, cost


if __name__ == "__main__":
    start_time = time.time()

    graph = create_graph("data_small.txt")

    tour, cost = tsp_christofides(graph)

    print("%s sconds run time for heapq" % (time.time() - start_time))



