# -*- coding: utf-8 -*-
"""
Data Preprocessing:
Read in data file and build graph
Author: Zhou Liu
"""


def euclidean_distance(p1, p2):
    """
    Calculate the euclidean distance between 2 points
    :param p1: x, y coordinate of point 1
    :param p2: x, y coordinate of point 2
    :return: the euclidean distance between p1 and p2
    """
    ans = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1.0 / 2.0)
    return round(ans, 2)


def read_in_input(filename):
    """
    Read in data file and output the data in dictionary format
    :param filename: data file name
    :return: dict of {city: [x, y]}
    """
    city_locations = dict()
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            city, loc_x, loc_y = line.strip().split(",")
            city_locations[int(city)] = [float(loc_x), float(loc_y)]
    return city_locations


def build_graph(data_file):
    """
    Build graph by the given data file
    :param data_file: data file name
    :return: a 2d array representing the graph's adjacent matrix
    """
    locations = read_in_input(data_file)
    n = len(locations)
    graph = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            graph[i][j] = euclidean_distance(locations[i], locations[j])
    return graph


# if __name__ == "__main__":
#     # print(read_in_input("data_small.txt"))
#     print(build_graph("data_small.txt"))
