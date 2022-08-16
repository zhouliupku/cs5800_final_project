# -*- coding: utf-8 -*-
"""
Author: Zhou Liu
Section 5: All Pairs Shortest Path Problem
By Floyd-Warshall Algorithm
"""


def floyd_warshall(graph):
    """
    A DP method to calculate all pairs shortest path
    :param graph: a n*n 2d array recording original distance between every 2 pairs
    :return: a 2d array recording all pairs shortest path
    """
    # deep copy the given graph as the initial dp array
    dp = [row[:] for row in graph]
    n = len(graph)
    # try every possible intermediate point to relax the edge
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dp[i][j] > dp[i][k] + dp[k][j]:
                    dp[i][j] = dp[i][k] + dp[k][j]

    return dp


if __name__ == "__main__":
    # TODO: More test cases and I/O
    graph1 = [[0, 2, 6, 4],
              [float('inf'), 0, 3, float('inf')],
              [7, float('inf'), 0, 1],
              [5, float('inf'), 12, 0]]
    ans1 = floyd_warshall(graph1)
    print(ans1)
