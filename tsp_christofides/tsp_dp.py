# -*- coding: utf-8 -*-
"""
Author: Zhou
Implementing TSP by DP method
"""

NUM_CITY = 5
BIN_NUM_CITY = 1 << (NUM_CITY - 1)

# dp[i][V]: i to s crossing all vs in V only one time
dp = [[0 for _ in range(BIN_NUM_CITY)] for _ in range(NUM_CITY)]
path = []
graph = [[0, 3, float('inf'), 8, 9],
         [3, 0, 3, 10, 5],
         [float('inf'), 3, 0, 4, 3],
         [8, 10, 4, 0, 20],
         [9, 5, 3, 20, 0]]


def tsp():
    """
    Find the tsp answer by DP method
    :return: No return but update the dp array. The final answer is dp[0][BIN_NUM_CITY - 1]
    """
    # Base case: initialize dp[i][0]
    for i in range(NUM_CITY):
        dp[i][0] = graph[i][0]

    # Update/Relax edge by recurrence relationship
    for j in range(1, BIN_NUM_CITY):
        for i in range(NUM_CITY):
            dp[i][j] = float('inf')
            # Check whether V contains i
            if i > 0 and ((j >> (i - 1)) & 1) == 1:
                continue
            for k in range(1, NUM_CITY):
                if ((j >> (k - 1)) & 1) == 0:
                    continue
                if dp[i][j] > graph[i][k] + dp[k][j ^ (1 << (k - 1))]:
                    dp[i][j] = graph[i][k] + dp[k][j ^ (1 << (k - 1))]


def is_visited(visited):
    """
    Check whether all cities are visited.
    Helper function for get_path().
    """
    for i in range(1, NUM_CITY):
        if not visited[i]:
            return False
    return True


def get_path():
    """
    Get the TSP path.
    :return: Update the path array.
    """
    visited = [False for _ in range(NUM_CITY)]
    pioneer = 0
    minimum = float('inf')
    s = BIN_NUM_CITY - 1
    temp = 0
    path.append(0)
    while not is_visited(visited):
        for i in range(1, NUM_CITY):
            if not visited[i] and (s & (1 << (i - 1)) != 0):
                if minimum > graph[i][pioneer] + dp[i][s ^ (1 << (i - 1))]:
                    minimum = graph[i][pioneer] + dp[i][s ^ (1 << (i - 1))]
                    temp = i
        pioneer = temp
        path.append(pioneer)
        visited[pioneer] = True
        s = s ^ (1 << (pioneer - 1))
        minimum = float("inf")


def print_path():
    print("Minimum path: ")
    for element in path:
        print(element, end = " ---> ")
    print("0")


if __name__ == "__main__":
    tsp()
    print("Minimum value: ", dp[0][BIN_NUM_CITY - 1])
    get_path()
    print_path()