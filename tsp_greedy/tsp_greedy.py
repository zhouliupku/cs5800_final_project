"""
Travelling Sales Man Problem With Visualization
Solve by Greedy Algorithm
Time complexity (Without visualization): O(n^2)
Space complexity (Without visualization): O(n^2)
Author: Jifan Xie
Date: Augest 16, 2022
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time

start = time.perf_counter()
# Preprocess spots list, source = line 0
N = 0
spots = []
with open("data.txt", "r")as f:
    lines = f.readlines()
    N = len(lines)
    for line in lines:
        line = line.strip().split(",")
        spots.append([float(line[1]), float(line[2])])
spots = np.array(spots)

# Build graph
graph = [[-1 for _ in range(N)] for _ in range(N)]
for i in range(N):
    for j in range(N):
        if i == j:
            graph[i][j] = 0
        else:
            graph[i][j] = math.sqrt((spots[i, 0] - spots[j, 0]) ** 2 + (spots[i, 1] - spots[j, 1]) ** 2)

def tsp():
    path = [0] # source = line 0
    cost = 0
    visited = [0 for _ in range(N)]
    visited[0] = 1
    curr = 0
    for k in range(N - 1):
        next = -1
        next_dis = float('inf')
        for i in range(1, N):
            if visited[i] == 0 and graph[curr][i] < next_dis:
                next_dis = graph[curr][i]
                next = i
        path.append(next)
        visited[next] = 1
        cost += next_dis
        curr = next
    path.append(0)
    return path, cost

path, cost = tsp()
print(path)
print("Cost =", cost)

end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))

# Visualization
def show(path):
    # Show spots
    for item in range(N):
        if item == path[0]:
            plt.plot(spots[item, 0], spots[item, 1], "o", color = 'r')
        else:
            plt.plot(spots[item, 0], spots[item, 1], "o", color = 'b')
        plt.text(spots[item, 0] + 0.1, spots[item, 1] + 0.1, item) 
    # Show path
    for item in range(1, len(path)):
        x = []
        y = []
        x.append(spots[path[item - 1], 0])
        y.append(spots[path[item - 1], 1])
        x.append(spots[path[item], 0])
        y.append(spots[path[item], 1])
        plt.plot(x, y, "-", color = 'k')
    plt.show()

show(path)
