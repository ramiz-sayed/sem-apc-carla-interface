import matplotlib.pyplot as plt
import heapq
import time
import numpy as np

# Settings
GRID_SIZE = (20, 30)  # rows, cols
OBSTACLES = [(5, i) for i in range(5, 25)] + [(10, i) for i in range(10, 20)] + [(15, i) for i in range(15, 25)]
START = (0, 0)
END = (19, 29)

# A* node
class Node:
    def __init__(self, pos, parent=None, g=0, h=0):
        self.pos = pos
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

# Heuristic: Manhattan distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Get neighbors
def get_neighbors(pos):
    x, y = pos
    directions = [(-1,0),(1,0),(0,-1),(0,1)]  # 4-directional
    result = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE[0] and 0 <= ny < GRID_SIZE[1]:
            result.append((nx, ny))
    return result

# Plotting function
def draw(grid, start, end, open_set, closed_set, path, pause=0.01):
    grid_copy = np.copy(grid)
    for node in closed_set:
        grid_copy[node[0], node[1]] = 0.5  # gray
    for node in open_set:
        grid_copy[node[0], node[1]] = 0.75  # light gray
    for node in path:
        grid_copy[node[0], node[1]] = 0.25  # dark path
    grid_copy[start[0], start[1]] = 0.7
    grid_copy[end[0], end[1]] = 0.9

    plt.imshow(grid_copy, cmap='viridis', origin='upper')
    plt.pause(pause)
    plt.clf()

# A* Algorithm
def a_star(start, end, obstacles):
    grid = np.ones(GRID_SIZE)
    for x, y in obstacles:
        grid[x, y] = 0  # wall

    open_heap = []
    open_dict = {}
    closed_set = set()

    start_node = Node(start, None, 0, heuristic(start, end))
    heapq.heappush(open_heap, start_node)
    open_dict[start] = start_node

    came_from = {}
    total_costs = {}

    while open_heap:
        current = heapq.heappop(open_heap)
        if current.pos in closed_set:
            continue
        closed_set.add(current.pos)

        if current.pos == end:
            path = []
            total = current.g
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], total, grid

        for neighbor in get_neighbors(current.pos):
            if neighbor in obstacles or neighbor in closed_set:
                continue

            g = current.g + 1
            h = heuristic(neighbor, end)
            node = Node(neighbor, current, g, h)

            if neighbor not in open_dict or g < open_dict[neighbor].g:
                heapq.heappush(open_heap, node)
                open_dict[neighbor] = node
                came_from[neighbor] = current.pos
                total_costs[neighbor] = g + h

        draw(grid, start, end, open_dict.keys(), closed_set, [], pause=0.01)

    return None, float('inf'), grid

# Main
def main():
    plt.figure(figsize=(12, 8))
    path, cost, grid = a_star(START, END, OBSTACLES)

    if path:
        print(f"✅ Path found! Total cost: {cost}")
    else:
        print("❌ No path found")

    # Final draw
    draw(grid, START, END, [], [], path, pause=0)
    plt.title(f"Final path with cost: {cost}")
    plt.show()

if __name__ == "__main__":
    main()
