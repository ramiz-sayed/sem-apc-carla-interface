import matplotlib.pyplot as plt
import numpy as np
import heapq
import matplotlib.patches as patches

# Grid size
rows, cols = 15, 15

# Start and goal positions
start = (0, 0)
goal = (8, 8)

# Obstacles
obstacles = {(3, i) for i in range(1, 9)}
obstacles |= {(i, 5) for i in range(5, 9)}

# A* Algorithm
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal, obstacles):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    directions = {
        (0, -1): 'L',
        (0, 1): 'R',
        (-1, 0): 'U',
        (1, 0): 'D'
    }

    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)

        if current == goal:
            break

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and neighbor not in obstacles):
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = (current, directions[(dx, dy)])
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))

    return came_from, g_score, f_score

came_from, g_score, f_score = a_star(start, goal, obstacles)

# Reconstruct path
path = []
current = goal
while current in came_from:
    path.append(current)
    current = came_from[current][0]
path.append(start)
path.reverse()

# Plotting
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_xticks(np.arange(0, cols+1, 1))
ax.set_yticks(np.arange(0, rows+1, 1))
ax.grid(True)
ax.set_aspect('equal')

# Draw obstacles
for (x, y) in obstacles:
    ax.add_patch(patches.Rectangle((y, rows-1-x), 1, 1, color='black'))

# Draw cells
for x in range(rows):
    for y in range(cols):
        pos = (x, y)
        if pos in obstacles:
            continue

        g = g_score.get(pos, None)
        h = heuristic(pos, goal)
        f = f_score.get(pos, None)
        if pos in came_from:
            direction = came_from[pos][1]
        elif pos == start:
            direction = 'S'
        elif pos == goal:
            direction = 'G'
        else:
            direction = ''

        label = f"g={g if g is not None else '-'}\nh={h}\nf={f if f is not None else '-'}\n←{direction}"
        ax.text(y + 0.5, rows - 1 - x + 0.5, label, va='center', ha='center', fontsize=7)

# Highlight path
for pos in path:
    x, y = pos
    ax.add_patch(patches.Rectangle((y, rows-1-x), 1, 1, fill=False, edgecolor='red', linewidth=2))

plt.title("A* Pathfinding with g, h, f values and directions")
plt.tight_layout()
plt.show()
