import matplotlib.pyplot as plt  # For creating plots and visualizations
import heapq  # For priority queue implementation (min-heap)
import time  # For adding delays in visualization
import numpy as np  # For numerical operations and arrays
import matplotlib.patches as patches  # For drawing rectangles on the plot
from collections import defaultdict  # For default dictionary (not used in this code)

# Grid and pathfinding settings
ROWS, COLS = 15, 15  # Define grid dimensions as 15x15
START = (0, 0)  # Starting position at bottom-left corner
GOAL = (0, 14)  # Goal position at top-right corner

# Define obstacle positions as sets of coordinates
OBSTACLES = {(3, i) for i in range(1, 9)}  # Create horizontal obstacle line at row 3
OBSTACLES |= {(i, 5) for i in range(5, 9)}  # Add vertical obstacle line at column 5

# Movement directions: up, down, left, right (4-directional movement)
DIRECTIONS = [
    (0, -1),  # Move left (decrease column)
    (0, 1),   # Move right (increase column)
    (-1, 0),  # Move up (decrease row)
    (1, 0)    # Move down (increase row)
]

# Node class to represent each position in the search space
class Node:
    def __init__(self, pos, parent=None, g=0, h=0):  # Initialize node with position and costs
        self.pos = pos      # Current position (x, y) coordinates
        self.parent = parent  # Reference to parent node for path reconstruction
        self.g = g          # Cost from start to current node
        self.h = h          # Heuristic cost from current node to goal
        self.f = g + h      # Total cost (f = g + h)

    def __lt__(self, other):  # Less than comparison for heap operations
        return self.f < other.f  # Compare nodes based on total cost f

# Manhattan distance heuristic function
def heuristic(a, b):  # Calculate Manhattan distance between two points
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Sum of absolute differences in x and y

# Convert grid position to readable coordinate label
def pos_to_label(pos):  # Convert (row, col) to "A1" format
    row, col = pos  # Extract row and column from position tuple
    return f"{chr(65 + row)}{col + 1}"  # Convert to letter-number format (A1, B2, etc.)

# Get valid neighboring positions
def get_neighbors(pos):  # Find all valid adjacent positions
    x, y = pos  # Extract current x, y coordinates
    result = []  # Initialize empty list for valid neighbors
    for dx, dy in DIRECTIONS:  # Check each possible movement direction
        nx, ny = x + dx, y + dy  # Calculate new position after movement
        if 0 <= nx < ROWS and 0 <= ny < COLS:  # Check if new position is within grid bounds
            result.append((nx, ny))  # Add valid neighbor to result list
    return result  # Return list of valid neighboring positions

# Initialize the matplotlib plot
def setup_plot():  # Create and configure the initial plot
    plt.ion()  # Turn on interactive mode for real-time updates
    fig, ax = plt.subplots(figsize=(15, 15))  # Create figure with 15x15 inch size
    ax.set_xlim(-0.5, COLS + 0.5)  # Set x-axis limits with padding
    ax.set_ylim(-0.5, ROWS + 0.5)  # Set y-axis limits with padding
    ax.set_xticks(np.arange(0, COLS + 1, 1))  # Set x-axis tick marks at integer positions
    ax.set_yticks(np.arange(0, ROWS + 1, 1))  # Set y-axis tick marks at integer positions
    ax.grid(True)  # Enable grid lines for better visualization
    ax.set_title("A* Pathfinding - Step 0")  # Set initial title
    ax.set_xlabel("Column")  # Label x-axis
    ax.set_ylabel("Row")  # Label y-axis
    
    # Add coordinate labels around the grid
    for i in range(ROWS):  # Loop through each row
        ax.text(-0.3, i + 0.5, f"{chr(65+i)}", ha='center', va='center',   # Add row labels (A, B, C...)
                fontsize=10, fontweight='bold')  # Set font properties
    for j in range(COLS):  # Loop through each column
        ax.text(j + 0.5, -0.3, f"{j+1}", ha='center', va='center',   # Add column labels (1, 2, 3...)
                fontsize=10, fontweight='bold')  # Set font properties
    
    return fig, ax  # Return figure and axis objects

# Draw the current state of the grid
def draw_grid(ax, all_nodes, open_dict, closed_set, current, path, step, cost):
    ax.clear()  # Clear the previous drawing
    
    # Reconfigure the plot after clearing
    ax.set_xlim(-0.5, COLS + 0.5)  # Reset x-axis limits
    ax.set_ylim(-0.5, ROWS + 0.5)  # Reset y-axis limits
    ax.set_xticks(np.arange(0, COLS + 1, 1))  # Reset x-axis ticks
    ax.set_yticks(np.arange(0, ROWS + 1, 1))  # Reset y-axis ticks
    ax.grid(True)  # Re-enable grid
    ax.set_title(f"A* Pathfinding - Step {step} (Cost: {cost})")  # Update title with current step
    ax.set_xlabel("Column")  # Re-add x-axis label
    ax.set_ylabel("Row")  # Re-add y-axis label
    
    # Re-add coordinate labels after clearing
    for i in range(ROWS):  # Add row labels again
        ax.text(-0.3, i + 0.5, f"{chr(65+i)}", ha='center', va='center', 
                fontsize=10, fontweight='bold')
    for j in range(COLS):  # Add column labels again
        ax.text(j + 0.5, -0.3, f"{j+1}", ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Draw background grid cells
    for x in range(ROWS):  # Loop through each row
        for y in range(COLS):  # Loop through each column
            ax.add_patch(patches.Rectangle((y, x), 1, 1, fill=True,   # Draw white background cell
                                         edgecolor='gray', facecolor='white', alpha=0.3))
    
    # Draw obstacle cells in black
    for (x, y) in OBSTACLES:  # Loop through each obstacle position
        ax.add_patch(patches.Rectangle((y, x), 1, 1, color='black'))  # Draw black obstacle cell
    
    # Draw start position in green and goal position in blue
    ax.add_patch(patches.Rectangle((START[1], START[0]), 1, 1, color='green'))  # Green start cell
    ax.add_patch(patches.Rectangle((GOAL[1], GOAL[0]), 1, 1, color='blue'))  # Blue goal cell
    ax.text(START[1] + 0.5, START[0] + 0.5, "S", va='center', ha='center',   # Add "S" text to start
            fontsize=12, color='white', weight='bold')
    ax.text(GOAL[1] + 0.5, GOAL[0] + 0.5, "G", va='center', ha='center',   # Add "G" text to goal
            fontsize=12, color='white', weight='bold')
    
    # Draw all processed nodes with their information
    for pos, node in all_nodes.items():  # Loop through all nodes that have been processed
        if pos == START or pos == GOAL:  # Skip start and goal positions
            continue  # They are already drawn with special colors
            
        # Determine cell color based on node status
        if pos in path:  # If position is part of the final path
            color = 'red'  # Color it red
            alpha = 0.5  # Semi-transparent
        elif pos == current:  # If this is the currently processing node
            color = 'orange'  # Color it orange
            alpha = 0.7  # More opaque
        elif pos in open_dict:  # If node is in open set (to be processed)
            color = 'yellow'  # Color it yellow
            alpha = 0.5  # Semi-transparent
        elif pos in closed_set:  # If node has been processed
            color = 'lightgray'  # Color it light gray
            alpha = 0.5  # Semi-transparent
        else:  # Default case
            color = 'white'  # Keep it white
            alpha = 1.0  # Fully opaque
        
        # Draw the colored cell
        ax.add_patch(patches.Rectangle((pos[1], pos[0]), 1, 1,   # Draw rectangle at position
                                      facecolor=color, alpha=alpha))  # With determined color
        
        # Display the 4 data points in each cell
        ax.text(pos[1] + 0.15, pos[0] + 0.85, f"g:{node.g}",   # Top-left: g value (cost from start)
                ha='left', va='top', fontsize=8, weight='bold')
        
        ax.text(pos[1] + 0.85, pos[0] + 0.85, f"h:{node.h}",   # Top-right: h value (heuristic to goal)
                ha='right', va='top', fontsize=8, weight='bold')
        
        ax.text(pos[1] + 0.15, pos[0] + 0.15, f"f:{node.f}",   # Bottom-left: f value (total cost)
                ha='left', va='bottom', fontsize=8, weight='bold')
        
        # Bottom-right: parent coordinate (where this node came from)
        if node.parent:  # If node has a parent
            parent_label = pos_to_label(node.parent.pos)  # Convert parent position to label
            ax.text(pos[1] + 0.85, pos[0] + 0.15, parent_label,   # Display parent label
                    ha='right', va='bottom', fontsize=8, weight='bold', color='blue')
    
    # Draw arrows showing the path
    for i in range(1, len(path)):  # Loop through path positions (skip first)
        start_pos = path[i-1]  # Previous position in path
        end_pos = path[i]  # Current position in path
        ax.annotate("", xy=(end_pos[1] + 0.5, end_pos[0] + 0.5),   # Draw arrow to current position
                   xytext=(start_pos[1] + 0.5, start_pos[0] + 0.5),  # From previous position
                   arrowprops=dict(arrowstyle="->", color="red", lw=2))  # Red arrow style
    
    plt.draw()  # Update the display
    plt.pause(0.0001)  # Very short pause for animation effect

# A* Algorithm implementation with visualization
def a_star_visualized(start, goal, obstacles):
    fig, ax = setup_plot()  # Initialize the plot
    
    # Create grid representation (1 = free, 0 = obstacle)
    grid = np.ones((ROWS, COLS))  # Initialize grid with all free spaces
    for x, y in obstacles:  # Loop through obstacle positions
        grid[x, y] = 0  # Mark obstacle positions as blocked

    # Initialize A* data structures
    open_heap = []  # Priority queue for nodes to be processed
    open_dict = {}  # Dictionary for quick lookup of open nodes
    closed_set = set()  # Set of already processed node positions
    step_count = 0  # Counter for algorithm steps
    current = None  # Currently processing node
    path = []  # Final path (initially empty)
    
    all_nodes = {}  # Store all nodes that have been created
    best_cost = {}  # Track the best cost to reach each position

    # Create and add start node
    start_node = Node(start, None, 0, heuristic(start, goal))  # Create start node with g=0
    heapq.heappush(open_heap, start_node)  # Add to priority queue
    open_dict[start] = start_node  # Add to open dictionary
    all_nodes[start] = start_node  # Add to all nodes
    best_cost[start] = 0  # Set best cost to start as 0

    # Draw initial state
    draw_grid(ax, all_nodes, open_dict, closed_set, current, path, step_count, 0)
    time.sleep(0.5)  # Wait before starting algorithm

    final_path = None  # Will store the final optimal path
    min_cost = float('inf')  # Track minimum cost found

    # Main A* loop
    while open_heap:  # Continue while there are nodes to process
        current = heapq.heappop(open_heap)  # Get node with lowest f value
        step_count += 1  # Increment step counter
        
        # Skip if we've already found a better path to this position
        if current.pos in closed_set and current.g > best_cost.get(current.pos, float('inf')):
            continue  # Skip this node
            
        closed_set.add(current.pos)  # Mark current position as processed
        all_nodes[current.pos] = current  # Update node information
        best_cost[current.pos] = min(best_cost.get(current.pos, float('inf')), current.g)  # Update best cost
        
        # Check if we reached the goal
        if current.pos == goal:  # If current position is the goal
            if current.g < min_cost:  # If this path is better than previous
                min_cost = current.g  # Update minimum cost
                # Reconstruct the path by following parent pointers
                temp_path = []  # Temporary path list
                temp = current  # Start from goal node
                while temp:  # Follow parent chain back to start
                    temp_path.append(temp.pos)  # Add position to path
                    temp = temp.parent  # Move to parent node
                temp_path.reverse()  # Reverse to get start-to-goal order
                final_path = temp_path  # Store as final path
            continue  # Continue searching for potentially better paths

        # Explore neighbors of current node
        for neighbor in get_neighbors(current.pos):  # Get all valid neighbors
            nx, ny = neighbor  # Extract neighbor coordinates
            if grid[nx, ny] == 0:  # If neighbor is an obstacle
                continue  # Skip this neighbor

            g = current.g + 1  # Calculate cost to reach neighbor (assuming cost 1 per step)
            h = heuristic(neighbor, goal)  # Calculate heuristic cost to goal
            
            node = Node(neighbor, current, g, h)  # Create new node for neighbor
            
            # Add neighbor to open set if it's a good candidate
            if neighbor not in best_cost or g <= best_cost[neighbor] + 1:  # If path is competitive
                heapq.heappush(open_heap, node)  # Add to priority queue
                open_dict[neighbor] = node  # Add to open dictionary
                all_nodes[neighbor] = node  # Store node information

        # Update visualization periodically
        if step_count % 2 == 0:  # Only draw every 2nd step for performance
            current_path = final_path if final_path else []  # Use final path if found
            draw_grid(ax, all_nodes, open_dict, closed_set, current.pos, current_path, 
                     step_count, current.g)  # Update display
        
        # Remove current node from open set
        if current.pos in open_dict and open_dict[current.pos] == current:
            del open_dict[current.pos]  # Remove from open dictionary

    # Draw final result
    if final_path:  # If a path was found
        draw_grid(ax, all_nodes, {}, closed_set, None, final_path, step_count, min_cost)  # Show final state
    
    plt.ioff()  # Turn off interactive mode
    
    return final_path, step_count  # Return path and number of steps

# Main execution function
def main():
    print("Starting A* pathfinding with g, h, f values and parent coordinates...")  # Intro message
    print("Each cell shows:")  # Explain display format
    print("- Top-left: g (cost from start)")
    print("- Top-right: h (heuristic to goal)")
    print("- Bottom-left: f (total cost)")
    print("- Bottom-right: parent coordinate (where it came from)")
    print()
    
    path, steps = a_star_visualized(START, GOAL, OBSTACLES)  # Run A* algorithm
    
    if path:  # If path was found
        cost = len(path) - 1  # Calculate path cost (number of steps)
        print(f"✅ Optimal path found in {steps} steps! Total cost: {cost}")  # Success message
        print("Path coordinates:")  # List path coordinates
        for i, pos in enumerate(path):  # Loop through path positions
            row_label = chr(65 + pos[0])  # Convert row to letter
            col_label = pos[1] + 1  # Convert column to 1-based number
            print(f"Step {i}: {row_label}{col_label} ({pos})")  # Print step information
    else:  # If no path found
        print("❌ No path found")  # Failure message
    
    plt.show()  # Keep plot window open

# Program entry point
if __name__ == "__main__":  # If script is run directly (not imported)
    main()  # Execute main function
