import matplotlib.pyplot as plt  # For creating plots and visualizations
import random  # For random number generation
import time  # For adding delays in visualization
import numpy as np  # For numerical operations and arrays
import matplotlib.patches as patches  # For drawing rectangles and circles on the plot
import math  # For mathematical operations like distance calculation

# Grid and pathfinding settings
ROWS, COLS = 15, 15  # Define grid dimensions as 15x15
START = (0, 0)  # Starting position at bottom-left corner
GOAL = (0, 14)  # Goal position at top-right corner
STEP_SIZE = 1.0  # Maximum distance for each RRT expansion step
MAX_ITERATIONS = 1000  # Maximum number of iterations before giving up
GOAL_RADIUS = 1.0  # Radius around goal considered as "reached"

# Define obstacle positions as sets of coordinates
OBSTACLES = {(3, i) for i in range(1, 9)}  # Create horizontal obstacle line at row 3
OBSTACLES |= {(i, 5) for i in range(5, 9)}  # Add vertical obstacle line at column 5

# RRT Node class to represent each point in the tree
class RRTNode:
    def __init__(self, pos, parent=None):  # Initialize RRT node
        self.pos = pos  # Position (x, y) coordinates
        self.parent = parent  # Reference to parent node for path reconstruction
        self.children = []  # List of child nodes

# Calculate Euclidean distance between two points
def distance(p1, p2):  # Calculate distance between two points
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)  # Euclidean distance formula

# Check if a position is valid (within bounds and not an obstacle)
def is_valid_position(pos):  # Check if position is valid
    x, y = pos  # Extract coordinates
    # Check bounds
    if not (0 <= x < ROWS and 0 <= y < COLS):  # If outside grid boundaries
        return False  # Position is invalid
    # Check obstacles (with some tolerance for floating point positions)
    grid_x, grid_y = int(round(x)), int(round(y))  # Round to nearest grid position
    if (grid_x, grid_y) in OBSTACLES:  # If position is an obstacle
        return False  # Position is invalid
    return True  # Position is valid

# Check if path between two points is collision-free
def is_collision_free(p1, p2):  # Check if straight line path is clear
    steps = int(distance(p1, p2) * 10)  # Number of points to check along path
    if steps == 0:  # If points are the same
        return True  # No collision
    
    for i in range(steps + 1):  # Check points along the path
        t = i / steps  # Interpolation parameter (0 to 1)
        x = p1[0] + t * (p2[0] - p1[0])  # Interpolated x coordinate
        y = p1[1] + t * (p2[1] - p1[1])  # Interpolated y coordinate
        if not is_valid_position((x, y)):  # If any point along path is invalid
            return False  # Collision detected
    return True  # Path is collision-free

# Generate random position within grid bounds
def random_position():  # Generate random valid position
    max_attempts = 100  # Maximum attempts to find valid position
    for _ in range(max_attempts):  # Try multiple times
        x = random.uniform(0, ROWS - 1)  # Random x coordinate
        y = random.uniform(0, COLS - 1)  # Random y coordinate
        pos = (x, y)  # Create position tuple
        if is_valid_position(pos):  # If position is valid
            return pos  # Return valid position
    # If no valid position found, return a default safe position
    return (1.0, 1.0)  # Return safe default position

# Find nearest node in tree to given position
def find_nearest_node(tree, pos):  # Find closest node in tree
    min_dist = float('inf')  # Initialize minimum distance as infinity
    nearest_node = None  # Initialize nearest node as None
    
    for node in tree:  # Loop through all nodes in tree
        dist = distance(node.pos, pos)  # Calculate distance to current node
        if dist < min_dist:  # If this is the closest node so far
            min_dist = dist  # Update minimum distance
            nearest_node = node  # Update nearest node
    
    return nearest_node  # Return closest node

# Create new position by stepping from nearest node toward random position
def steer(from_pos, to_pos, step_size):  # Create new position with limited step size
    dist = distance(from_pos, to_pos)  # Calculate distance between positions
    if dist <= step_size:  # If target is within step size
        return to_pos  # Return target position
    
    # Calculate unit vector toward target
    dx = (to_pos[0] - from_pos[0]) / dist  # x component of unit vector
    dy = (to_pos[1] - from_pos[1]) / dist  # y component of unit vector
    
    # Calculate new position at step_size distance
    new_x = from_pos[0] + dx * step_size  # New x coordinate
    new_y = from_pos[1] + dy * step_size  # New y coordinate
    
    return (new_x, new_y)  # Return new position

# Convert position to coordinate label
def pos_to_label(pos):  # Convert (row, col) to "A1" format
    row, col = pos  # Extract row and column
    return f"{chr(65 + int(row))}{int(col) + 1}"  # Convert to letter-number format

# Initialize the matplotlib plot
def setup_plot():  # Create and configure the initial plot
    plt.ion()  # Turn on interactive mode for real-time updates
    fig, ax = plt.subplots(figsize=(15, 15))  # Create figure with 15x15 inch size
    ax.set_xlim(-0.5, COLS + 0.5)  # Set x-axis limits with padding
    ax.set_ylim(-0.5, ROWS + 0.5)  # Set y-axis limits with padding
    ax.set_xticks(np.arange(0, COLS + 1, 1))  # Set x-axis tick marks
    ax.set_yticks(np.arange(0, ROWS + 1, 1))  # Set y-axis tick marks
    ax.grid(True)  # Enable grid lines
    ax.set_title("RRT Pathfinding - Step 0")  # Set initial title
    ax.set_xlabel("Column")  # Label x-axis
    ax.set_ylabel("Row")  # Label y-axis
    
    # Add coordinate labels around the grid
    for i in range(ROWS):  # Loop through each row
        ax.text(-0.3, i + 0.5, f"{chr(65+i)}", ha='center', va='center', 
                fontsize=10, fontweight='bold')  # Add row labels
    for j in range(COLS):  # Loop through each column
        ax.text(j + 0.5, -0.3, f"{j+1}", ha='center', va='center', 
                fontsize=10, fontweight='bold')  # Add column labels
    
    return fig, ax  # Return figure and axis objects

# Draw the current state of the RRT
def draw_rrt(ax, tree, path, iteration, goal_found):
    ax.clear()  # Clear the previous drawing
    
    # Reconfigure the plot after clearing
    ax.set_xlim(-0.5, COLS + 0.5)  # Reset x-axis limits
    ax.set_ylim(-0.5, ROWS + 0.5)  # Reset y-axis limits
    ax.set_xticks(np.arange(0, COLS + 1, 1))  # Reset x-axis ticks
    ax.set_yticks(np.arange(0, ROWS + 1, 1))  # Reset y-axis ticks
    ax.grid(True)  # Re-enable grid
    status = "Goal Found!" if goal_found else "Searching..."  # Set status message
    ax.set_title(f"RRT Pathfinding - Iteration {iteration} - {status}")  # Update title
    ax.set_xlabel("Column")  # Re-add x-axis label
    ax.set_ylabel("Row")  # Re-add y-axis label
    
    # Re-add coordinate labels
    for i in range(ROWS):  # Add row labels again
        ax.text(-0.3, i + 0.5, f"{chr(65+i)}", ha='center', va='center', 
                fontsize=10, fontweight='bold')
    for j in range(COLS):  # Add column labels again
        ax.text(j + 0.5, -0.3, f"{j+1}", ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Draw background grid
    for x in range(ROWS):  # Loop through each row
        for y in range(COLS):  # Loop through each column
            ax.add_patch(patches.Rectangle((y, x), 1, 1, fill=True, 
                                         edgecolor='gray', facecolor='white', alpha=0.3))
    
    # Draw obstacles in black
    for (x, y) in OBSTACLES:  # Loop through obstacle positions
        ax.add_patch(patches.Rectangle((y, x), 1, 1, color='black'))
    
    # Draw start position in green
    ax.add_patch(patches.Circle((START[1] + 0.5, START[0] + 0.5), 0.3, color='green'))
    ax.text(START[1] + 0.5, START[0] + 0.5, "S", va='center', ha='center', 
            fontsize=12, color='white', weight='bold')
    
    # Draw goal position in blue with goal radius
    goal_circle = patches.Circle((GOAL[1] + 0.5, GOAL[0] + 0.5), GOAL_RADIUS, 
                                color='blue', alpha=0.3)  # Goal area
    ax.add_patch(goal_circle)
    ax.add_patch(patches.Circle((GOAL[1] + 0.5, GOAL[0] + 0.5), 0.3, color='blue'))
    ax.text(GOAL[1] + 0.5, GOAL[0] + 0.5, "G", va='center', ha='center', 
            fontsize=12, color='white', weight='bold')
    
    # Draw RRT tree edges
    for node in tree:  # Loop through all nodes in tree
        if node.parent:  # If node has a parent
            # Draw line from parent to current node
            ax.plot([node.parent.pos[1] + 0.5, node.pos[1] + 0.5], 
                   [node.parent.pos[0] + 0.5, node.pos[0] + 0.5], 
                   'b-', linewidth=1, alpha=0.6)  # Blue line for tree edges
    
    # Draw RRT nodes
    for node in tree:  # Loop through all nodes
        ax.plot(node.pos[1] + 0.5, node.pos[0] + 0.5, 'bo', markersize=3)  # Blue dots for nodes
    
    # Draw final path if found
    if path:  # If path exists
        for i in range(1, len(path)):  # Loop through path segments
            # Draw thick red line for path
            ax.plot([path[i-1].pos[1] + 0.5, path[i].pos[1] + 0.5], 
                   [path[i-1].pos[0] + 0.5, path[i].pos[0] + 0.5], 
                   'r-', linewidth=3)  # Red line for final path
        
        # Draw path nodes
        for node in path:  # Loop through path nodes
            ax.plot(node.pos[1] + 0.5, node.pos[0] + 0.5, 'ro', markersize=5)  # Red dots for path
    
    plt.draw()  # Update the display
    plt.pause(0.01)  # Short pause for animation

# Reconstruct path from goal node back to start
def reconstruct_path(goal_node):  # Build path from goal to start
    path = []  # Initialize empty path list
    current = goal_node  # Start from goal node
    
    while current:  # Follow parent chain back to start
        path.append(current)  # Add current node to path
        current = current.parent  # Move to parent node
    
    path.reverse()  # Reverse to get start-to-goal order
    return path  # Return complete path

# RRT Algorithm implementation
def rrt_pathfinding(start, goal, obstacles):
    fig, ax = setup_plot()  # Initialize the plot
    
    # Initialize RRT tree with start node
    start_node = RRTNode(start)  # Create start node
    tree = [start_node]  # Initialize tree with start node
    
    iteration = 0  # Iteration counter
    goal_node = None  # Will store goal node when found
    
    # Draw initial state
    draw_rrt(ax, tree, [], iteration, False)
    time.sleep(1)  # Initial delay
    
    # Main RRT loop
    for iteration in range(1, MAX_ITERATIONS + 1):  # Loop for maximum iterations
        # Generate random position (with bias toward goal)
        if random.random() < 0.1:  # 10% chance to sample goal directly
            rand_pos = goal  # Use goal position
        else:
            rand_pos = random_position()  # Generate random position
        
        # Find nearest node in tree
        nearest_node = find_nearest_node(tree, rand_pos)  # Find closest node
        
        # Steer from nearest node toward random position
        new_pos = steer(nearest_node.pos, rand_pos, STEP_SIZE)  # Create new position
        
        # Check if new position and path are valid
        if is_valid_position(new_pos) and is_collision_free(nearest_node.pos, new_pos):
            # Create new node and add to tree
            new_node = RRTNode(new_pos, nearest_node)  # Create new node
            nearest_node.children.append(new_node)  # Add as child to nearest node
            tree.append(new_node)  # Add to tree
            
            # Check if goal is reached
            if distance(new_pos, goal) <= GOAL_RADIUS:  # If within goal radius
                goal_node = new_node  # Store goal node
                print(f"Goal reached at iteration {iteration}!")  # Success message
                break  # Exit main loop
        
        # Update visualization every 10 iterations for performance
        if iteration % 10 == 0:  # Every 10th iteration
            draw_rrt(ax, tree, [], iteration, False)  # Update display
    
    # Reconstruct path if goal was found
    final_path = []  # Initialize empty path
    if goal_node:  # If goal was reached
        final_path = reconstruct_path(goal_node)  # Build path from goal to start
        print(f"Path found with {len(final_path)} nodes")  # Path info
    else:
        print("Goal not reached within maximum iterations")  # Failure message
    
    # Draw final result
    draw_rrt(ax, tree, final_path, iteration, goal_node is not None)
    
    plt.ioff()  # Turn off interactive mode
    
    return final_path, iteration, len(tree)  # Return path, iterations, and tree size

# Main execution function
def main():
    print("Starting RRT pathfinding algorithm...")  # Intro message
    print(f"Start: {START}")  # Show start position
    print(f"Goal: {GOAL}")  # Show goal position
    print(f"Step size: {STEP_SIZE}")  # Show step size
    print(f"Goal radius: {GOAL_RADIUS}")  # Show goal radius
    print(f"Max iterations: {MAX_ITERATIONS}")  # Show max iterations
    print()
    
    # Set random seed for reproducible results (optional)
    random.seed(42)  # Set seed for consistent results
    
    # Run RRT algorithm
    path, iterations, tree_size = rrt_pathfinding(START, GOAL, OBSTACLES)
    
    # Display results
    if path:  # If path was found
        print(f"✅ Path found in {iterations} iterations!")  # Success message
        print(f"Tree size: {tree_size} nodes")  # Tree statistics
        print(f"Path length: {len(path)} nodes")  # Path statistics
        
        # Calculate path distance
        total_distance = 0  # Initialize total distance
        for i in range(1, len(path)):  # Loop through path segments
            total_distance += distance(path[i-1].pos, path[i].pos)  # Add segment distance
        print(f"Total path distance: {total_distance:.2f}")  # Show total distance
        
        print("\nPath coordinates:")  # Header for path coordinates
        for i, node in enumerate(path):  # Loop through path nodes
            pos = node.pos  # Get node position
            label = pos_to_label(pos)  # Convert to coordinate label
            print(f"Step {i}: {label} ({pos[0]:.2f}, {pos[1]:.2f})")  # Print step info
    else:  # If no path found
        print(f"❌ No path found after {iterations} iterations")  # Failure message
        print(f"Tree size: {tree_size} nodes")  # Tree statistics
    
    plt.show()  # Keep plot window open

# Program entry point
if __name__ == "__main__":  # If script is run directly (not imported)
    main()  # Execute main function
