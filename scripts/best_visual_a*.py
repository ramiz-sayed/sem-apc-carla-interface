import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
from collections import defaultdict
import time

class AStarVisualizer:
    def __init__(self, grid_size=(10, 10), start=(0, 0), goal=(9, 9)):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = set()
        
        # A* algorithm data structures
        self.open_list = []
        self.closed_list = set()
        self.came_from = {}
        self.g_score = defaultdict(lambda: float('inf'))
        self.f_score = defaultdict(lambda: float('inf'))
        self.h_score = {}
        
        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(-0.5, grid_size[1] - 0.5)
        self.ax.set_ylim(-0.5, grid_size[0] - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # To match typical grid coordinates
        
        # Colors
        self.colors = {
            'obstacle': 'black',
            'start': 'blue',
            'goal': 'red',
            'open': 'lightblue',
            'closed': 'lightgray',
            'path': 'green'
        }
        
        # Direction arrows for came_from visualization
        self.direction_arrows = {
            (0, 1): '←',   # came from right
            (0, -1): '→',  # came from left
            (1, 0): '↑',   # came from below
            (-1, 0): '↓',  # came from above
            (1, 1): '↖',   # came from bottom-right
            (1, -1): '↗',  # came from bottom-left
            (-1, 1): '↙',  # came from top-right
            (-1, -1): '↘'  # came from top-left
        }
        
        # Text objects for displaying g, h, f values
        self.text_objects = {}
        
        self.setup_obstacles()
        self.draw_initial_grid()
        
    def setup_obstacles(self):
        """Define obstacles on the grid. Modify this method to change obstacle layout."""
        # Example obstacle pattern - you can modify this easily
        obstacles = [
            # Vertical wall
            (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
            # Horizontal wall
            (4, 7), (5, 7), (6, 7), (7, 7),
            # L-shaped obstacle
            (6, 2), (6, 3), (7, 3), (8, 3),
            # Random obstacles
            (1, 8), (3, 6), (8, 1), (9, 5)
        ]
        
        for obs in obstacles:
            if (obs != self.start and obs != self.goal and 
                0 <= obs[0] < self.grid_size[0] and 0 <= obs[1] < self.grid_size[1]):
                self.obstacles.add(obs)
    
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_neighbors(self, pos):
        """Get valid neighbors of a position (8-directional movement)."""
        neighbors = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in directions:
            new_pos = (pos[0] + dr, pos[1] + dc)
            if (0 <= new_pos[0] < self.grid_size[0] and 
                0 <= new_pos[1] < self.grid_size[1] and 
                new_pos not in self.obstacles):
                neighbors.append(new_pos)
        
        return neighbors
    
    def get_movement_cost(self, from_pos, to_pos):
        """Calculate movement cost between adjacent positions."""
        # Diagonal movement costs more than horizontal/vertical
        dr = abs(to_pos[0] - from_pos[0])
        dc = abs(to_pos[1] - from_pos[1])
        return 1.414 if (dr == 1 and dc == 1) else 1.0
    
    def draw_initial_grid(self):
        """Draw the initial grid with start, goal, and obstacles."""
        # Draw grid lines
        for i in range(self.grid_size[0] + 1):
            self.ax.axhline(y=i-0.5, color='gray', linewidth=0.5)
        for j in range(self.grid_size[1] + 1):
            self.ax.axvline(x=j-0.5, color='gray', linewidth=0.5)
        
        # Draw obstacles
        for obs in self.obstacles:
            rect = patches.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1, 
                                   linewidth=1, edgecolor='black', 
                                   facecolor=self.colors['obstacle'])
            self.ax.add_patch(rect)
        
        # Draw start and goal
        start_rect = patches.Rectangle((self.start[1]-0.5, self.start[0]-0.5), 1, 1,
                                     linewidth=2, edgecolor='blue',
                                     facecolor=self.colors['start'], alpha=0.7)
        self.ax.add_patch(start_rect)
        
        goal_rect = patches.Rectangle((self.goal[1]-0.5, self.goal[0]-0.5), 1, 1,
                                    linewidth=2, edgecolor='red',
                                    facecolor=self.colors['goal'], alpha=0.7)
        self.ax.add_patch(goal_rect)
        
        # Labels
        self.ax.text(self.start[1], self.start[0]-0.3, 'START', 
                    ha='center', va='center', fontweight='bold', fontsize=8)
        self.ax.text(self.goal[1], self.goal[0]-0.3, 'GOAL', 
                    ha='center', va='center', fontweight='bold', fontsize=8)
        
        self.ax.set_title('A* Pathfinding Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
    
    def update_cell_display(self, pos, cell_type='open'):
        """Update the visual display of a cell with g, h, f values and direction arrow."""
        if pos == self.start or pos == self.goal:
            return  # Don't override start/goal colors completely
        
        # Add colored background for open/closed cells
        if cell_type == 'open':
            rect = patches.Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1,
                                   linewidth=1, edgecolor='blue',
                                   facecolor=self.colors['open'], alpha=0.5)
        elif cell_type == 'closed':
            rect = patches.Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1,
                                   linewidth=1, edgecolor='gray',
                                   facecolor=self.colors['closed'], alpha=0.5)
        elif cell_type == 'path':
            rect = patches.Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1,
                                   linewidth=3, edgecolor='green',
                                   facecolor=self.colors['path'], alpha=0.7)
        
        self.ax.add_patch(rect)
        
        # Display g, h, f values
        g_val = self.g_score[pos]
        h_val = self.h_score.get(pos, 0)
        f_val = self.f_score[pos]
        
        # Remove old text if exists
        if pos in self.text_objects:
            for text_obj in self.text_objects[pos]:
                text_obj.remove()
        
        self.text_objects[pos] = []
        
        # Add g, h, f text
        if g_val != float('inf'):
            g_text = self.ax.text(pos[1]-0.3, pos[0]-0.1, f'g:{g_val:.1f}', 
                                ha='left', va='center', fontsize=7, fontweight='bold')
            h_text = self.ax.text(pos[1]-0.3, pos[0]+0.1, f'h:{h_val:.1f}', 
                                ha='left', va='center', fontsize=7, fontweight='bold')
            f_text = self.ax.text(pos[1]-0.3, pos[0]+0.3, f'f:{f_val:.1f}', 
                                ha='left', va='center', fontsize=7, fontweight='bold', color='red')
            
            self.text_objects[pos].extend([g_text, h_text, f_text])
        
        # Add direction arrow
        if pos in self.came_from and pos != self.start:
            parent = self.came_from[pos]
            direction = (parent[0] - pos[0], parent[1] - pos[1])
            if direction in self.direction_arrows:
                arrow_text = self.ax.text(pos[1]+0.3, pos[0], self.direction_arrows[direction],
                                        ha='center', va='center', fontsize=12, fontweight='bold')
                self.text_objects[pos].append(arrow_text)
    
    def reconstruct_path(self):
        """Reconstruct and return the optimal path."""
        path = []
        current = self.goal
        
        while current in self.came_from:
            path.append(current)
            current = self.came_from[current]
        path.append(self.start)
        
        return path[::-1]  # Reverse to get start-to-goal path
    
    def run_astar(self):
        """Run the A* algorithm with step-by-step visualization."""
        # Initialize
        self.g_score[self.start] = 0
        self.h_score[self.start] = self.manhattan_distance(self.start, self.goal)
        self.f_score[self.start] = self.h_score[self.start]
        
        heapq.heappush(self.open_list, (self.f_score[self.start], self.start))
        
        step_count = 0
        
        while self.open_list:
            step_count += 1
            
            # Get the node with lowest f_score
            current_f, current = heapq.heappop(self.open_list)
            
            if current in self.closed_list:
                continue
            
            # Add current to closed list and update visualization
            self.closed_list.add(current)
            self.update_cell_display(current, 'closed')
            
            # Update title with current step
            self.ax.set_title(f'A* Pathfinding - Step {step_count} - Exploring: {current}', 
                            fontsize=14, fontweight='bold')
            
            plt.pause(0.3)  # Animation delay
            
            # Check if we reached the goal
            if current == self.goal:
                print(f"Goal reached in {step_count} steps!")
                break
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_list:
                    continue
                
                # Calculate tentative g_score
                tentative_g = self.g_score[current] + self.get_movement_cost(current, neighbor)
                
                if tentative_g < self.g_score[neighbor]:
                    # This path is better
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g
                    self.h_score[neighbor] = self.manhattan_distance(neighbor, self.goal)
                    self.f_score[neighbor] = tentative_g + self.h_score[neighbor]
                    
                    # Add to open list if not already there
                    heapq.heappush(self.open_list, (self.f_score[neighbor], neighbor))
                    
                    # Update visualization for open nodes
                    if neighbor not in self.closed_list:
                        self.update_cell_display(neighbor, 'open')
        
        # Reconstruct and highlight the optimal path
        if self.goal in self.came_from or self.goal == self.start:
            path = self.reconstruct_path()
            total_cost = self.g_score[self.goal]
            
            print(f"Optimal path found!")
            print(f"Path: {' -> '.join(map(str, path))}")
            print(f"Total cost: {total_cost:.2f}")
            
            # Highlight the path
            for pos in path:
                if pos != self.start and pos != self.goal:
                    self.update_cell_display(pos, 'path')
            
            # Update title with final result
            self.ax.set_title(f'A* Pathfinding Complete - Optimal Path Cost: {total_cost:.2f}', 
                            fontsize=14, fontweight='bold')
            
            # Add cost text to the plot
            self.ax.text(0.02, 0.98, f'Total Path Cost: {total_cost:.2f}', 
                        transform=self.ax.transAxes, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                        verticalalignment='top')
            
        else:
            print("No path found!")
            self.ax.set_title('A* Pathfinding Complete - No Path Found!', 
                            fontsize=14, fontweight='bold', color='red')
        
        plt.pause(1)  # Final pause to show result

def main():
    """Main function to run the A* visualization."""
    # Configuration - easily adjustable parameters
    GRID_SIZE = (12, 15)  # (rows, columns)
    START_POS = (0, 0)    # (row, column)
    GOAL_POS = (11, 14)   # (row, column)
    
    print("Starting A* Pathfinding Visualization...")
    print(f"Grid size: {GRID_SIZE}")
    print(f"Start position: {START_POS}")
    print(f"Goal position: {GOAL_POS}")
    print("Close the plot window to exit.")
    
    # Create and run the visualizer
    visualizer = AStarVisualizer(GRID_SIZE, START_POS, GOAL_POS)
    
    # Start the algorithm after a short delay
    plt.pause(1)
    visualizer.run_astar()
    
    # Keep the plot open
    plt.show()

if __name__ == "__main__":
    main()
