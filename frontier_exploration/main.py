import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
from scipy.ndimage import binary_dilation
from collections import deque
import heapq

class FrontierExplorer:
    def __init__(self, map_size=50, sensor_range=5):
        self.map_size = map_size
        self.sensor_range = sensor_range
        
        # Map states: 0=unknown, 1=free, 2=obstacle
        self.map = np.zeros((map_size, map_size))
        
        # Ground truth environment (for simulating sensor data)
        self.ground_truth = self._generate_environment()
        
        # Robot initial position
        self.robot_pos = np.array([map_size//2, map_size//2])
        self.path_history = [self.robot_pos.copy()]
        
        # Current path and frontier points
        self.current_path = []
        self.frontiers = []
        self.target_frontier = None
        
        # Algorithm state flag
        self.current_stage = "Initialization"
        
    def _generate_environment(self):
        """Generate random obstacle environment"""
        env = np.ones((self.map_size, self.map_size))
        
        # Add boundaries
        env[0, :] = 2
        env[-1, :] = 2
        env[:, 0] = 2
        env[:, -1] = 2
        
        # Add random obstacles
        np.random.seed(42)
        num_obstacles = 15
        for _ in range(num_obstacles):
            x = np.random.randint(5, self.map_size-5)
            y = np.random.randint(5, self.map_size-5)
            w = np.random.randint(2, 6)
            h = np.random.randint(2, 6)
            env[max(0,y):min(self.map_size,y+h), 
                max(0,x):min(self.map_size,x+w)] = 2
        
        return env
    
    def sense_environment(self):
        """Step 1: Environment sensing and map building - simulate sensor scanning"""
        self.current_stage = "Step 1: Environment Sensing and Mapping"
        x, y = self.robot_pos
        
        # Update map within sensor range
        for i in range(max(0, x-self.sensor_range), 
                      min(self.map_size, x+self.sensor_range+1)):
            for j in range(max(0, y-self.sensor_range), 
                          min(self.map_size, y+self.sensor_range+1)):
                dist = np.sqrt((i-x)**2 + (j-y)**2)
                if dist <= self.sensor_range:
                    self.map[i, j] = self.ground_truth[i, j]
    
    def detect_frontiers(self):
        """Step 2: Frontier region detection"""
        self.current_stage = "Step 2: Frontier Detection"
        self.frontiers = []
        
        # Find boundaries between known free areas and unknown areas
        for i in range(1, self.map_size-1):
            for j in range(1, self.map_size-1):
                if self.map[i, j] == 1:  # Known free
                    # Check if there are unknown areas around
                    neighbors = [
                        self.map[i-1, j], self.map[i+1, j],
                        self.map[i, j-1], self.map[i, j+1]
                    ]
                    if 0 in neighbors:  # Adjacent to unknown area
                        self.frontiers.append((i, j))
        
        # Cluster frontier points (simplified version)
        self.frontiers = self._cluster_frontiers(self.frontiers)
        
    def _cluster_frontiers(self, frontiers, min_size=3):
        """Cluster frontier points and filter small clusters"""
        if not frontiers:
            return []
        
        visited = set()
        clusters = []
        
        for point in frontiers:
            if point in visited:
                continue
            
            cluster = []
            queue = deque([point])
            
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                
                # Find adjacent frontier points
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    neighbor = (current[0]+dx, current[1]+dy)
                    if neighbor in frontiers and neighbor not in visited:
                        queue.append(neighbor)
            
            if len(cluster) >= min_size:
                clusters.append(cluster)
        
        # Return center point of each cluster
        return [self._cluster_center(c) for c in clusters]
    
    def _cluster_center(self, cluster):
        """Calculate cluster center point"""
        return tuple(np.mean(cluster, axis=0).astype(int))
    
    def select_target_and_plan_path(self):
        """Step 3: Target point selection and path planning"""
        self.current_stage = "Step 3: Target Selection and Path Planning"
        
        if not self.frontiers:
            self.current_path = []
            self.target_frontier = None
            return False
        
        # Select the nearest frontier point
        distances = [np.linalg.norm(np.array(f) - self.robot_pos) 
                    for f in self.frontiers]
        target_idx = np.argmin(distances)
        self.target_frontier = self.frontiers[target_idx]
        
        # A* path planning
        self.current_path = self._astar_planning(
            tuple(self.robot_pos), 
            self.target_frontier
        )
        
        return len(self.current_path) > 0
    
    def _astar_planning(self, start, goal):
        """A* path planning algorithm"""
        def heuristic(a, b):
            return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), 
                          (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0]+dx, current[1]+dy)
                
                if (0 <= neighbor[0] < self.map_size and 
                    0 <= neighbor[1] < self.map_size and
                    self.map[neighbor] != 2):  # Not obstacle
                    
                    tentative_g = g_score[current] + np.sqrt(dx**2 + dy**2)
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
        
        return []
    
    def move_robot(self):
        """Move robot along the path"""
        if self.current_path:
            next_pos = self.current_path.pop(0)
            self.robot_pos = np.array(next_pos)
            self.path_history.append(self.robot_pos.copy())
            return True
        return False
    
    def is_exploration_complete(self):
        """Check if exploration is complete"""
        return len(self.frontiers) == 0 or len(self.current_path) == 0

# Visualization class
class ExplorationVisualizer:
    def __init__(self, explorer):
        self.explorer = explorer
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 7))
        self.fig.suptitle('Frontier Exploration Algorithm Demonstration', 
                         fontsize=16, fontweight='bold')
        
        self.step_count = 0
        self.exploration_complete = False
        
    def init_plot(self):
        """Initialize plot"""
        for ax in self.axes:
            ax.clear()
        
        # Left plot: Ground truth
        ax1 = self.axes[0]
        ax1.imshow(self.explorer.ground_truth, cmap='gray_r', 
                  origin='lower', vmin=0, vmax=2)
        ax1.set_title('Ground Truth Environment', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Robot view
        ax2 = self.axes[1]
        self._draw_robot_view(ax2)
        
        plt.tight_layout()
        
    def _draw_robot_view(self, ax):
        """Draw map from robot's perspective"""
        # Create color map
        display_map = np.ones((self.explorer.map_size, 
                              self.explorer.map_size, 3)) * 0.5  # Gray (unknown)
        
        # Known free areas (white)
        display_map[self.explorer.map == 1] = [1, 1, 1]
        
        # Obstacles (black)
        display_map[self.explorer.map == 2] = [0, 0, 0]
        
        ax.imshow(display_map, origin='lower')
        
        # Draw frontier points
        if self.explorer.frontiers:
            frontiers_array = np.array(self.explorer.frontiers)
            ax.scatter(frontiers_array[:, 1], frontiers_array[:, 0], 
                      c='yellow', s=100, marker='*', 
                      edgecolors='orange', linewidths=2,
                      label='Frontiers', zorder=5)
        
        # Draw target frontier
        if self.explorer.target_frontier:
            ax.scatter(self.explorer.target_frontier[1], 
                      self.explorer.target_frontier[0],
                      c='red', s=200, marker='*', 
                      edgecolors='darkred', linewidths=2,
                      label='Target Frontier', zorder=6)
        
        # Draw path
        if self.explorer.current_path:
            path_array = np.array(self.explorer.current_path)
            ax.plot(path_array[:, 1], path_array[:, 0], 
                   'b--', linewidth=2, label='Planned Path', alpha=0.7)
        
        # Draw history trajectory
        if len(self.explorer.path_history) > 1:
            history = np.array(self.explorer.path_history)
            ax.plot(history[:, 1], history[:, 0], 
                   'g-', linewidth=2, label='History Path', alpha=0.5)
        
        # Draw robot
        robot_circle = Circle((self.explorer.robot_pos[1], 
                              self.explorer.robot_pos[0]),
                             0.8, color='green', zorder=10)
        ax.add_patch(robot_circle)
        
        # Draw sensor range
        sensor_circle = Circle((self.explorer.robot_pos[1], 
                               self.explorer.robot_pos[0]),
                              self.explorer.sensor_range, 
                              fill=False, edgecolor='cyan', 
                              linewidth=2, linestyle='--', 
                              alpha=0.5, label='Sensor Range')
        ax.add_patch(sensor_circle)
        
        ax.set_title(f'Robot View - {self.explorer.current_stage}', 
                    fontsize=12, fontweight='bold', color='red')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, self.explorer.map_size)
        ax.set_ylim(-1, self.explorer.map_size)
        
        # Add statistics
        explored_ratio = np.sum(self.explorer.map > 0) / (self.explorer.map_size ** 2) * 100
        info_text = f'Step: {self.step_count}\nExplored: {explored_ratio:.1f}%\nFrontiers: {len(self.explorer.frontiers)}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def update(self, frame):
        """Animation update function"""
        if self.exploration_complete:
            return
        
        # Execute exploration loop
        if not self.explorer.current_path:
            # Sense environment
            self.explorer.sense_environment()
            self.step_count += 1
            
            # Detect frontiers
            self.explorer.detect_frontiers()
            
            # Select target and plan path
            if not self.explorer.select_target_and_plan_path():
                self.exploration_complete = True
                self.explorer.current_stage = "Exploration Complete!"
                print(f"\nExploration Complete! Total steps: {self.step_count}")
        else:
            # Move robot
            self.explorer.move_robot()
            self.explorer.sense_environment()
            self.step_count += 1
        
        # Update display
        for ax in self.axes:
            ax.clear()
        
        # Redraw left plot
        self.axes[0].imshow(self.explorer.ground_truth, cmap='gray_r', 
                           origin='lower', vmin=0, vmax=2)
        self.axes[0].set_title('Ground Truth Environment', 
                              fontsize=12, fontweight='bold')
        self.axes[0].grid(True, alpha=0.3)
        
        # Show robot position on ground truth
        self.axes[0].scatter(self.explorer.robot_pos[1], 
                            self.explorer.robot_pos[0],
                            c='green', s=100, marker='o', zorder=10)
        
        # Redraw right plot
        self._draw_robot_view(self.axes[1])
        
        if self.exploration_complete:
            self.fig.suptitle('Frontier Exploration - Complete!', 
                            fontsize=16, fontweight='bold', color='green')

# Run demonstration
def run_exploration_demo():
    """Run frontier exploration demonstration"""
    print("=" * 60)
    print("Frontier Exploration Algorithm Demonstration")
    print("=" * 60)
    print("\nAlgorithm Process:")
    print("1. Environment Sensing & Mapping")
    print("2. Frontier Detection")
    print("3. Target Selection & Path Planning")
    print("\nLegend:")
    print("- Green circle: Robot position")
    print("- Cyan dashed circle: Sensor range")
    print("- Yellow star: Frontier candidate points")
    print("- Red star: Current target frontier")
    print("- Blue dashed line: Planned path")
    print("- Green solid line: History path")
    print("=" * 60)
    
    # Create explorer and visualizer
    explorer = FrontierExplorer(map_size=50, sensor_range=5)
    visualizer = ExplorationVisualizer(explorer)
    
    # Initial sensing
    explorer.sense_environment()
    
    # Create animation
    visualizer.init_plot()
    anim = FuncAnimation(visualizer.fig, visualizer.update, 
                        frames=500, interval=100, repeat=False)
    
    # 保存视频
    print("正在保存视频...")
    anim.save('frontier_exploration.mp4', 
                writer='ffmpeg', 
                fps=10,  # 帧率
                dpi=100)
    print("视频已保存为 'frontier_exploration.mp4'")
    
    # plt.show()

# Run demonstration
if __name__ == "__main__":
    run_exploration_demo()
