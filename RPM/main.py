import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy.spatial import KDTree
import matplotlib.animation as animation

class PRMVisualizer:
    def __init__(self, width=100, height=100, n_samples=200, k_neighbors=5):
        """
        Initialize PRM Visualizer
        
        Parameters:
        - width, height: Space dimensions
        - n_samples: Number of sample points
        - k_neighbors: Number of neighbors to connect per point
        """
        self.width = width
        self.height = height
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
        
        # Define obstacles (rectangles and circles)
        self.obstacles = [
            {'type': 'rect', 'x': 20, 'y': 20, 'width': 15, 'height': 30},
            {'type': 'rect', 'x': 60, 'y': 50, 'width': 20, 'height': 25},
            {'type': 'circle', 'x': 40, 'y': 70, 'radius': 10},
            {'type': 'circle', 'x': 75, 'y': 20, 'radius': 8},
        ]
        
        self.start = np.array([5, 5])
        self.goal = np.array([95, 95])
        
        self.samples = []
        self.edges = []
        
    def is_collision_free(self, point):
        """Check if point collides with obstacles"""
        x, y = point
        
        for obs in self.obstacles:
            if obs['type'] == 'rect':
                if (obs['x'] <= x <= obs['x'] + obs['width'] and 
                    obs['y'] <= y <= obs['y'] + obs['height']):
                    return False
            elif obs['type'] == 'circle':
                dist = np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
                if dist <= obs['radius']:
                    return False
        return True
    
    def is_path_collision_free(self, p1, p2, n_checks=20):
        """Check if path between two points is collision-free"""
        for i in range(n_checks + 1):
            t = i / n_checks
            point = p1 + t * (p2 - p1)
            if not self.is_collision_free(point):
                return False
        return True
    
    def sample_free_space(self):
        """Randomly sample points in free space"""
        samples = [self.start, self.goal]
        
        while len(samples) < self.n_samples + 2:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            point = np.array([x, y])
            
            if self.is_collision_free(point):
                samples.append(point)
        
        self.samples = np.array(samples)
        return self.samples
    
    def build_roadmap(self):
        """Build roadmap"""
        # Use KD-tree to find nearest neighbors
        tree = KDTree(self.samples)
        
        for i, sample in enumerate(self.samples):
            # Find k nearest neighbors
            distances, indices = tree.query(sample, k=self.k_neighbors + 1)
            
            for j, idx in enumerate(indices[1:]):  # Skip itself
                neighbor = self.samples[idx]
                
                # Check if path is collision-free
                if self.is_path_collision_free(sample, neighbor):
                    self.edges.append((i, idx))
    
    def visualize_step_by_step(self):
        """Visualize PRM construction process step by step"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        fig.suptitle('PRM Algorithm: Randomly Scatter Points to Build Road Network', 
                     fontsize=16, fontweight='bold')
        
        # Step 1: Show environment and obstacles
        ax1 = axes[0, 0]
        self.draw_environment(ax1)
        ax1.set_title('Step 1: Environment & Obstacles', fontsize=12, fontweight='bold')
        
        # Step 2: Random sampling
        ax2 = axes[0, 1]
        self.draw_environment(ax2)
        self.sample_free_space()
        ax2.scatter(self.samples[2:, 0], self.samples[2:, 1], 
                   c='lightblue', s=30, alpha=0.6, label='Sample Points')
        ax2.scatter(*self.start, c='green', s=200, marker='*', 
                   label='Start', zorder=5)
        ax2.scatter(*self.goal, c='red', s=200, marker='*', 
                   label='Goal', zorder=5)
        ax2.legend()
        ax2.set_title(f'Step 2: Random Sampling {self.n_samples} Points', 
                     fontsize=12, fontweight='bold')
        
        # Step 3: Build connections
        ax3 = axes[1, 0]
        self.draw_environment(ax3)
        self.build_roadmap()
        
        # Draw edges
        for i, j in self.edges:
            ax3.plot([self.samples[i][0], self.samples[j][0]], 
                    [self.samples[i][1], self.samples[j][1]], 
                    'gray', linewidth=0.5, alpha=0.3)
        
        ax3.scatter(self.samples[2:, 0], self.samples[2:, 1], 
                   c='lightblue', s=30, alpha=0.6)
        ax3.scatter(*self.start, c='green', s=200, marker='*', zorder=5)
        ax3.scatter(*self.goal, c='red', s=200, marker='*', zorder=5)
        ax3.set_title(f'Step 3: Connect Neighbors ({self.k_neighbors} neighbors per point)', 
                     fontsize=12, fontweight='bold')
        
        # Step 4: Complete roadmap
        ax4 = axes[1, 1]
        self.draw_environment(ax4)
        
        # Draw edges (more visible)
        for i, j in self.edges:
            ax4.plot([self.samples[i][0], self.samples[j][0]], 
                    [self.samples[i][1], self.samples[j][1]], 
                    'blue', linewidth=1, alpha=0.5)
        
        ax4.scatter(self.samples[2:, 0], self.samples[2:, 1], 
                   c='cyan', s=40, alpha=0.8, edgecolors='blue', linewidth=0.5)
        ax4.scatter(*self.start, c='green', s=200, marker='*', 
                   label='Start', zorder=5)
        ax4.scatter(*self.goal, c='red', s=200, marker='*', 
                   label='Goal', zorder=5)
        ax4.legend()
        ax4.set_title(f'Step 4: Complete Road Network ({len(self.edges)} edges)', 
                     fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('prm_roadmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def draw_environment(self, ax):
        """Draw environment and obstacles"""
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Draw obstacles
        for obs in self.obstacles:
            if obs['type'] == 'rect':
                rect = Rectangle((obs['x'], obs['y']), obs['width'], 
                               obs['height'], facecolor='dimgray', 
                               edgecolor='black', linewidth=2)
                ax.add_patch(rect)
            elif obs['type'] == 'circle':
                circle = Circle((obs['x'], obs['y']), obs['radius'], 
                              facecolor='dimgray', edgecolor='black', linewidth=2)
                ax.add_patch(circle)
    
    def animate_construction(self):
        """Animate PRM construction process"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        self.sample_free_space()
        self.build_roadmap()
        
        def update(frame):
            ax.clear()
            self.draw_environment(ax)
            
            if frame < len(self.samples):
                # Progressively show sample points
                ax.scatter(self.samples[:frame, 0], self.samples[:frame, 1], 
                          c='lightblue', s=50, alpha=0.6)
                ax.set_title(f'PRM Construction - Sample Points: {frame}/{len(self.samples)}', 
                           fontsize=14, fontweight='bold')
            else:
                # Show all points
                ax.scatter(self.samples[:, 0], self.samples[:, 1], 
                          c='lightblue', s=50, alpha=0.6)
                
                # Progressively show edges
                edge_frame = frame - len(self.samples)
                if edge_frame < len(self.edges):
                    for i in range(edge_frame):
                        idx1, idx2 = self.edges[i]
                        ax.plot([self.samples[idx1][0], self.samples[idx2][0]], 
                               [self.samples[idx1][1], self.samples[idx2][1]], 
                               'blue', linewidth=1, alpha=0.5)
                    ax.set_title(f'PRM Construction - Connecting Edges: {edge_frame}/{len(self.edges)}', 
                               fontsize=14, fontweight='bold')
                else:
                    for i, j in self.edges:
                        ax.plot([self.samples[i][0], self.samples[j][0]], 
                               [self.samples[i][1], self.samples[j][1]], 
                               'blue', linewidth=1, alpha=0.5)
                    ax.set_title('PRM Construction Complete!', fontsize=14, fontweight='bold')
            
            # Always show start and goal
            ax.scatter(*self.start, c='green', s=200, marker='*', zorder=5)
            ax.scatter(*self.goal, c='red', s=200, marker='*', zorder=5)
        
        anim = animation.FuncAnimation(fig, update, 
                                      frames=len(self.samples) + len(self.edges) + 10,
                                      interval=50, repeat=True)
        plt.show()
        
        return anim


# Run visualization
if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Create PRM visualizer
    prm = PRMVisualizer(width=100, height=100, n_samples=200, k_neighbors=5)
    
    # Show step-by-step construction process
    prm.visualize_step_by_step()
    
    # Uncomment below to see animation
    prm.animate_construction()
