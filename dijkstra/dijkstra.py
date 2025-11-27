import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import heapq
from typing import List, Tuple, Set, Dict
import matplotlib.lines as mlines

class DijkstraVisualizer:
    """Dijkstra Algorithm Visualizer (Designed for Teaching Materials)"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.start = None
        self.goal = None
        
    def add_obstacle(self, x: int, y: int):
        """Add obstacle"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1
    
    def add_obstacles_rect(self, x: int, y: int, w: int, h: int):
        """Add rectangular obstacles"""
        for i in range(y, min(y + h, self.height)):
            for j in range(x, min(x + w, self.width)):
                self.grid[i, j] = 1
    
    def set_start(self, x: int, y: int):
        """Set start point"""
        self.start = (x, y)
    
    def set_goal(self, x: int, y: int):
        """Set goal point"""
        self.goal = (x, y)
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int, float]]:
        """Get neighbor nodes with edge costs (4-connected, uniform cost=1)"""
        neighbors = []
        # Order: up, right, down, left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 
                0 <= ny < self.height and 
                self.grid[ny, nx] == 0):
                neighbors.append((nx, ny, 1.0))  # cost = 1
        
        return neighbors
    
    def dijkstra_step_by_step(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Dijkstra step-by-step search, returns state of each step
        
        Returns:
            List of step information dictionaries
        """
        # Priority queue: (distance, node)
        pq = [(0, start)]
        distances = {start: 0}
        visited = set()
        parent = {start: None}
        steps = []
        
        # Record initial state
        steps.append({
            'current': start,
            'pq': list(pq),
            'distances': distances.copy(),
            'visited': visited.copy(),
            'parent': parent.copy(),
            'path': None,
            'found': False,
            'exploring': [],
            'step_type': 'init'
        })
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            # Skip if already visited
            if current in visited:
                continue
            
            visited.add(current)
            
            # Record currently exploring node
            steps.append({
                'current': current,
                'pq': list(pq),
                'distances': distances.copy(),
                'visited': visited.copy(),
                'parent': parent.copy(),
                'path': None,
                'found': False,
                'exploring': [],
                'step_type': 'dequeue',
                'current_dist': current_dist
            })
            
            # Found goal
            if current == goal:
                # Reconstruct path
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                
                steps.append({
                    'current': current,
                    'pq': list(pq),
                    'distances': distances.copy(),
                    'visited': visited.copy(),
                    'parent': parent.copy(),
                    'path': path,
                    'found': True,
                    'exploring': [],
                    'step_type': 'found',
                    'current_dist': current_dist
                })
                return steps
            
            # Explore neighbors
            neighbors = self.get_neighbors(*current)
            new_neighbors = []
            updated_neighbors = []
            
            for nx, ny, cost in neighbors:
                neighbor = (nx, ny)
                new_distance = current_dist + cost
                
                # If not visited and (not in distances or found shorter path)
                if neighbor not in visited:
                    if neighbor not in distances or new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        parent[neighbor] = current
                        heapq.heappush(pq, (new_distance, neighbor))
                        
                        if neighbor not in distances or new_distance < distances[neighbor]:
                            new_neighbors.append(neighbor)
                        else:
                            updated_neighbors.append(neighbor)
            
            # Record state after exploring neighbors
            if new_neighbors or updated_neighbors:
                steps.append({
                    'current': current,
                    'pq': list(pq),
                    'distances': distances.copy(),
                    'visited': visited.copy(),
                    'parent': parent.copy(),
                    'path': None,
                    'found': False,
                    'exploring': new_neighbors,
                    'step_type': 'explore',
                    'current_dist': current_dist
                })
        
        # No path found
        steps.append({
            'current': None,
            'pq': [],
            'distances': distances.copy(),
            'visited': visited.copy(),
            'parent': parent.copy(),
            'path': None,
            'found': False,
            'exploring': [],
            'step_type': 'no_path'
        })
        
        return steps
    
    def visualize_static_explanation(self):
        """Create static Dijkstra algorithm explanation diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        # Run Dijkstra to get all steps
        steps = self.dijkstra_step_by_step(self.start, self.goal)
        
        # Find key frames
        init_step = steps[0]
        mid_step = steps[len(steps) // 2]
        near_end_step = steps[-3] if len(steps) > 3 else steps[-1]
        final_step = steps[-1]
        
        # Draw four key stages
        self._draw_dijkstra_state(axes[0, 0], init_step, "Step 1: Initialization")
        self._draw_dijkstra_state(axes[0, 1], mid_step, f"Step {len(steps)//2}: Searching")
        self._draw_dijkstra_state(axes[1, 0], near_end_step, f"Step {len(steps)-2}: Near Goal")
        self._draw_dijkstra_state(axes[1, 1], final_step, "Final: Shortest Path Found")
        
        plt.tight_layout()
        return fig
    
    def _draw_dijkstra_state(self, ax, step_info, title):
        """Draw Dijkstra state at a given step"""
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        current = step_info['current']
        visited = step_info['visited']
        exploring = step_info['exploring']
        path = step_info['path']
        distances = step_info['distances']
        pq_nodes = [node for dist, node in step_info['pq']]
        
        # Draw all edges (graph structure)
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 0:
                    neighbors = self.get_neighbors(j, i)
                    for nx, ny, cost in neighbors:
                        ax.plot([j + 0.5, nx + 0.5], [i + 0.5, ny + 0.5],
                               'lightgray', linewidth=1, alpha=0.3, zorder=1)
        
        # Draw grid and obstacles
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 1:
                    rect = patches.Rectangle((j, i), 1, 1,
                                            linewidth=1, edgecolor='black',
                                            facecolor='#2C3E50')
                    ax.add_patch(rect)
        
        # Draw visited nodes with distance labels
        for (x, y) in visited:
            if (x, y) != self.start and (x, y) != self.goal:
                ax.plot(x + 0.5, y + 0.5, 'o', color='#AED6F1', 
                       markersize=15, zorder=5, alpha=0.7)
                # Show distance
                if (x, y) in distances:
                    ax.text(x + 0.5, y + 0.8, f'{distances[(x, y)]:.0f}',
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', alpha=0.8))
        
        # Draw nodes in priority queue
        for (x, y) in pq_nodes:
            if (x, y) != self.start and (x, y) != self.goal:
                ax.plot(x + 0.5, y + 0.5, 's', color='#F9E79F', 
                       markersize=14, zorder=6, alpha=0.8)
                # Show tentative distance
                if (x, y) in distances:
                    ax.text(x + 0.5, y + 0.8, f'{distances[(x, y)]:.0f}',
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='yellow', alpha=0.8))
        
        # Draw neighbors being explored
        for (x, y) in exploring:
            ax.plot(x + 0.5, y + 0.5, 'D', color='#82E0AA', 
                   markersize=12, zorder=7, alpha=0.9)
        
        # Draw current node
        if current:
            cx, cy = current
            ax.plot(cx + 0.5, cy + 0.5, 'o', color='#FF6B6B', 
                   markersize=20, zorder=8,
                   markeredgecolor='darkred', markeredgewidth=2)
            if 'current_dist' in step_info:
                ax.text(cx + 0.5, cy + 0.8, f'{step_info["current_dist"]:.0f}',
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='red', alpha=0.7, edgecolor='darkred'))
        
        # Draw start point
        sx, sy = self.start
        ax.plot(sx + 0.5, sy + 0.5, 'o', color='#2ECC71', 
               markersize=22, zorder=10,
               markeredgecolor='darkgreen', markeredgewidth=3)
        ax.text(sx + 0.5, sy + 0.5, 'S', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')
        
        # Draw goal point
        gx, gy = self.goal
        ax.plot(gx + 0.5, gy + 0.5, 's', color='#E74C3C', 
               markersize=22, zorder=10,
               markeredgecolor='darkred', markeredgewidth=3)
        ax.text(gx + 0.5, gy + 0.5, 'G', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')
        
        # If path found, draw path
        if path:
            path_x = [x + 0.5 for x, y in path]
            path_y = [y + 0.5 for x, y in path]
            ax.plot(path_x, path_y, 'r-', linewidth=4, alpha=0.7, zorder=9,
                   label=f'Shortest Path (Cost: {distances[self.goal]:.0f})')
        
        ax.set_xlim(-0.5, self.width + 0.5)
        ax.set_ylim(-0.5, self.height + 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)
        
        # Add legend
        legend_elements = [
            mlines.Line2D([], [], color='#2ECC71', marker='o', linestyle='None',
                         markersize=12, label='Start', 
                         markeredgecolor='darkgreen', markeredgewidth=2),
            mlines.Line2D([], [], color='#E74C3C', marker='s', linestyle='None',
                         markersize=12, label='Goal',
                         markeredgecolor='darkred', markeredgewidth=2),
            mlines.Line2D([], [], color='#FF6B6B', marker='o', linestyle='None',
                         markersize=12, label='Current Node',
                         markeredgecolor='darkred', markeredgewidth=2),
            mlines.Line2D([], [], color='#F9E79F', marker='s', linestyle='None',
                         markersize=10, label='In Priority Queue'),
            mlines.Line2D([], [], color='#82E0AA', marker='D', linestyle='None',
                         markersize=8, label='New Neighbors'),
            mlines.Line2D([], [], color='#AED6F1', marker='o', linestyle='None',
                         markersize=10, label='Visited'),
        ]
        
        if path:
            legend_elements.append(
                mlines.Line2D([], [], color='red', linewidth=3, 
                             label=f'Shortest Path (Cost: {distances[self.goal]:.0f})')
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Add statistics
        info_text = f"Visited: {len(visited)} nodes\n"
        info_text += f"PQ Size: {len(pq_nodes)}\n"
        if exploring:
            info_text += f"New Discovered: {len(exploring)} neighbors"
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def create_dijkstra_animation(self, filename='dijkstra_animation.gif', fps=2):
        """Create animation of Dijkstra search process"""
        steps = self.dijkstra_step_by_step(self.start, self.goal)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def animate(frame_num):
            ax.clear()
            
            step_info = steps[frame_num]
            
            # Set title to show current step
            step_type = step_info['step_type']
            title_dict = {
                'init': f'Step {frame_num + 1}/{len(steps)}: Initialize - Start from Begin',
                'dequeue': f'Step {frame_num + 1}/{len(steps)}: Extract Min Distance Node',
                'explore': f'Step {frame_num + 1}/{len(steps)}: Update Neighbor Distances',
                'found': f'Step {frame_num + 1}/{len(steps)}: ✓ Shortest Path Found!',
                'no_path': f'Step {frame_num + 1}/{len(steps)}: ✗ No Path to Goal'
            }
            
            ax.set_title(title_dict.get(step_type, f'Step {frame_num + 1}/{len(steps)}'),
                        fontsize=16, fontweight='bold', pad=20)
            
            current = step_info['current']
            visited = step_info['visited']
            exploring = step_info['exploring']
            path = step_info['path']
            distances = step_info['distances']
            pq_nodes = [node for dist, node in step_info['pq']]
            
            # Draw graph structure edges
            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i, j] == 0:
                        neighbors = self.get_neighbors(j, i)
                        for nx, ny, cost in neighbors:
                            ax.plot([j + 0.5, nx + 0.5], [i + 0.5, ny + 0.5],
                                   'lightgray', linewidth=1.5, alpha=0.3, zorder=1)
            
            # Draw obstacles
            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i, j] == 1:
                        rect = patches.Rectangle((j, i), 1, 1,
                                                linewidth=1.5, edgecolor='black',
                                                facecolor='#2C3E50')
                        ax.add_patch(rect)
            
            # Draw visited nodes with distances
            for (x, y) in visited:
                if (x, y) != self.start and (x, y) != self.goal:
                    ax.plot(x + 0.5, y + 0.5, 'o', color='#AED6F1', 
                           markersize=18, zorder=5, alpha=0.7,
                           markeredgecolor='#5DADE2', markeredgewidth=1)
                    if (x, y) in distances:
                        ax.text(x + 0.5, y + 0.85, f'{distances[(x, y)]:.0f}',
                               ha='center', va='center', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', alpha=0.9))
            
            # Draw nodes in priority queue
            for (x, y) in pq_nodes:
                if (x, y) != self.start and (x, y) != self.goal:
                    ax.plot(x + 0.5, y + 0.5, 's', color='#F9E79F', 
                           markersize=16, zorder=6, alpha=0.9,
                           markeredgecolor='#F4D03F', markeredgewidth=2)
                    if (x, y) in distances:
                        ax.text(x + 0.5, y + 0.85, f'{distances[(x, y)]:.0f}',
                               ha='center', va='center', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='yellow', alpha=0.9))
            
            # Draw neighbors being explored (with animation effect)
            for (x, y) in exploring:
                ax.plot(x + 0.5, y + 0.5, 'D', color='#82E0AA', 
                       markersize=14, zorder=7, alpha=0.95,
                       markeredgecolor='#27AE60', markeredgewidth=2)
                # Add arrow pointing to new neighbor
                if current:
                    cx, cy = current
                    ax.annotate('', xy=(x + 0.5, y + 0.5), 
                               xytext=(cx + 0.5, cy + 0.5),
                               arrowprops=dict(arrowstyle='->', 
                                             color='green', 
                                             lw=2, 
                                             alpha=0.6))
            
            # Draw current node
            if current:
                cx, cy = current
                # Add pulse effect
                ax.plot(cx + 0.5, cy + 0.5, 'o', color='#FF6B6B', 
                       markersize=24, zorder=8, alpha=0.5)
                ax.plot(cx + 0.5, cy + 0.5, 'o', color='#FF6B6B', 
                       markersize=20, zorder=8,
                       markeredgecolor='darkred', markeredgewidth=3)
                if 'current_dist' in step_info:
                    ax.text(cx + 0.5, cy + 0.85, f'{step_info["current_dist"]:.0f}',
                           ha='center', va='center', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='red', alpha=0.8, edgecolor='darkred'))
            
            # Draw start point
            sx, sy = self.start
            ax.plot(sx + 0.5, sy + 0.5, 'o', color='#2ECC71', 
                   markersize=24, zorder=10,
                   markeredgecolor='darkgreen', markeredgewidth=3)
            ax.text(sx + 0.5, sy + 0.5, 'S', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white', zorder=11)
            
            # Draw goal point
            gx, gy = self.goal
            ax.plot(gx + 0.5, gy + 0.5, 's', color='#E74C3C', 
                   markersize=24, zorder=10,
                   markeredgecolor='darkred', markeredgewidth=3)
            ax.text(gx + 0.5, gy + 0.5, 'G', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white', zorder=11)
            
            # If path found, draw path
            if path:
                path_x = [x + 0.5 for x, y in path]
                path_y = [y + 0.5 for x, y in path]
                ax.plot(path_x, path_y, 'r-', linewidth=5, alpha=0.8, zorder=9)
                
                # Draw nodes on path
                for x, y in path[1:-1]:
                    ax.plot(x + 0.5, y + 0.5, 'o', color='#FF6B6B',
                           markersize=16, zorder=9, alpha=0.9,
                           markeredgecolor='darkred', markeredgewidth=2)
            
            ax.set_xlim(-0.5, self.width + 0.5)
            ax.set_ylim(-0.5, self.height + 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            
            # Add legend
            legend_elements = [
                mlines.Line2D([], [], color='#2ECC71', marker='o', linestyle='None',
                             markersize=12, label='Start', 
                             markeredgecolor='darkgreen', markeredgewidth=2),
                mlines.Line2D([], [], color='#E74C3C', marker='s', linestyle='None',
                             markersize=12, label='Goal',
                             markeredgecolor='darkred', markeredgewidth=2),
                mlines.Line2D([], [], color='#FF6B6B', marker='o', linestyle='None',
                             markersize=12, label='Current',
                             markeredgecolor='darkred', markeredgewidth=2),
                mlines.Line2D([], [], color='#F9E79F', marker='s', linestyle='None',
                             markersize=10, label='In PQ'),
                mlines.Line2D([], [], color='#82E0AA', marker='D', linestyle='None',
                             markersize=8, label='New Found'),
                mlines.Line2D([], [], color='#AED6F1', marker='o', linestyle='None',
                             markersize=10, label='Visited'),
            ]
            
            if path and self.goal in distances:
                legend_elements.append(
                    mlines.Line2D([], [], color='red', linewidth=3, 
                                 label=f'Shortest Path (Cost: {distances[self.goal]:.0f})')
                )
            
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                     framealpha=0.95)
            
            # Add detailed Dijkstra information
            info_text = f"Dijkstra Algorithm Status:\n"
            info_text += f"━━━━━━━━━━━━━━━━━━━━━━\n"
            info_text += f"✓ Visited: {len(visited)}\n"
            info_text += f"⟳ PQ Size: {len(pq_nodes)}\n"
            
            if exploring:
                info_text += f"⊕ New Found: {len(exploring)}\n"
            
            if current and 'current_dist' in step_info:
                info_text += f"⊙ Current: {current}\n"
                info_text += f"⊙ Distance: {step_info['current_dist']:.0f}\n"
            
            if path and self.goal in distances:
                info_text += f"━━━━━━━━━━━━━━━━━━━━━━\n"
                info_text += f"✓ Path Cost: {distances[self.goal]:.0f}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', 
                            alpha=0.95, edgecolor='orange', linewidth=2))
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(steps),
                           interval=1000/fps, repeat=True)
        
        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=100)
        plt.close()
        
        print(f"✓ Animation saved as: {filename}")
        print(f"  Total steps: {len(steps)}")
        print(f"  Frame rate: {fps} FPS")
        
        return anim


# ==================== Usage Examples ====================

def demo_simple_dijkstra():
    """Simple scenario Dijkstra demonstration"""
    print("=" * 60)
    print("Example 1: Simple Scenario Dijkstra Visualization")
    print("=" * 60)
    
    viz = DijkstraVisualizer(width=10, height=8)
    
    # Add obstacles
    viz.add_obstacles_rect(3, 2, 2, 4)
    viz.add_obstacles_rect(6, 1, 1, 3)
    viz.add_obstacle(7, 5)
    viz.add_obstacle(8, 5)
    
    # Set start and goal
    viz.set_start(1, 3)
    viz.set_goal(8, 3)
    
    # Create static explanation diagram
    print("Generating static step diagram...")
    fig = viz.visualize_static_explanation()
    plt.savefig('dijkstra_steps_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create animation
    print("\nGenerating GIF animation...")
    viz.create_dijkstra_animation('dijkstra_simple.gif', fps=2)
    print("✓ Done!")


def demo_maze_dijkstra():
    """Maze scenario Dijkstra demonstration"""
    print("\n" + "=" * 60)
    print("Example 2: Maze Scenario Dijkstra Visualization")
    print("=" * 60)
    
    viz = DijkstraVisualizer(width=12, height=10)
    
    # Create maze-like obstacles
    viz.add_obstacles_rect(2, 1, 1, 6)
    viz.add_obstacles_rect(4, 3, 1, 6)
    viz.add_obstacles_rect(6, 1, 1, 5)
    viz.add_obstacles_rect(8, 4, 1, 5)
    viz.add_obstacles_rect(10, 2, 1, 4)
    
    # Set start and goal
    viz.set_start(0, 0)
    viz.set_goal(11, 9)
    
    # Create animation (faster frame rate)
    print("Generating GIF animation...")
    viz.create_dijkstra_animation('dijkstra_maze.gif', fps=3)
    print("✓ Done!")


def demo_complex_dijkstra():
    """Complex scenario Dijkstra demonstration"""
    print("\n" + "=" * 60)
    print("Example 3: Complex Scenario Dijkstra Visualization")
    print("=" * 60)
    
    viz = DijkstraVisualizer(width=15, height=12)
    
    # Create complex obstacle layout
    viz.add_obstacles_rect(3, 2, 3, 2)
    viz.add_obstacles_rect(3, 6, 3, 2)
    viz.add_obstacles_rect(8, 3, 2, 5)
    viz.add_obstacles_rect(11, 1, 2, 4)
    viz.add_obstacles_rect(11, 7, 2, 4)
    viz.add_obstacle(6, 4)
    viz.add_obstacle(6, 5)
    
    # Set start and goal
    viz.set_start(1, 1)
    viz.set_goal(13, 10)
    
    # Create static explanation diagram
    print("Generating static step diagram...")
    fig = viz.visualize_static_explanation()
    plt.savefig('dijkstra_complex_steps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create animation
    print("\nGenerating GIF animation...")
    viz.create_dijkstra_animation('dijkstra_complex.gif', fps=2.5)
    print("✓ Done!")


def demo_no_path_dijkstra():
    """No solution scenario Dijkstra demonstration"""
    print("\n" + "=" * 60)
    print("Example 4: No Solution Scenario Dijkstra Visualization")
    print("=" * 60)
    
    viz = DijkstraVisualizer(width=10,    height=8)
    
    # Create an enclosed area
    viz.add_obstacles_rect(4, 2, 1, 4)
    viz.add_obstacles_rect(5, 2, 3, 1)
    viz.add_obstacles_rect(5, 5, 3, 1)
    viz.add_obstacles_rect(7, 3, 1, 2)
    
    # Start outside the enclosed area, goal inside
    viz.set_start(1, 3)
    viz.set_goal(6, 4)
    
    # Create animation
    print("Generating GIF animation (showing how Dijkstra handles no solution)...")
    viz.create_dijkstra_animation('dijkstra_no_path.gif', fps=2)
    print("✓ Done!")


# ==================== Main Function ====================

if __name__ == "__main__":
    print("🎓 Dijkstra Path Planning Algorithm Visualization")
    print("=" * 60)
    print()
    
    # Example 1: Simple scenario
    demo_simple_dijkstra()
    
    # Example 2: Maze scenario
    demo_maze_dijkstra()
    
    # Example 3: Complex scenario
    demo_complex_dijkstra()
    
    # Example 4: No solution scenario
    demo_no_path_dijkstra()
    
    print("\n" + "=" * 60)
    print("✓ All demonstrations completed!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  📊 dijkstra_steps_explanation.png - Static step explanation diagram")
    print("  📊 dijkstra_complex_steps.png - Complex scenario step diagram")
    print("  🎬 dijkstra_simple.gif - Simple scenario animation")
    print("  🎬 dijkstra_maze.gif - Maze scenario animation")
    print("  🎬 dijkstra_complex.gif - Complex scenario animation")
    print("  🎬 dijkstra_no_path.gif - No solution scenario animation")
    print("\n" + "=" * 60)
    print("📚 Key Differences from BFS:")
    print("=" * 60)
    print("  ✓ Uses Priority Queue (not FIFO queue)")
    print("  ✓ Always extracts node with minimum distance")
    print("  ✓ Tracks cumulative distance to each node")
    print("  ✓ Guarantees optimal path in weighted graphs")
    print("  ✓ Shows distance labels on nodes")
    print("=" * 60)
