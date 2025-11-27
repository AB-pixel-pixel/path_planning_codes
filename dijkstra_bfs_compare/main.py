import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import deque
import heapq
from typing import List, Tuple, Set, Dict
import matplotlib.lines as mlines

class PathFindingComparison:
    """Compare BFS and Dijkstra on 8-neighborhood grid"""
    
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
    
    def get_neighbors_8(self, x: int, y: int) -> List[Tuple[int, int, float]]:
        """
        Get neighbor nodes in 8-neighborhood with distances
        Returns: List of (nx, ny, cost)
        - Lateral moves (↑↓←→): cost = 1.0
        - Diagonal moves (↖↗↙↘): cost = √2 ≈ 1.414
        """
        neighbors = []
        # 8 directions: N, NE, E, SE, S, SW, W, NW
        directions = [
            (0, -1, 1.0),      # North
            (1, -1, np.sqrt(2)),  # Northeast (diagonal)
            (1, 0, 1.0),       # East
            (1, 1, np.sqrt(2)),   # Southeast (diagonal)
            (0, 1, 1.0),       # South
            (-1, 1, np.sqrt(2)),  # Southwest (diagonal)
            (-1, 0, 1.0),      # West
            (-1, -1, np.sqrt(2))  # Northwest (diagonal)
        ]
        
        for dx, dy, cost in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 
                0 <= ny < self.height and 
                self.grid[ny, nx] == 0):
                neighbors.append((nx, ny, cost))
        
        return neighbors
    
    def bfs_8_neighborhood(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        BFS on 8-neighborhood grid (ignores edge weights)
        Returns: (path, visited_order, total_cost)
        """
        queue = deque([start])
        visited = {start}
        parent = {start: None}
        visited_order = [start]
        
        while queue:
            current = queue.popleft()
            
            if current == goal:
                # Reconstruct path
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                
                # Calculate actual cost of BFS path
                total_cost = 0.0
                for i in range(len(path) - 1):
                    x1, y1 = path[i]
                    x2, y2 = path[i + 1]
                    dx, dy = abs(x2 - x1), abs(y2 - y1)
                    if dx == 1 and dy == 1:
                        total_cost += np.sqrt(2)
                    else:
                        total_cost += 1.0
                
                return path, visited_order, total_cost
            
            neighbors = self.get_neighbors_8(*current)
            for nx, ny, _ in neighbors:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = current
                    queue.append((nx, ny))
                    visited_order.append((nx, ny))
        
        return None, visited_order, float('inf')
    
    def dijkstra_8_neighborhood(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Dijkstra's algorithm on 8-neighborhood grid (considers edge weights)
        Returns: (path, visited_order, total_cost)
        """
        # Priority queue: (cost, node)
        pq = [(0, start)]
        visited = set()
        distances = {start: 0}
        parent = {start: None}
        visited_order = []
        
        while pq:
            current_cost, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            visited_order.append(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path, visited_order, distances[goal]
            
            neighbors = self.get_neighbors_8(*current)
            for nx, ny, edge_cost in neighbors:
                if (nx, ny) not in visited:
                    new_cost = current_cost + edge_cost
                    if (nx, ny) not in distances or new_cost < distances[(nx, ny)]:
                        distances[(nx, ny)] = new_cost
                        parent[(nx, ny)] = current
                        heapq.heappush(pq, (new_cost, (nx, ny)))
        
        return None, visited_order, float('inf')
    
    def visualize_comparison(self, save_file='path_comparison.png'):
        """Create side-by-side comparison of BFS vs Dijkstra"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        
        # Run both algorithms
        bfs_path, bfs_visited, bfs_cost = self.bfs_8_neighborhood(self.start, self.goal)
        dijkstra_path, dijkstra_visited, dijkstra_cost = self.dijkstra_8_neighborhood(self.start, self.goal)
        
        # Draw BFS result
        self._draw_path_result(axes[0], bfs_path, bfs_visited, bfs_cost, 
                               "BFS (8-neighborhood, ignores edge weights)",
                               color='#3498DB')
        
        # Draw Dijkstra result
        self._draw_path_result(axes[1], dijkstra_path, dijkstra_visited, dijkstra_cost,
                               "Dijkstra (8-neighborhood, considers edge weights)",
                               color='#E74C3C')
        
        # Add overall comparison text
        fig.suptitle('BFS vs Dijkstra on 8-Neighborhood Grid\n' + 
                     f'BFS Cost: {bfs_cost:.3f} | Dijkstra Cost: {dijkstra_cost:.3f} | ' +
                     f'Difference: {bfs_cost - dijkstra_cost:.3f} ({((bfs_cost/dijkstra_cost - 1)*100):.2f}% longer)',
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison saved as: {save_file}")
        plt.show()
        
        return bfs_path, bfs_cost, dijkstra_path, dijkstra_cost
    
    def _draw_path_result(self, ax, path, visited, cost, title, color):
        """Draw single algorithm result"""
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        
        # Draw grid edges for 8-neighborhood
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 0:
                    neighbors = self.get_neighbors_8(j, i)
                    for nx, ny, edge_cost in neighbors:
                        # Different line styles for lateral vs diagonal
                        if edge_cost == 1.0:
                            ax.plot([j + 0.5, nx + 0.5], [i + 0.5, ny + 0.5],
                                   'lightgray', linewidth=1, alpha=0.3, zorder=1)
                        else:
                            ax.plot([j + 0.5, nx + 0.5], [i + 0.5, ny + 0.5],
                                   'lightgray', linewidth=1, alpha=0.2, 
                                   linestyle='--', zorder=1)
        
        # Draw obstacles
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 1:
                    rect = patches.Rectangle((j, i), 1, 1,
                                            linewidth=1, edgecolor='black',
                                            facecolor='#2C3E50')
                    ax.add_patch(rect)
        
        # Draw visited nodes
        for (x, y) in visited:
            if (x, y) != self.start and (x, y) != self.goal:
                ax.plot(x + 0.5, y + 0.5, 'o', color='#E8F4F8', 
                       markersize=12, zorder=5, alpha=0.6,
                       markeredgecolor='#AED6F1', markeredgewidth=1)
        
        # Draw path
        if path:
            path_x = [x + 0.5 for x, y in path]
            path_y = [y + 0.5 for x, y in path]
            ax.plot(path_x, path_y, '-', color=color, linewidth=5, 
                   alpha=0.8, zorder=9, label=f'Path (Cost: {cost:.3f})')
            
            # Highlight path nodes
            for x, y in path[1:-1]:
                ax.plot(x + 0.5, y + 0.5, 'o', color=color,
                       markersize=14, zorder=10, alpha=0.9,
                       markeredgecolor='darkred', markeredgewidth=2)
            
            # Draw edge costs on path
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                edge_cost = np.sqrt(2) if (dx == 1 and dy == 1) else 1.0
                
                mid_x = (x1 + x2) / 2 + 0.5
                mid_y = (y1 + y2) / 2 + 0.5
                
                ax.text(mid_x, mid_y, f'{edge_cost:.2f}',
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', alpha=0.7),
                       zorder=11)
        
        # Draw start point
        sx, sy = self.start
        ax.plot(sx + 0.5, sy + 0.5, 'o', color='#2ECC71', 
               markersize=22, zorder=12,
               markeredgecolor='darkgreen', markeredgewidth=3)
        ax.text(sx + 0.5, sy + 0.5, 'S', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white', zorder=13)
        
        # Draw goal point
        gx, gy = self.goal
        ax.plot(gx + 0.5, gy + 0.5, 's', color='#E74C3C', 
               markersize=22, zorder=12,
               markeredgecolor='darkred', markeredgewidth=3)
        ax.text(gx + 0.5, gy + 0.5, 'G', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white', zorder=13)
        
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
            mlines.Line2D([], [], color='lightgray', linewidth=2,
                         label='Lateral edge (cost=1.0)'),
            mlines.Line2D([], [], color='lightgray', linewidth=2, linestyle='--',
                         label='Diagonal edge (cost=√2≈1.414)'),
        ]
        
        if path:
            legend_elements.append(
                mlines.Line2D([], [], color=color, linewidth=3, 
                             label=f'Found Path (cost={cost:.3f})')
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Add statistics
        info_text = f"Algorithm Statistics:\n"
        info_text += f"━━━━━━━━━━━━━━━━━━\n"
        info_text += f"Visited: {len(visited)} nodes\n"
        if path:
            info_text += f"Path length: {len(path)-1} steps\n"
            info_text += f"Total cost: {cost:.4f}\n"
            
            # Count lateral vs diagonal moves
            lateral = sum(1 for i in range(len(path)-1) 
                         if abs(path[i][0]-path[i+1][0]) + abs(path[i][1]-path[i+1][1]) == 1)
            diagonal = len(path) - 1 - lateral
            info_text += f"Lateral moves: {lateral}\n"
            info_text += f"Diagonal moves: {diagonal}"
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', 
                        alpha=0.9, edgecolor='orange', linewidth=2))


# ==================== Demonstration Examples ====================

def demo_simple_comparison():
    """Simple scenario showing BFS vs Dijkstra difference"""
    print("=" * 70)
    print("Demo 1: Simple Diagonal Path - BFS vs Dijkstra")
    print("=" * 70)
    print("\nScenario: Direct diagonal path available")
    print("Expected: BFS prefers mixed moves, Dijkstra prefers pure diagonal\n")
    
    viz = PathFindingComparison(width=10, height=10)
    
    # Simple open space - diagonal path is optimal
    viz.set_start(1, 1)
    viz.set_goal(8, 8)
    
    bfs_path, bfs_cost, dij_path, dij_cost = viz.visualize_comparison('comparison_simple.png')
    
    print(f"\nResults:")
    print(f"  BFS path cost: {bfs_cost:.4f}")
    print(f"  Dijkstra path cost: {dij_cost:.4f}")
    print(f"  Difference: {bfs_cost - dij_cost:.4f} ({((bfs_cost/dij_cost-1)*100):.2f}% longer)")
    print(f"  BFS is suboptimal: {bfs_cost > dij_cost}")


def demo_obstacle_comparison():
    """Scenario with obstacles showing clearer difference"""
    print("\n" + "=" * 70)
    print("Demo 2: Path with Obstacles - BFS vs Dijkstra")
    print("=" * 70)
    print("\nScenario: Obstacles force different path choices")
    print("Expected: BFS ignores edge weights, Dijkstra optimizes\n")
    
    viz = PathFindingComparison(width=12, height=10)
    
    # Add obstacles to create interesting paths
    viz.add_obstacles_rect(4, 2, 1, 5)
    viz.add_obstacles_rect(7, 3, 1, 4)
    
    viz.set_start(1, 4)
    viz.set_goal(10, 5)
    
    bfs_path, bfs_cost, dij_path, dij_cost = viz.visualize_comparison('comparison_obstacles.png')
    
    print(f"\nResults:")
    print(f"  BFS path cost: {bfs_cost:.4f}")
    print(f"  Dijkstra path cost: {dij_cost:.4f}")
    print(f"  Difference: {bfs_cost - dij_cost:.4f} ({((bfs_cost/dij_cost-1)*100):.2f}% longer)")
    print(f"  BFS is suboptimal: {bfs_cost > dij_cost}")


def demo_extreme_comparison():
    """Scenario maximizing the difference between BFS and Dijkstra"""
    print("\n" + "=" * 70)
    print("Demo 3: Extreme Case - Maximum BFS Suboptimality")
    print("=" * 70)
    print("\nScenario: Layout designed to expose BFS weakness")
    print("Expected: BFS finds much longer path than Dijkstra\n")
    
    viz = PathFindingComparison(width=15, height=12)
    
    # Create a scenario where BFS explores in a bad order
    # Obstacles force a choice between many short moves vs few diagonal moves
    viz.add_obstacles_rect(5, 3, 1, 6)
    viz.add_obstacles_rect(9, 3, 1, 6)
    
    viz.set_start(1, 5)
    viz.set_goal(13, 5)
    
    bfs_path, bfs_cost, dij_path, dij_cost = viz.visualize_comparison('comparison_extreme.png')
    
    print(f"\nResults:")
    print(f"  BFS path cost: {bfs_cost:.4f}")
    print(f"  Dijkstra path cost: {dij_cost:.4f}")
    print(f"  Difference: {bfs_cost - dij_cost:.4f} ({((bfs_cost/dij_cost-1)*100):.2f}% longer)")
    print(f"  BFS is suboptimal: {bfs_cost > dij_cost}")


def demonstrate_theory():
    """Demonstrate the theoretical principle with clear examples"""
    print("\n" + "=" * 70)
    print("THEORETICAL DEMONSTRATION")
    print("=" * 70)
    print("\nStatement to Prove:")
    print("  'BFS is complete and optimal for uniform edge weights.'")
    print("  'BFS is NOT optimal when edge weights are non-uniform.'")
    print("\nIn 8-neighborhood grids:")
    print("  - Lateral moves (←↑→↓): cost = 1.0")
    print("  - Diagonal moves (↖↗↘↙): cost = √2 ≈ 1.414")
    print("\nBFS explores by HOPS (ignoring distance), not by COST.")
    print("Dijkstra explores by COST, guaranteeing optimal paths.")
    print("=" * 70)


# ==================== Main Function ====================

if __name__ == "__main__":
    print("🎓 Path Finding Algorithm Comparison: BFS vs Dijkstra")
    print("    (Demonstrating BFS suboptimality with non-uniform edge weights)")
    print()
    
    # Show theoretical background
    demonstrate_theory()
    
    # Demo 1: Simple case
    demo_simple_comparison()
    
    # Demo 2: With obstacles
    demo_obstacle_comparison()
    
    # Demo 3: Extreme case
    demo_extreme_comparison()
    
    print("\n" + "=" * 70)
    print("✓ All demonstrations completed!")
    print("=" * 70)
    print("\nKey Findings:")
    print("  ✓ BFS treats all edges equally (explores by 'hops')")
    print("  ✓ Dijkstra considers actual edge costs")
    print("  ✓ In 8-neighborhood grids, diagonal moves cost √2 ≈ 1.414")
    print("  ✓ BFS often chooses paths with more total cost")
    print("  ✓ Dijkstra always finds the minimum-cost path")
    print("\nGenerated files:")
    print("  📊 comparison_simple.png - Simple diagonal path comparison")
    print("  📊 comparison_obstacles.png - Path with obstacles comparison")
    print("  📊 comparison_extreme.png - Extreme case showing maximum difference")
    print("\n" + "=" * 70)
