import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines

class GridToGraphVisualizer:
    """Grid Map to Graph Structure Visualization Tool (Designed for Courseware)"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        
    def add_obstacle(self, x: int, y: int):
        """Add obstacle"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1
    
    def add_obstacles_rect(self, x: int, y: int, w: int, h: int):
        """Add rectangular obstacle"""
        for i in range(y, min(y + h, self.height)):
            for j in range(x, min(x + w, self.width)):
                self.grid[i, j] = 1
    
    def get_free_cells(self):
        """Get all free cells"""
        free_cells = []
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 0:
                    free_cells.append((j, i))
        return free_cells
    
    def get_neighbors(self, x: int, y: int, diagonal: bool = False):
        """Get neighbor nodes"""
        neighbors = []
        # 4-connectivity
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # 8-connectivity
        if diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 
                0 <= ny < self.height and 
                self.grid[ny, nx] == 0):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def visualize_transformation(self, diagonal: bool = False, 
                                show_coordinates: bool = True,
                                highlight_cell: tuple = None):
        """
        Visualize the transformation from grid map to graph structure
        
        Parameters:
        -----------
        diagonal : bool
            Whether to show diagonal connections (8-connectivity)
        show_coordinates : bool
            Whether to show coordinates
        highlight_cell : tuple
            Highlight a cell and its neighbors
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Left: Grid Map
        self._draw_grid_map(ax1, show_coordinates, highlight_cell, diagonal)
        
        # Right: Graph Structure
        self._draw_graph_structure(ax2, diagonal, highlight_cell)
        
        plt.tight_layout()
        return fig
    
    def _draw_grid_map(self, ax, show_coordinates, highlight_cell, diagonal):
        """Draw grid map"""
        ax.set_title('Grid Map', fontsize=16, fontweight='bold', pad=20)
        
        # Draw grid
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 1:
                    # Obstacle
                    rect = patches.Rectangle((j, i), 1, 1, 
                                            linewidth=2, edgecolor='black',
                                            facecolor='#2C3E50')
                    ax.add_patch(rect)
                else:
                    # Free cell
                    rect = patches.Rectangle((j, i), 1, 1, 
                                            linewidth=1.5, edgecolor='#34495E',
                                            facecolor='white')
                    ax.add_patch(rect)
                    
                    # Show coordinates
                    if show_coordinates:
                        ax.text(j + 0.5, i + 0.5, f'({j},{i})', 
                               ha='center', va='center',
                               fontsize=8, color='#7F8C8D')
        
        # Highlight a cell and its neighbors
        if highlight_cell:
            hx, hy = highlight_cell
            if self.grid[hy, hx] == 0:
                # Highlight center cell
                rect = patches.Rectangle((hx, hy), 1, 1, 
                                        linewidth=3, edgecolor='red',
                                        facecolor='#FFD93D', alpha=0.6)
                ax.add_patch(rect)
                
                # Highlight neighbors
                neighbors = self.get_neighbors(hx, hy, diagonal)
                for nx, ny in neighbors:
                    rect = patches.Rectangle((nx, ny), 1, 1, 
                                            linewidth=2, edgecolor='orange',
                                            facecolor='#6BCB77', alpha=0.4)
                    ax.add_patch(rect)
                    
                    # Draw connection arrows
                    arrow = FancyArrowPatch((hx + 0.5, hy + 0.5), 
                                          (nx + 0.5, ny + 0.5),
                                          arrowstyle='->', 
                                          color='red', 
                                          linewidth=2,
                                          mutation_scale=20,
                                          zorder=10)
                    ax.add_patch(arrow)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='#2C3E50', edgecolor='black', 
                         label='Obstacle', linewidth=2),
            patches.Patch(facecolor='white', edgecolor='#34495E', 
                         label='Free Cell', linewidth=1.5),
        ]
        
        if highlight_cell:
            legend_elements.extend([
                patches.Patch(facecolor='#FFD93D', edgecolor='red', 
                             label='Center Node', linewidth=3, alpha=0.6),
                patches.Patch(facecolor='#6BCB77', edgecolor='orange', 
                             label='Neighbor Node', linewidth=2, alpha=0.4),
            ])
        
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(0, 1), fontsize=10)
    
    def _draw_graph_structure(self, ax, diagonal, highlight_cell):
        """Draw graph structure"""
        connectivity = "8-Connected" if diagonal else "4-Connected"
        ax.set_title(f'Graph Structure - {connectivity}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        free_cells = self.get_free_cells()
        
        # Draw all edges
        edges_drawn = set()
        for (x, y) in free_cells:
            neighbors = self.get_neighbors(x, y, diagonal)
            for nx, ny in neighbors:
                # Avoid duplicate edges
                edge = tuple(sorted([(x, y), (nx, ny)]))
                if edge not in edges_drawn:
                    edges_drawn.add(edge)
                    
                    # Check if it's a highlighted edge
                    is_highlight = (highlight_cell and 
                                  ((x, y) == highlight_cell or 
                                   (nx, ny) == highlight_cell))
                    
                    if is_highlight:
                        ax.plot([x + 0.5, nx + 0.5], [y + 0.5, ny + 0.5],
                               'r-', linewidth=3, alpha=0.8, zorder=5)
                    else:
                        # Distinguish between 4-connected and 8-connected edges
                        if abs(x - nx) + abs(y - ny) == 2:  # Diagonal
                            ax.plot([x + 0.5, nx + 0.5], [y + 0.5, ny + 0.5],
                                   'b--', linewidth=1, alpha=0.4, zorder=1)
                        else:  # Straight
                            ax.plot([x + 0.5, nx + 0.5], [y + 0.5, ny + 0.5],
                                   'gray', linewidth=1.5, alpha=0.5, zorder=2)
        
        # Draw nodes
        for (x, y) in free_cells:
            if highlight_cell and (x, y) == highlight_cell:
                # Highlight center node
                ax.plot(x + 0.5, y + 0.5, 'o', color='red', 
                       markersize=20, zorder=10, 
                       markeredgecolor='darkred', markeredgewidth=2)
                ax.text(x + 0.5, y + 0.5, f'({x},{y})', 
                       ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
            elif highlight_cell and (x, y) in self.get_neighbors(*highlight_cell, diagonal):
                # Highlight neighbor nodes
                ax.plot(x + 0.5, y + 0.5, 'o', color='#6BCB77', 
                       markersize=16, zorder=8,
                       markeredgecolor='green', markeredgewidth=2)
                ax.text(x + 0.5, y + 0.5, f'({x},{y})', 
                       ha='center', va='center',
                       fontsize=7, fontweight='bold', color='white')
            else:
                # Normal nodes
                ax.plot(x + 0.5, y + 0.5, 'o', color='#3498DB', 
                       markersize=12, zorder=6,
                       markeredgecolor='#2874A6', markeredgewidth=1.5)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # Add legend
        legend_elements = [
            mlines.Line2D([], [], color='#3498DB', marker='o', linestyle='None',
                         markersize=10, markeredgecolor='#2874A6', 
                         markeredgewidth=1.5, label='Node'),
            mlines.Line2D([], [], color='gray', linewidth=1.5, 
                         label='4-Connected Edge'),
        ]
        
        if diagonal:
            legend_elements.append(
                mlines.Line2D([], [], color='b', linewidth=1, linestyle='--',
                             label='8-Connected Edge')
            )
        
        if highlight_cell:
            legend_elements.extend([
                mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                             markersize=10, label='Center Node'),
                mlines.Line2D([], [], color='#6BCB77', marker='o', linestyle='None',
                             markersize=10, label='Neighbor Node'),
            ])
        
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(0, 1), fontsize=10)
    
    def create_comparison_view(self):
        """Create comparison view: 4-Connected vs 8-Connected"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        # 4-Connected - Grid Map
        self._draw_grid_map(axes[0, 0], show_coordinates=False, 
                           highlight_cell=None, diagonal=False)
        axes[0, 0].set_title('Grid Map', 
                            fontsize=14, fontweight='bold')
        
        # 4-Connected - Graph Structure
        self._draw_graph_structure(axes[0, 1], diagonal=False, 
                                   highlight_cell=None)
        axes[0, 1].set_title('Graph Structure - 4-Connected', 
                            fontsize=14, fontweight='bold')
        
        # 8-Connected - Grid Map
        self._draw_grid_map(axes[1, 0], show_coordinates=False, 
                           highlight_cell=None, diagonal=True)
        axes[1, 0].set_title('Grid Map', 
                            fontsize=14, fontweight='bold')
        
        # 8-Connected - Graph Structure
        self._draw_graph_structure(axes[1, 1], diagonal=True, 
                                   highlight_cell=None)
        axes[1, 1].set_title('Graph Structure - 8-Connected', 
                            fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig


# ==================== Usage Examples ====================

def demo_basic():
    """Basic example: Simple grid map transformation"""
    print("Example 1: Basic Transformation (4-Connected)")
    
    viz = GridToGraphVisualizer(width=8, height=6)
    
    # Add obstacles
    viz.add_obstacles_rect(2, 1, 2, 3)
    viz.add_obstacles_rect(5, 3, 2, 2)
    
    # Show transformation
    fig = viz.visualize_transformation(diagonal=False, 
                                       show_coordinates=True)
    plt.savefig('grid_to_graph_4connected.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_with_highlight():
    """Example with highlighting: Show neighbor relationships"""
    print("Example 2: Highlight Neighbor Relationships (8-Connected)")
    
    viz = GridToGraphVisualizer(width=8, height=6)
    
    # Add obstacles
    viz.add_obstacles_rect(2, 1, 2, 3)
    viz.add_obstacles_rect(5, 3, 2, 2)
    
    # Highlight a cell and its neighbors
    fig = viz.visualize_transformation(diagonal=True, 
                                       show_coordinates=True,
                                       highlight_cell=(3, 4))
    plt.savefig('grid_to_graph_8connected_highlight.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_comparison():
    """Comparison example: 4-Connected vs 8-Connected"""
    print("Example 3: 4-Connected vs 8-Connected Comparison")
    
    viz = GridToGraphVisualizer(width=10, height=8)
    
    # Add some obstacles
    viz.add_obstacles_rect(3, 2, 2, 4)
    viz.add_obstacles_rect(7, 1, 1, 3)
    viz.add_obstacles_rect(6, 5, 2, 2)
    
    # Create comparison view
    fig = viz.create_comparison_view()
    plt.savefig('grid_to_graph_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_simple_clean():
    """Clean version example: Suitable for courseware presentation"""
    print("Example 4: Clean Version (Suitable for Courseware)")
    
    # Create a small simple map
    viz = GridToGraphVisualizer(width=6, height=5)
    
    # Add only a few obstacles
    viz.add_obstacle(2, 2)
    viz.add_obstacle(3, 2)
    viz.add_obstacle(4, 3)
    
    # Don't show coordinates for cleaner look
    fig = viz.visualize_transformation(diagonal=False, 
                                       show_coordinates=False)
    plt.savefig('grid_to_graph_simple.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_larger_map():
    """Large map example"""
    print("Example 5: Large Complex Map")
    
    viz = GridToGraphVisualizer(width=15, height=12)
    
    # Create more complex obstacle layout
    viz.add_obstacles_rect(3, 2, 2, 6)
    viz.add_obstacles_rect(7, 4, 3, 4)
    viz.add_obstacles_rect(11, 1, 2, 5)
    viz.add_obstacles_rect(5, 9, 4, 2)
    viz.add_obstacle(1, 7)
    viz.add_obstacle(2, 7)
    
    # Highlight connections of a node
    fig = viz.visualize_transformation(diagonal=True, 
                                       show_coordinates=False,
                                       highlight_cell=(8, 6))
    plt.savefig('grid_to_graph_large.png', dpi=300, bbox_inches='tight')
    plt.show()



# ==================== A* Path Planning + Visualization ====================

import heapq

def astar_search(viz: GridToGraphVisualizer, start, goal, diagonal=False):
    """
    A* path planning on the visualizer's grid.
    Returns: path, explored_nodes, explored_edges
    """
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g = {start: 0}
    f = {start: heuristic(start, goal)}

    explored_nodes = set()
    explored_edges = []

    while open_set:
        _, current = heapq.heappop(open_set)

        explored_nodes.add(current)

        if current == goal:
            return reconstruct_path(came_from, start, goal), explored_nodes, explored_edges

        for nx, ny in viz.get_neighbors(*current, diagonal):
            neighbor = (nx, ny)
            tentative_g = g[current] + 1

            explored_edges.append((current, neighbor))

            if neighbor not in g or tentative_g < g[neighbor]:
                g[neighbor] = tentative_g
                f[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f[neighbor], neighbor))
                came_from[neighbor] = current

    return None, explored_nodes, explored_edges


def heuristic(a, b):
    """Manhattan heuristic for 4-connected, diagonal heuristic for 8-connected"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from, start, goal):
    """Reconstruct A* path"""
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    return list(reversed(path))


# ==================== Visualization of the A* Search ====================

def visualize_pathfinding(viz: GridToGraphVisualizer, start, goal, diagonal=False):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw original grid
    for y in range(viz.height):
        for x in range(viz.width):
            if viz.grid[y, x] == 1:  # obstacle
                rect = patches.Rectangle((x, y), 1, 1, facecolor="#2C3E50")
            else:
                rect = patches.Rectangle((x, y), 1, 1, facecolor="white")
            ax.add_patch(rect)

    # Run A*
    path, explored_nodes, explored_edges = astar_search(viz, start, goal, diagonal)

    # Draw explored edges
    for (x1, y1), (x2, y2) in explored_edges:
        ax.plot([x1+0.5, x2+0.5], [y1+0.5, y2+0.5],
                linewidth=1.2, color="#5DADE2", alpha=0.6)

    # Draw explored nodes
    for x, y in explored_nodes:
        rect = patches.Rectangle((x, y), 1, 1, facecolor="#85C1E9", alpha=0.7)
        ax.add_patch(rect)

    # Draw final path
    if path:
        for (x, y) in path:
            rect = patches.Rectangle((x, y), 1, 1, facecolor="#F7DC6F", alpha=0.9)
            ax.add_patch(rect)

        # Draw path edges (red)
        for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
            ax.plot([x1+0.5, x2+0.5], [y1+0.5, y2+0.5],
                    linewidth=3, color="red")

    # Draw start & goal
    sx, sy = start
    gx, gy = goal
    ax.add_patch(patches.Rectangle((sx, sy), 1, 1, facecolor="#2ECC71"))  # green
    ax.add_patch(patches.Rectangle((gx, gy), 1, 1, facecolor="#E74C3C"))  # red

    ax.set_xlim(0, viz.width)
    ax.set_ylim(0, viz.height)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("A* Path Planning Visualization")
    plt.show()



# ==================== Main Function ====================

if __name__ == "__main__":
    # print("=" * 60)
    # print("Grid Map to Graph Structure Visualization Demo")
    # print("=" * 60)
    # print()
    
    # # Run different demonstrations
    
    # # 1. Basic demo
    # demo_basic()
    
    # # 2. Demo with highlighting
    # demo_with_highlight()
    
    # # 3. Comparison demo
    # demo_comparison()
    
    # # 4. Clean version demo
    # demo_simple_clean()
    
    # # 5. Large map demo
    # demo_larger_map()
    
    # print()
    # print("All images have been saved!")
    # print("=" * 60)

    viz = GridToGraphVisualizer(width=8, height=6)
    viz.add_obstacles_rect(2, 1, 2, 3)
    viz.add_obstacles_rect(5, 3, 2, 2)

    start = (0, 0)
    goal = (7, 5)

    visualize_pathfinding(viz, start, goal, diagonal=True)
