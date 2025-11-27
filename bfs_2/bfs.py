import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import deque
from typing import List, Dict, Tuple, Set
import numpy as np

class BFSVisualizer:
    def __init__(self, graph: Dict[str, List[str]]):
        """
        Initialize BFS visualizer
        graph: adjacency list representation, e.g., {'A': ['B', 'C'], 'B': ['D'], ...}
        """
        self.graph = graph
        self.visited = set()
        self.queue = deque()
        self.steps = []  # Record each step
        self.parent = {}  # Record parent nodes for path construction
        
    def bfs(self, start: str, target: str = None) -> Tuple[Set[str], List[str]]:
        """
        Execute BFS search and record each step
        start: starting node
        target: target node (optional)
        """
        self.visited.clear()
        self.queue.clear()
        self.steps.clear()
        self.parent.clear()
        
        self.queue.append(start)
        self.visited.add(start)
        self.parent[start] = None
        
        # Initial state
        self.steps.append({
            'queue': list(self.queue),
            'visited': set(self.visited),
            'current': None,
            'exploring': [],
            'just_added': []
        })
        
        path = []
        
        while self.queue:
            current = self.queue.popleft()
            path.append(current)
            
            just_added = []
            # Explore neighbors
            for neighbor in self.graph.get(current, []):
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    self.queue.append(neighbor)
                    self.parent[neighbor] = current
                    just_added.append(neighbor)
            
            # Record current step
            self.steps.append({
                'queue': list(self.queue),
                'visited': set(self.visited),
                'current': current,
                'exploring': list(self.graph.get(current, [])),
                'just_added': just_added
            })
            
            # Early return if target found
            if target and current == target:
                return self.visited, path
        
        return self.visited, path
    
    def get_path(self, start: str, end: str) -> List[str]:
        """Get path from start to end"""
        if end not in self.parent:
            return []
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = self.parent[current]
        
        return path[::-1]

def create_bfs_animation(visualizer: BFSVisualizer, graph_layout: Dict[str, Tuple[float, float]], 
                        start: str, target: str = None, interval=1000, save_as=None):
    """
    Create animated BFS visualization
    interval: milliseconds between frames (default 1000ms = 1 second)
    save_as: filename to save animation (e.g., 'bfs_animation.gif')
    """
    steps = visualizer.steps
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    def init():
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.8, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        return []
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.8, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        step = steps[frame]
        
        # Draw edges first
        for node, (x1, y1) in graph_layout.items():
            for neighbor in visualizer.graph.get(node, []):
                x2, y2 = graph_layout[neighbor]
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.3, zorder=0)
        
        # Highlight edges being explored
        if step['current'] and step['exploring']:
            x1, y1 = graph_layout[step['current']]
            for neighbor in step['exploring']:
                x2, y2 = graph_layout[neighbor]
                ax.plot([x1, x2], [y1, y2], 'orange', linewidth=3, alpha=0.6, zorder=1)
        
        # Draw nodes
        for node, (x, y) in graph_layout.items():
            if node == step['current']:
                # Current node (orange)
                circle = patches.Circle((x, y), 0.15, color='orange', ec='black', linewidth=3, zorder=3)
                pulse_circle = patches.Circle((x, y), 0.20, color='orange', ec='orange', 
                                            linewidth=2, alpha=0.3, zorder=2)
                ax.add_patch(pulse_circle)
            elif node in step['just_added']:
                # Just added nodes (yellow)
                circle = patches.Circle((x, y), 0.15, color='yellow', ec='black', linewidth=3, zorder=3)
            elif node in step['visited']:
                # Visited node (green)
                circle = patches.Circle((x, y), 0.15, color='lightgreen', ec='black', linewidth=3, zorder=2)
            else:
                # Unvisited node (gray)
                circle = patches.Circle((x, y), 0.15, color='lightgray', ec='black', linewidth=3, zorder=1)
            
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontsize=20, fontweight='bold', zorder=4)
        
        # Title and information
        queue_str = ' → '.join(step['queue']) if step['queue'] else 'Empty'
        visited_str = ', '.join(sorted(step['visited']))
        
        if step['current']:
            title = f"Step {frame}/{len(steps)-1}\n"
            title += f"Current Node: {step['current']}\n"
            title += f"Queue: [{queue_str}]\n"
            title += f"Visited: {{{visited_str}}}"
            if step['just_added']:
                title += f"\nJust Added to Queue: {{{', '.join(step['just_added'])}}}"
        else:
            title = f"Step {frame}/{len(steps)-1}\n"
            title += f"Starting BFS from node: {start}\n"
            title += f"Queue: [{queue_str}]"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='orange', edgecolor='black', label='Current Node'),
            patches.Patch(facecolor='yellow', edgecolor='black', label='Just Added'),
            patches.Patch(facecolor='lightgreen', edgecolor='black', label='Visited'),
            patches.Patch(facecolor='lightgray', edgecolor='black', label='Unvisited')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        return []
    
    anim = FuncAnimation(fig, update, init_func=init, frames=len(steps), 
                        interval=interval, blit=True, repeat=True)
    
    if save_as:
        print(f"Saving animation to {save_as}...")
        writer = PillowWriter(fps=1)
        anim.save(save_as, writer=writer)
        print(f"Animation saved!")
    
    plt.tight_layout()
    plt.show()
    
    return anim

def create_final_path_animation(visualizer: BFSVisualizer, graph_layout: Dict[str, Tuple[float, float]], 
                               start: str, target: str, interval=800, save_as=None):
    """
    Create animation showing the final shortest path
    """
    path = visualizer.get_path(start, target)
    if not path:
        print("No path found!")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    def init():
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.8, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        return []
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.8, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw all edges
        for node, (x1, y1) in graph_layout.items():
            for neighbor in visualizer.graph.get(node, []):
                x2, y2 = graph_layout[neighbor]
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.2, zorder=0)
        
        # Draw path edges up to current frame
        for i in range(min(frame, len(path) - 1)):
            x1, y1 = graph_layout[path[i]]
            x2, y2 = graph_layout[path[i + 1]]
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=5, alpha=0.7, zorder=5)
            # Add arrow
            dx, dy = x2 - x1, y2 - y1
            ax.arrow(x1, y1, dx*0.7, dy*0.7, head_width=0.08, head_length=0.06, 
                    fc='red', ec='red', alpha=0.7, zorder=5)
        
        # Draw nodes
        for node, (x, y) in graph_layout.items():
            if frame < len(path) and node == path[frame]:
                # Current path node (red with pulse)
                circle = patches.Circle((x, y), 0.15, color='red', ec='black', linewidth=3, zorder=6)
                pulse_circle = patches.Circle((x, y), 0.22, color='red', ec='red', 
                                            linewidth=2, alpha=0.3, zorder=5)
                ax.add_patch(pulse_circle)
            elif node in path[:frame]:
                # Already traced path node (red)
                circle = patches.Circle((x, y), 0.15, color='salmon', ec='black', linewidth=3, zorder=4)
            elif node in visualizer.visited:
                # Visited but not in path (green)
                circle = patches.Circle((x, y), 0.15, color='lightgreen', ec='black', linewidth=3, zorder=2)
            else:
                # Unvisited (gray)
                circle = patches.Circle((x, y), 0.15, color='lightgray', ec='black', linewidth=3, zorder=1)
            
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontsize=20, fontweight='bold', zorder=7)
        
        # Title
        current_path = ' → '.join(path[:frame+1])
        title = f"Shortest Path Visualization\n"
        title += f"From {start} to {target}\n"
        title += f"Path: {current_path}\n"
        title += f"Length: {min(frame, len(path)-1)} edges"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        return []
    
    anim = FuncAnimation(fig, update, init_func=init, frames=len(path)+3, 
                        interval=interval, blit=True, repeat=True)
    
    if save_as:
        print(f"Saving path animation to {save_as}...")
        writer = PillowWriter(fps=1)
        anim.save(save_as, writer=writer)
        print(f"Path animation saved!")
    
    plt.tight_layout()
    plt.show()
    
    return anim

# ===== Example 1: Tree Structure =====
print("=" * 60)
print("Example 1: Tree Structure BFS Animation")
print("=" * 60)

graph1 = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

layout1 = {
    'A': (0, 1),
    'B': (-0.5, 0),
    'C': (0.5, 0),
    'D': (-0.8, -1),
    'E': (-0.2, -1),
    'F': (0.5, -1)
}

viz1 = BFSVisualizer(graph1)
visited1, path1 = viz1.bfs('A')

print(f"Visit Order: {' → '.join(path1)}")
print(f"Visited Nodes: {sorted(visited1)}\n")

# Create animation (1 second per frame)
anim1 = create_bfs_animation(viz1, layout1, 'A', interval=1000)
# Uncomment to save: create_bfs_animation(viz1, layout1, 'A', interval=1000, save_as='tree_bfs.gif')

# ===== Example 2: Complex Graph with Shortest Path =====
print("=" * 60)
print("Example 2: Complex Graph BFS + Shortest Path Animation")
print("=" * 60)

graph2 = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E', 'G'],
    'G': ['F']
}

layout2 = {
    'A': (0, 1),
    'B': (-0.5, 0),
    'C': (0.5, 0),
    'D': (-1, -1),
    'E': (-0.2, -1),
    'F': (0.5, -0.5),
    'G': (1, -1.3)
}

viz2 = BFSVisualizer(graph2)
visited2, path2 = viz2.bfs('A', target='G')

shortest_path = viz2.get_path('A', 'G')
print(f"Visit Order: {' → '.join(path2)}")
print(f"Shortest Path from A to G: {' → '.join(shortest_path)}")
print(f"Path Length: {len(shortest_path) - 1} edges\n")

# BFS search animation
anim2 = create_bfs_animation(viz2, layout2, 'A', target='G', interval=1200,save_as='graph_bfs.gif')
# Uncomment to save: create_bfs_animation(viz2, layout2, 'A', target='G', interval=1200, save_as='graph_bfs.gif')

# Shortest path animation
anim3 = create_final_path_animation(viz2, layout2, 'A', 'G', interval=800,save_as='shortest_path_ag.gif')
anim3 = create_final_path_animation(viz2, layout2, 'A', 'C', interval=800,save_as='shortest_path_ac.gif')
# Uncomment to save: create_final_path_animation(viz2, layout2, 'A', 'G', interval=800, save_as='shortest_path.gif')

print("\nAnimation Tips:")
print("- The animation will loop automatically")
print("- Close the window to see the next animation")
print("- Uncomment save_as parameter to save animations as GIF files")
print("- Adjust 'interval' parameter to change animation speed (in milliseconds)")
