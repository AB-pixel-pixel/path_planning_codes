import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import deque
from typing import List, Dict, Tuple, Set, Optional
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
        
    def bfs(self, start: str, target: Optional[str] = None) -> Tuple[Set[str], List[str]]:
        """
        Execute BFS search and record each step with detailed parent tracking
        start: starting node
        target: target node (optional)
        """
        self.visited.clear()
        self.queue.clear()
        self.steps.clear()
        self.parent.clear()
        
        # Step 1: Initialize queue with start node
        self.queue.append(start)
        self.visited.add(start)
        self.parent[start] = None
        
        self.steps.append({
            'action': 'initialize',
            'queue': list(self.queue),
            'visited': set(self.visited),
            'current': None,
            'exploring': [],
            'just_added': [],
            'parent': dict(self.parent),
            'description': f'Initialize: Add {start} to queue, mark as visited, parent[{start}] = None'
        })
        
        path = []
        
        while self.queue:
            # Step 2: Dequeue front node
            current = self.queue.popleft()
            path.append(current)
            
            self.steps.append({
                'action': 'dequeue',
                'queue': list(self.queue),
                'visited': set(self.visited),
                'current': current,
                'exploring': [],
                'just_added': [],
                'parent': dict(self.parent),
                'description': f'Dequeue: Remove {current} from front of queue'
            })
            
            # Check if target found
            if target and current == target:
                self.steps.append({
                    'action': 'target_found',
                    'queue': list(self.queue),
                    'visited': set(self.visited),
                    'current': current,
                    'exploring': [],
                    'just_added': [],
                    'parent': dict(self.parent),
                    'description': f'Target Found: {current} is the goal node!'
                })
                return self.visited, path
            
            # Step 3: Examine neighbors
            neighbors = self.graph.get(current, [])
            
            if neighbors:
                self.steps.append({
                    'action': 'examine',
                    'queue': list(self.queue),
                    'visited': set(self.visited),
                    'current': current,
                    'exploring': neighbors,
                    'just_added': [],
                    'parent': dict(self.parent),
                    'description': f'Examine: Check neighbors of {current}: {neighbors}'
                })
            
            # Step 4: Process each unvisited neighbor
            just_added = []
            for i, neighbor in enumerate(neighbors):
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    self.queue.append(neighbor)
                    self.parent[neighbor] = current
                    just_added.append(neighbor)
                    
                    self.steps.append({
                        'action': 'add_neighbor',
                        'queue': list(self.queue),
                        'visited': set(self.visited),
                        'current': current,
                        'exploring': neighbors,
                        'just_added': [neighbor],
                        'parent': dict(self.parent),
                        'description': f'Add Neighbor: Mark {neighbor} as visited, set parent[{neighbor}] = {current}, enqueue {neighbor}'
                    })
                else:
                    self.steps.append({
                        'action': 'skip_neighbor',
                        'queue': list(self.queue),
                        'visited': set(self.visited),
                        'current': current,
                        'exploring': neighbors,
                        'just_added': [],
                        'parent': dict(self.parent),
                        'description': f'Skip: {neighbor} already visited (parent[{neighbor}] = {self.parent.get(neighbor, "?")})'
                    })
        
        # Final step: Queue empty
        self.steps.append({
            'action': 'complete',
            'queue': [],
            'visited': set(self.visited),
            'current': None,
            'exploring': [],
            'just_added': [],
            'parent': dict(self.parent),
            'description': 'Complete: Queue is empty, BFS finished'
        })
        
        return self.visited, path
    
    def get_path(self, start: str, end: str) -> List[str]:
        """Get path from start to end by following parent pointers"""
        if end not in self.parent:
            return []
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = self.parent[current]
        
        return path[::-1]

def create_bfs_animation(visualizer: BFSVisualizer, graph_layout: Dict[str, Tuple[float, float]], 
                        start: str, target: Optional[str] = None, interval=1500, save_as=None):
    """
    Create detailed animated BFS visualization with parent tracking
    interval: milliseconds between frames (default 1500ms)
    save_as: filename to save animation (e.g., 'bfs_animation.gif')
    """
    steps = visualizer.steps
    
    fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=(18, 9), 
                                            gridspec_kw={'width_ratios': [2, 1]})
    
    def init():
        ax_main.clear()
        ax_info.clear()
        return []
    
    def update(frame):
        ax_main.clear()
        ax_info.clear()
        
        # Configure main axis
        ax_main.set_xlim(-1.5, 1.5)
        ax_main.set_ylim(-1.8, 1.5)
        ax_main.set_aspect('equal')
        ax_main.axis('off')
        
        step = steps[frame]
        
        # Draw all edges first (faded)
        for node, (x1, y1) in graph_layout.items():
            for neighbor in visualizer.graph.get(node, []):
                x2, y2 = graph_layout[neighbor]
                ax_main.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.15, zorder=0)
        
        # Highlight parent edges (show the tree being built)
        for child, parent in step['parent'].items():
            if parent is not None:
                x1, y1 = graph_layout[parent]
                x2, y2 = graph_layout[child]
                ax_main.plot([x1, x2], [y1, y2], 'blue', linewidth=3, alpha=0.5, zorder=1)
                # Add arrow to show direction
                dx, dy = x2 - x1, y2 - y1
                ax_main.arrow(x1 + dx*0.2, y1 + dy*0.2, dx*0.5, dy*0.5, 
                            head_width=0.06, head_length=0.04, 
                            fc='blue', ec='blue', alpha=0.5, zorder=1)
        
        # Highlight edges being explored
        if step['current'] and step['exploring']:
            x1, y1 = graph_layout[step['current']]
            for neighbor in step['exploring']:
                x2, y2 = graph_layout[neighbor]
                ax_main.plot([x1, x2], [y1, y2], 'orange', linewidth=4, alpha=0.7, zorder=2)
        
        # Draw nodes
        for node, (x, y) in graph_layout.items():
            if node == step['current']:
                # Current node (orange with pulse)
                circle = patches.Circle((x, y), 0.15, color='orange', ec='black', linewidth=3, zorder=5)
                pulse_circle = patches.Circle((x, y), 0.22, color='orange', ec='orange', 
                                            linewidth=2, alpha=0.3, zorder=4)
                ax_main.add_patch(pulse_circle)
            elif node in step['just_added']:
                # Just added nodes (yellow)
                circle = patches.Circle((x, y), 0.15, color='yellow', ec='black', linewidth=3, zorder=5)
            elif node in step['visited']:
                # Visited node (light green)
                circle = patches.Circle((x, y), 0.15, color='lightgreen', ec='black', linewidth=3, zorder=3)
            else:
                # Unvisited node (light gray)
                circle = patches.Circle((x, y), 0.15, color='lightgray', ec='black', linewidth=3, zorder=2)
            
            ax_main.add_patch(circle)
            ax_main.text(x, y, node, ha='center', va='center', fontsize=20, fontweight='bold', zorder=6)
        
        # Title
        ax_main.set_title(f"BFS Algorithm Visualization - Step {frame + 1}/{len(steps)}", 
                         fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='orange', edgecolor='black', label='Current Node'),
            patches.Patch(facecolor='yellow', edgecolor='black', label='Just Added'),
            patches.Patch(facecolor='lightgreen', edgecolor='black', label='Visited'),
            patches.Patch(facecolor='lightgray', edgecolor='black', label='Unvisited'),
            plt.Line2D([0], [0], color='blue', linewidth=3, alpha=0.5, label='Parent Link')
        ]
        ax_main.legend(handles=legend_elements, loc='upper left', fontsize=11)
        
        # Information panel
        ax_info.axis('off')
        info_text = []
        
        # Action description
        info_text.append(f"Action: {step['action'].replace('_', ' ').title()}")
        info_text.append("-" * 40)
        info_text.append(f"{step['description']}")
        info_text.append("")
        
        # Queue state
        queue_str = ' ← '.join(step['queue']) if step['queue'] else 'Empty'
        info_text.append(f"Queue (FIFO):")
        info_text.append(f"  Front → [{queue_str}] ← Back")
        info_text.append("")
        
        # Visited set
        visited_str = ', '.join(sorted(step['visited']))
        info_text.append(f"Visited Set:")
        info_text.append(f"  {{{visited_str}}}")
        info_text.append("")
        
        # Parent dictionary
        info_text.append(f"Parent Dictionary:")
        if step['parent']:
            for child in sorted(step['parent'].keys()):
                parent = step['parent'][child]
                parent_str = str(parent) if parent else 'None'
                info_text.append(f"  parent[{child}] = {parent_str}")
        else:
            info_text.append("  (empty)")
        
        # Display info text
        y_pos = 0.95
        for line in info_text:
            if line.startswith('Action:'):
                ax_info.text(0.05, y_pos, line, fontsize=13, fontweight='bold', 
                           transform=ax_info.transAxes, family='monospace')
            elif line.startswith('-'):
                ax_info.text(0.05, y_pos, line, fontsize=10, 
                           transform=ax_info.transAxes, family='monospace', color='gray')
            elif any(line.startswith(x) for x in ['Queue', 'Visited', 'Parent']):
                ax_info.text(0.05, y_pos, line, fontsize=12, fontweight='bold',
                           transform=ax_info.transAxes, family='monospace', color='darkblue')
            else:
                ax_info.text(0.05, y_pos, line, fontsize=11, 
                           transform=ax_info.transAxes, family='monospace')
            y_pos -= 0.04
        
        return []
    
    anim = FuncAnimation(fig, update, init_func=init, frames=len(steps), 
                        interval=interval, blit=True, repeat=True)
    
    if save_as:
        print(f"Saving animation to {save_as}...")
        writer = PillowWriter(fps=1) # 000//interval
        anim.save(save_as, writer=writer)
        print(f"Animation saved!")
    
    plt.tight_layout()
    plt.show()
    
    return anim

def create_path_reconstruction_animation(visualizer: BFSVisualizer, 
                                        graph_layout: Dict[str, Tuple[float, float]], 
                                        start: str, target: str, 
                                        interval=1000, save_as=None):
    """
    Create animation showing path reconstruction using parent pointers
    """
    path = visualizer.get_path(start, target)
    if not path:
        print("No path found!")
        return None
    
    # Create reconstruction steps
    recon_steps = []
    current = target
    traced = []
    
    while current is not None:
        traced.append(current)
        parent = visualizer.parent[current]
        recon_steps.append({
            'current': current,
            'parent': parent,
            'traced': list(traced),
            'description': f"Trace: {current} → parent[{current}] = {parent if parent else 'None (start)'}"
        })
        current = parent
    
    fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=(18, 9),
                                            gridspec_kw={'width_ratios': [2, 1]})
    
    def init():
        ax_main.clear()
        ax_info.clear()
        return []
    
    def update(frame):
        ax_main.clear()
        ax_info.clear()
        
        ax_main.set_xlim(-1.5, 1.5)
        ax_main.set_ylim(-1.8, 1.5)
        ax_main.set_aspect('equal')
        ax_main.axis('off')
        
        if frame < len(recon_steps):
            step = recon_steps[frame]
        else:
            step = recon_steps[-1]  # Final frame
        
        # Draw all edges (faded)
        for node, (x1, y1) in graph_layout.items():
            for neighbor in visualizer.graph.get(node, []):
                x2, y2 = graph_layout[neighbor]
                ax_main.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.1, zorder=0)
        
        # Draw all parent edges
        for child, parent in visualizer.parent.items():
            if parent is not None:
                x1, y1 = graph_layout[parent]
                x2, y2 = graph_layout[child]
                ax_main.plot([x1, x2], [y1, y2], 'lightblue', linewidth=2, alpha=0.3, zorder=1)
        
        # Draw traced path
        traced_path = step['traced'][::-1]  # Reverse to show start→end
        for i in range(len(traced_path) - 1):
            x1, y1 = graph_layout[traced_path[i]]
            x2, y2 = graph_layout[traced_path[i + 1]]
            ax_main.plot([x1, x2], [y1, y2], 'red', linewidth=5, alpha=0.8, zorder=5)
            # Arrow
            dx, dy = x2 - x1, y2 - y1
            ax_main.arrow(x1, y1, dx*0.7, dy*0.7, head_width=0.08, head_length=0.06,
                        fc='red', ec='red', alpha=0.8, zorder=5)
        
        # Draw nodes
        for node, (x, y) in graph_layout.items():
            if node == step['current']:
                # Current tracing node
                circle = patches.Circle((x, y), 0.15, color='red', ec='black', linewidth=3, zorder=6)
                pulse_circle = patches.Circle((x, y), 0.22, color='red', ec='red',
                                            linewidth=2, alpha=0.3, zorder=5)
                ax_main.add_patch(pulse_circle)
            elif node in step['traced']:
                # Already traced
                circle = patches.Circle((x, y), 0.15, color='salmon', ec='black', linewidth=3, zorder=4)
            elif node in visualizer.visited:
                # Visited but not in path
                circle = patches.Circle((x, y), 0.15, color='lightgreen', ec='black', linewidth=2, zorder=2)
            else:
                # Unvisited
                circle = patches.Circle((x, y), 0.15, color='lightgray', ec='black', linewidth=2, zorder=1)
            
            ax_main.add_patch(circle)
            ax_main.text(x, y, node, ha='center', va='center', fontsize=20, fontweight='bold', zorder=7)
        
        ax_main.set_title(f"Path Reconstruction - Step {frame + 1}/{len(recon_steps)}", 
                         fontsize=16, fontweight='bold', pad=20)
        
        # Info panel
        ax_info.axis('off')
        info_text = []
        
        info_text.append("Path Reconstruction:")
        info_text.append("Following parent pointers backward")
        info_text.append("-" * 40)
        info_text.append("")
        info_text.append(f"Target: {target}")
        info_text.append(f"Start: {start}")
        info_text.append("")
        info_text.append(step['description'])
        info_text.append("")
        info_text.append("Path traced so far:")
        path_so_far = ' → '.join(step['traced'][::-1])
        info_text.append(f"  {path_so_far}")
        info_text.append("")
        
        if frame >= len(recon_steps) - 1:
            info_text.append("✓ Reconstruction Complete!")
            info_text.append("")
            final_path = ' → '.join(path)
            info_text.append(f"Final Path: {final_path}")
            info_text.append(f"Path Length: {len(path) - 1} edges")
        
        y_pos = 0.95
        for line in info_text:
            if 'Reconstruction' in line or 'Complete' in line:
                ax_info.text(0.05, y_pos, line, fontsize=13, fontweight='bold',
                           transform=ax_info.transAxes, family='monospace', color='darkred')
            elif line.startswith('-'):
                ax_info.text(0.05, y_pos, line, fontsize=10,
                           transform=ax_info.transAxes, family='monospace', color='gray')
            else:
                ax_info.text(0.05, y_pos, line, fontsize=11,
                           transform=ax_info.transAxes, family='monospace')
            y_pos -= 0.04
        
        return []
    
    anim = FuncAnimation(fig, update, init_func=init, frames=len(recon_steps) + 2,
                        interval=interval, blit=True, repeat=True)
    
    if save_as:
        print(f"Saving path reconstruction animation to {save_as}...")
        writer = PillowWriter(fps=1) # 000//interval
        anim.save(save_as, writer=writer)
        print(f"Animation saved!")
    
    plt.tight_layout()
    plt.show()
    
    return anim

# ===== Example 1: Tree Structure =====
print("=" * 70)
print("Example 1: Tree Structure BFS with Detailed Steps")
print("=" * 70)

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
print(f"Total Steps Recorded: {len(viz1.steps)}\n")

# Create detailed animation
anim1 = create_bfs_animation(viz1, layout1, 'A', interval=1500, save_as='tree_bfs_detailed.gif')

# ===== Example 2: Complex Graph with Path Reconstruction =====
print("=" * 70)
print("Example 2: Graph BFS + Path Reconstruction")
print("=" * 70)

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
print(f"Path Length: {len(shortest_path) - 1} edges")
print(f"Total Steps Recorded: {len(viz2.steps)}\n")

# BFS animation
anim2 = create_bfs_animation(viz2, layout2, 'A', target='G', interval=1800, 
                            save_as='graph_bfs_detailed.gif')

# Path reconstruction animation
anim3 = create_path_reconstruction_animation(viz2, layout2, 'A', 'G', interval=1200,
                                            save_as='path_reconstruction.gif')

print("\n" + "=" * 70)
print("Animation Complete!")
print("=" * 70)
print("Features:")
print("✓ Detailed step-by-step BFS execution")
print("✓ Parent pointer tracking at each step")
print("✓ Queue and visited set visualization")
print("✓ Path reconstruction using parent pointers")
print("✓ Information panel showing algorithm state")
