import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import heapq
from typing import List, Tuple, Set, Dict, Optional
import matplotlib.lines as mlines

class AStarVisualizer:
    """A* Algorithm Visualizer (Designed for Teaching Materials)"""
    
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
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Heuristic function: Manhattan distance
        (Admissible for 4-connected grid)
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get neighbor nodes (4-connected)"""
        neighbors = []
        # Order: up, right, down, left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 
                0 <= ny < self.height and 
                self.grid[ny, nx] == 0):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to current"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def astar_step_by_step(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        A* step-by-step search, returns state of each step
        
        Returns:
            List of step information dictionaries
        """
        # Priority queue: (f_score, counter, node)
        counter = 0
        open_set = [(0, counter, start)]
        counter += 1
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        open_set_hash = {start}
        closed_set = set()
        
        steps = []
        
        # Record initial state
        steps.append({
            'current': start,
            'open_set': [(start, f_score[start], g_score[start])],
            'closed_set': closed_set.copy(),
            'came_from': came_from.copy(),
            'g_score': g_score.copy(),
            'f_score': f_score.copy(),
            'path': None,
            'found': False,
            'exploring': [],
            'step_type': 'init'
        })
        
        while open_set:
            # Pop node with lowest f_score
            current_f, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            
            # Record dequeue state
            steps.append({
                'current': current,
                'open_set': [(node, f_score.get(node, float('inf')), g_score.get(node, float('inf'))) 
                            for _, _, node in open_set],
                'closed_set': closed_set.copy(),
                'came_from': came_from.copy(),
                'g_score': g_score.copy(),
                'f_score': f_score.copy(),
                'path': None,
                'found': False,
                'exploring': [],
                'step_type': 'dequeue'
            })
            
            # Found goal
            if current == goal:
                path = self.reconstruct_path(came_from, current)
                steps.append({
                    'current': current,
                    'open_set': [(node, f_score.get(node, float('inf')), g_score.get(node, float('inf'))) 
                                for _, _, node in open_set],
                    'closed_set': closed_set.copy(),
                    'came_from': came_from.copy(),
                    'g_score': g_score.copy(),
                    'f_score': f_score.copy(),
                    'path': path,
                    'found': True,
                    'exploring': [],
                    'step_type': 'found'
                })
                return steps
            
            closed_set.add(current)
            
            # Explore neighbors
            neighbors = self.get_neighbors(*current)
            new_neighbors = []
            
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                        counter += 1
                        open_set_hash.add(neighbor)
                        new_neighbors.append(neighbor)
            
            # Record state after exploring neighbors
            if new_neighbors:
                steps.append({
                    'current': current,
                    'open_set': [(node, f_score.get(node, float('inf')), g_score.get(node, float('inf'))) 
                                for _, _, node in open_set],
                    'closed_set': closed_set.copy(),
                    'came_from': came_from.copy(),
                    'g_score': g_score.copy(),
                    'f_score': f_score.copy(),
                    'path': None,
                    'found': False,
                    'exploring': new_neighbors,
                    'step_type': 'explore'
                })
        
        # No path found
        steps.append({
            'current': None,
            'open_set': [],
            'closed_set': closed_set.copy(),
            'came_from': came_from.copy(),
            'g_score': g_score.copy(),
            'f_score': f_score.copy(),
            'path': None,
            'found': False,
            'exploring': [],
            'step_type': 'no_path'
        })
        
        return steps
    
    def visualize_static_explanation(self, show_edges=True):
        """Create static A* algorithm explanation diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        # Run A* to get all steps
        steps = self.astar_step_by_step(self.start, self.goal)
        
        # Find key frames
        init_step = steps[0]
        mid_step = steps[len(steps) // 2]
        near_end_step = steps[-3] if len(steps) > 3 else steps[-1]
        final_step = steps[-1]
        
        # Draw four key stages
        self._draw_astar_state(axes[0, 0], init_step, "Step 1: Initialization", show_edges)
        self._draw_astar_state(axes[0, 1], mid_step, f"Step {len(steps)//2}: Searching", show_edges)
        self._draw_astar_state(axes[1, 0], near_end_step, f"Step {len(steps)-2}: Near Goal", show_edges)
        self._draw_astar_state(axes[1, 1], final_step, "Final: Path Found", show_edges)
        
        plt.tight_layout()
        return fig
    
    def _draw_astar_state(self, ax, step_info, title, show_edges=True):
        """Draw A* state at a given step"""
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        current = step_info['current']
        open_set_nodes = [node for node, _, _ in step_info['open_set']]
        closed_set = step_info['closed_set']
        exploring = step_info['exploring']
        path = step_info['path']
        f_score = step_info['f_score']
        g_score = step_info['g_score']
        
        # Draw all edges (graph structure) - only for smaller grids
        if show_edges and self.width * self.height <= 500:
            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i, j] == 0:
                        neighbors = self.get_neighbors(j, i)
                        for nx, ny in neighbors:
                            ax.plot([j + 0.5, nx + 0.5], [i + 0.5, ny + 0.5],
                                   'lightgray', linewidth=1, alpha=0.3, zorder=1)
        
        # Draw grid and obstacles
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 1:
                    rect = patches.Rectangle((j, i), 1, 1,
                                            linewidth=0.5 if self.width > 50 else 1,
                                            edgecolor='black',
                                            facecolor='#2C3E50')
                    ax.add_patch(rect)
        
        # Determine marker size based on grid size
        if self.width * self.height > 10000:
            marker_size_closed = 3
            marker_size_open = 3
            marker_size_exploring = 3
            marker_size_current = 5
            marker_size_start = 6
            marker_size_goal = 6
            text_size = 6
        elif self.width * self.height > 1000:
            marker_size_closed = 5
            marker_size_open = 5
            marker_size_exploring = 5
            marker_size_current = 8
            marker_size_start = 10
            marker_size_goal = 10
            text_size = 8
        else:
            marker_size_closed = 15
            marker_size_open = 14
            marker_size_exploring = 12
            marker_size_current = 20
            marker_size_start = 22
            marker_size_goal = 22
            text_size = 12
        
        # Draw closed set (visited nodes)
        for (x, y) in closed_set:
            if (x, y) != self.start and (x, y) != self.goal:
                ax.plot(x + 0.5, y + 0.5, 'o', color='#AED6F1', 
                       markersize=marker_size_closed, zorder=5, alpha=0.7)
        
        # Draw open set nodes
        for (x, y) in open_set_nodes:
            if (x, y) != self.start and (x, y) != self.goal:
                ax.plot(x + 0.5, y + 0.5, 's', color='#F9E79F', 
                       markersize=marker_size_open, zorder=6, alpha=0.8)
        
        # Draw neighbors being explored
        for (x, y) in exploring:
            ax.plot(x + 0.5, y + 0.5, 'D', color='#82E0AA', 
                   markersize=marker_size_exploring, zorder=7, alpha=0.9)
        
        # Draw current node
        if current:
            cx, cy = current
            ax.plot(cx + 0.5, cy + 0.5, 'o', color='#FF6B6B', 
                   markersize=marker_size_current, zorder=8,
                   markeredgecolor='darkred', markeredgewidth=2)
            
            # Show f, g, h values for current node (only for smaller grids)
            if self.width * self.height <= 500:
                g = g_score.get(current, 0)
                h = self.heuristic(current, self.goal)
                f = f_score.get(current, 0)
                ax.text(cx + 0.5, cy + 1.2, f'f={f}\ng={g}\nh={h}',
                       ha='center', va='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Draw start point
        sx, sy = self.start
        ax.plot(sx + 0.5, sy + 0.5, 'o', color='#2ECC71', 
               markersize=marker_size_start, zorder=10,
               markeredgecolor='darkgreen', markeredgewidth=2)
        if self.width * self.height <= 1000:
            ax.text(sx + 0.5, sy + 0.5, 'S', ha='center', va='center',
                   fontsize=text_size, fontweight='bold', color='white')
        
        # Draw goal point
        gx, gy = self.goal
        ax.plot(gx + 0.5, gy + 0.5, 's', color='#E74C3C', 
               markersize=marker_size_goal, zorder=10,
               markeredgecolor='darkred', markeredgewidth=2)
        if self.width * self.height <= 1000:
            ax.text(gx + 0.5, gy + 0.5, 'G', ha='center', va='center',
                   fontsize=text_size, fontweight='bold', color='white')
        
        # If path found, draw path
        if path:
            path_x = [x + 0.5 for x, y in path]
            path_y = [y + 0.5 for x, y in path]
            linewidth = 2 if self.width * self.height > 1000 else 4
            ax.plot(path_x, path_y, 'r-', linewidth=linewidth, alpha=0.7, zorder=9)
        
        ax.set_xlim(-0.5, self.width + 0.5)
        ax.set_ylim(-0.5, self.height + 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)
        
        # Add legend
        legend_fontsize = 6 if self.width * self.height > 1000 else 9
        legend_markersize = 6 if self.width * self.height > 1000 else 12
        
        legend_elements = [
            mlines.Line2D([], [], color='#2ECC71', marker='o', linestyle='None',
                         markersize=legend_markersize, label='Start', 
                         markeredgecolor='darkgreen', markeredgewidth=2),
            mlines.Line2D([], [], color='#E74C3C', marker='s', linestyle='None',
                         markersize=legend_markersize, label='Goal',
                         markeredgecolor='darkred', markeredgewidth=2),
            mlines.Line2D([], [], color='#FF6B6B', marker='o', linestyle='None',
                         markersize=legend_markersize, label='Current Node',
                         markeredgecolor='darkred', markeredgewidth=2),
            mlines.Line2D([], [], color='#F9E79F', marker='s', linestyle='None',
                         markersize=legend_markersize-2, label='Open Set'),
            mlines.Line2D([], [], color='#82E0AA', marker='D', linestyle='None',
                         markersize=legend_markersize-4, label='New Neighbors'),
            mlines.Line2D([], [], color='#AED6F1', marker='o', linestyle='None',
                         markersize=legend_markersize-2, label='Closed Set'),
        ]
        
        if path:
            legend_elements.append(
                mlines.Line2D([], [], color='red', linewidth=3, 
                             label=f'Optimal Path ({len(path)-1} steps)')
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=legend_fontsize)
        
        # Add statistics
        info_text = f"Open Set: {len(open_set_nodes)} nodes\n"
        info_text += f"Closed Set: {len(closed_set)} nodes\n"
        info_text += f"Total Explored: {len(closed_set) + len(open_set_nodes)}\n"
        if exploring:
            info_text += f"New Discovered: {len(exploring)}"
        
        fontsize = 8 if self.width * self.height > 1000 else 10
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=fontsize, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def create_astar_animation(self, filename='astar_animation.gif', fps=2, show_edges=True):
        """Create animation of A* search process"""
        steps = self.astar_step_by_step(self.start, self.goal)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def animate(frame_num):
            ax.clear()
            
            step_info = steps[frame_num]
            
            # Set title to show current step
            step_type = step_info['step_type']
            title_dict = {
                'init': f'Step {frame_num + 1}/{len(steps)}: Initialize - Start A* Search',
                'dequeue': f'Step {frame_num + 1}/{len(steps)}: Pop Node with Lowest f-score',
                'explore': f'Step {frame_num + 1}/{len(steps)}: Explore & Update Neighbors',
                'found': f'Step {frame_num + 1}/{len(steps)}: ✓ Optimal Path Found!',
                'no_path': f'Step {frame_num + 1}/{len(steps)}: ✗ No Path to Goal'
            }
            
            ax.set_title(title_dict.get(step_type, f'Step {frame_num + 1}/{len(steps)}'),
                        fontsize=16, fontweight='bold', pad=20)
            
            current = step_info['current']
            open_set_nodes = step_info['open_set']
            closed_set = step_info['closed_set']
            exploring = step_info['exploring']
            path = step_info['path']
            f_score = step_info['f_score']
            g_score = step_info['g_score']
            
            # Draw graph structure edges - only for smaller grids
            if show_edges and self.width * self.height <= 500:
                for i in range(self.height):
                    for j in range(self.width):
                        if self.grid[i, j] == 0:
                            neighbors = self.get_neighbors(j, i)
                            for nx, ny in neighbors:
                                ax.plot([j + 0.5, nx + 0.5], [i + 0.5, ny + 0.5],
                                       'lightgray', linewidth=1.5, alpha=0.3, zorder=1)
            
            # Draw obstacles
            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i, j] == 1:
                        rect = patches.Rectangle((j, i), 1, 1,
                                                linewidth=0.5 if self.width > 50 else 1.5,
                                                edgecolor='black',
                                                facecolor='#2C3E50')
                        ax.add_patch(rect)
            
            # Determine marker size based on grid size
            if self.width * self.height > 10000:
                marker_size_closed = 3
                marker_size_open = 3
                marker_size_exploring = 3
                marker_size_current = 6
                marker_size_start = 8
                marker_size_goal = 8
                text_size = 6
            elif self.width * self.height > 1000:
                marker_size_closed = 5
                marker_size_open = 5
                marker_size_exploring = 5
                marker_size_current = 10
                marker_size_start = 12
                marker_size_goal = 12
                text_size = 8
            else:
                marker_size_closed = 18
                marker_size_open = 16
                marker_size_exploring = 14
                marker_size_current = 24
                marker_size_start = 24
                marker_size_goal = 24
                text_size = 14
            
            # Draw closed set (visited nodes)
            for (x, y) in closed_set:
                if (x, y) != self.start and (x, y) != self.goal:
                    ax.plot(x + 0.5, y + 0.5, 'o', color='#AED6F1', 
                           markersize=marker_size_closed, zorder=5, alpha=0.7,
                           markeredgecolor='#5DADE2', markeredgewidth=1)
            
            # Draw open set nodes with color gradient based on f-score
            for node, f, g in open_set_nodes:
                x, y = node
                if (x, y) != self.start and (x, y) != self.goal:
                    ax.plot(x + 0.5, y + 0.5, 's', color='#F9E79F', 
                           markersize=marker_size_open, zorder=6, alpha=0.9,
                           markeredgecolor='#F4D03F', markeredgewidth=1.5)
            
            # Draw neighbors being explored (with animation effect)
            for (x, y) in exploring:
                ax.plot(x + 0.5, y + 0.5, 'D', color='#82E0AA', 
                       markersize=marker_size_exploring, zorder=7, alpha=0.95,
                       markeredgecolor='#27AE60', markeredgewidth=2)
                # Add arrow pointing to new neighbor (only for smaller grids)
                if current and self.width * self.height <= 500:
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
                       markersize=marker_size_current + 4, zorder=8, alpha=0.5)
                ax.plot(cx + 0.5, cy + 0.5, 'o', color='#FF6B6B', 
                       markersize=marker_size_current, zorder=8,
                       markeredgecolor='darkred', markeredgewidth=2)
                
                # Show f, g, h values (only for smaller grids)
                if self.width * self.height <= 500:
                    g = g_score.get(current, 0)
                    h = self.heuristic(current, self.goal)
                    f = f_score.get(current, 0)
                    ax.text(cx + 0.5, cy + 1.2, f'f={f}\ng={g}\nh={h}',
                           ha='center', va='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Draw start point
            sx, sy = self.start
            ax.plot(sx + 0.5, sy + 0.5, 'o', color='#2ECC71', 
                   markersize=marker_size_start, zorder=10,
                   markeredgecolor='darkgreen', markeredgewidth=3)
            if self.width * self.height <= 1000:
                ax.text(sx + 0.5, sy + 0.5, 'S', ha='center', va='center',
                       fontsize=text_size, fontweight='bold', color='white', zorder=11)
            
            # Draw goal point
            gx, gy = self.goal
            ax.plot(gx + 0.5, gy + 0.5, 's', color='#E74C3C', 
                   markersize=marker_size_goal, zorder=10,
                   markeredgecolor='darkred', markeredgewidth=3)
            if self.width * self.height <= 1000:
                ax.text(gx + 0.5, gy + 0.5, 'G', ha='center', va='center',
                       fontsize=text_size, fontweight='bold', color='white', zorder=11)
            
            # If path found, draw path
            if path:
                path_x = [x + 0.5 for x, y in path]
                path_y = [y + 0.5 for x, y in path]
                linewidth = 2 if self.width * self.height > 1000 else 5
                ax.plot(path_x, path_y, 'r-', linewidth=linewidth, alpha=0.8, zorder=9)
                
                # Draw nodes on path (only for smaller grids)
                if self.width * self.height <= 1000:
                    for x, y in path[1:-1]:
                        ax.plot(x + 0.5, y + 0.5, 'o', color='#FF6B6B',
                               markersize=marker_size_exploring + 2, zorder=9, alpha=0.9,
                               markeredgecolor='darkred', markeredgewidth=2)
            
            ax.set_xlim(-0.5, self.width + 0.5)
            ax.set_ylim(-0.5, self.height + 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            
            # Add legend
            legend_fontsize = 8 if self.width * self.height > 1000 else 10
            legend_markersize = 8 if self.width * self.height > 1000 else 12
            
            legend_elements = [
                mlines.Line2D([], [], color='#2ECC71', marker='o', linestyle='None',
                             markersize=legend_markersize, label='Start', 
                             markeredgecolor='darkgreen', markeredgewidth=2),
                mlines.Line2D([], [], color='#E74C3C', marker='s', linestyle='None',
                             markersize=legend_markersize, label='Goal',
                             markeredgecolor='darkred', markeredgewidth=2),
                mlines.Line2D([], [], color='#FF6B6B', marker='o', linestyle='None',
                             markersize=legend_markersize, label='Current',
                             markeredgecolor='darkred', markeredgewidth=2),
                mlines.Line2D([], [], color='#F9E79F', marker='s', linestyle='None',
                             markersize=legend_markersize-2, label='Open Set'),
                mlines.Line2D([], [], color='#82E0AA', marker='D', linestyle='None',
                             markersize=legend_markersize-4, label='New Found'),
                mlines.Line2D([], [], color='#AED6F1', marker='o', linestyle='None',
                             markersize=legend_markersize-2, label='Closed Set'),
            ]
            
            if path:
                legend_elements.append(
                    mlines.Line2D([], [], color='red', linewidth=3, 
                                 label=f'Optimal Path ({len(path)-1} steps)')
                )
            
            ax.legend(handles=legend_elements, loc='upper right', fontsize=legend_fontsize,
                     framealpha=0.95)
            
            # Add detailed A* information
            info_text = f"A* Algorithm Status:\n"
            info_text += f"━━━━━━━━━━━━━━━━━━\n"
            info_text += f"⟳ Open Set: {len(open_set_nodes)}\n"
            info_text += f"✓ Closed Set: {len(closed_set)}\n"
            info_text += f"⊕ Total Explored: {len(closed_set) + len(open_set_nodes)}\n"
            
            if exploring:
                info_text += f"⊕ New Found: {len(exploring)}\n"
            
            if current:
                g = g_score.get(current, 0)
                h = self.heuristic(current, self.goal)
                f = f_score.get(current, 0)
                info_text += f"⊙ Current: {current}\n"
                info_text += f"  f={f}, g={g}, h={h}\n"
            
            if path:
                info_text += f"━━━━━━━━━━━━━━━━━━\n"
                info_text += f"✓ Path Length: {len(path)-1} steps"
            
            fontsize = 9 if self.width * self.height > 1000 else 11
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=fontsize, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', 
                           alpha=0.95, edgecolor='orange', linewidth=2))
       
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(steps),
                            interval=1000/fps, repeat=True)
        
        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=300)
        plt.close()
        
        print(f"✓ Animation saved as: {filename}")
        print(f"  Total steps: {len(steps)}")
        print(f"  Frame rate: {fps} FPS")
        
        return anim
    

def create_maze_obstacles(viz: AStarVisualizer):
   """Create a complex maze environment"""
   width = viz.width
   height = viz.height
   
   # Create maze walls
   # Vertical walls
   for i in range(5, height - 5, 8):
       viz.add_obstacles_rect(10, i, 1, 6)
       viz.add_obstacles_rect(20, i - 3, 1, 6)
       viz.add_obstacles_rect(30, i, 1, 6)
       viz.add_obstacles_rect(40, i - 3, 1, 6)
   
   # Horizontal walls
   for j in range(5, width - 5, 15):
       viz.add_obstacles_rect(j, 10, 8, 1)
       viz.add_obstacles_rect(j + 5, 20, 8, 1)
       viz.add_obstacles_rect(j, 30, 8, 1)
       viz.add_obstacles_rect(j + 5, 40, 8, 1)
   
   # Add some random obstacle blocks
   obstacle_positions = [
       (15, 15, 3, 3), (35, 25, 4, 2), (25, 35, 2, 4),
       (45, 15, 3, 3), (5, 25, 2, 3), (40, 40, 3, 3)
   ]
   
   for x, y, w, h in obstacle_positions:
       if x + w < width and y + h < height:
           viz.add_obstacles_rect(x, y, w, h)


def create_large_environment_obstacles(viz: AStarVisualizer):
   """Create obstacles for large environment (500x1500)"""
   width = viz.width
   height = viz.height
   
   # Create major vertical corridors
   for x in range(100, width, 200):
       # Leave gaps for passage
       viz.add_obstacles_rect(x, 0, 5, height // 3)
       viz.add_obstacles_rect(x, 2 * height // 3, 5, height // 3)
   
   # Create horizontal barriers
   for y in range(80, height, 150):
       # Create barriers with gaps
       for segment in range(0, width, 300):
           viz.add_obstacles_rect(segment, y, 200, 5)
   
   # Add building-like structures
   building_positions = [
       (50, 50, 80, 60),
       (200, 100, 100, 80),
       (400, 150, 90, 70),
       (600, 80, 100, 90),
       (800, 120, 85, 75),
       (1000, 50, 95, 85),
       (1200, 100, 110, 95),
       (1400, 75, 80, 70),
       
       (100, 250, 75, 65),
       (300, 280, 90, 80),
       (500, 260, 95, 75),
       (700, 290, 85, 70),
       (900, 270, 100, 85),
       (1100, 255, 90, 75),
       (1300, 280, 95, 80),
   ]
   
   for x, y, w, h in building_positions:
       if x + w < width and y + h < height:
           viz.add_obstacles_rect(x, y, w, h)
   
   # Add some scattered obstacles
   np.random.seed(42)
   for _ in range(50):
       x = np.random.randint(0, width - 20)
       y = np.random.randint(0, height - 20)
       w = np.random.randint(10, 30)
       h = np.random.randint(10, 30)
       
       if x + w < width and y + h < height:
           viz.add_obstacles_rect(x, y, w, h)


# ==================== Usage Examples ====================

def demo_maze_astar():
   """Maze scenario A* demonstration"""
   print("=" * 60)
   print("Example 1: Maze Environment A* Path Planning")
   print("=" * 60)
   
   viz = AStarVisualizer(width=50, height=50)
   
   # Create maze obstacles
   create_maze_obstacles(viz)
   
   # Set start and goal
   viz.set_start(2, 2)
   viz.set_goal(47, 47)
   
   # Create static explanation diagram
   print("Generating static step diagram...")
   fig = viz.visualize_static_explanation(show_edges=False)
   plt.savefig('astar_maze_steps.png', dpi=300, bbox_inches='tight')
   plt.show()
   print("✓ Static diagram saved: astar_maze_steps.png")
   
   # Create animation
   print("\nGenerating GIF animation...")
   viz.create_astar_animation('astar_maze.gif', fps=3, show_edges=False)
   print("✓ Done!")


def demo_large_environment_astar():
   """Large environment (500x1500) A* demonstration"""
   print("\n" + "=" * 60)
   print("Example 2: Large Environment (500x1500) A* Path Planning")
   print("=" * 60)
   
   viz = AStarVisualizer(width=1500, height=500)
   
   # Create obstacles for large environment
   print("Creating obstacles for large environment...")
   create_large_environment_obstacles(viz)
   
   # Set start and goal - long distance
   viz.set_start(50, 250)
   viz.set_goal(1450, 250)
   
   # Create static explanation diagram
   print("Generating static step diagram (this may take a moment)...")
   fig = viz.visualize_static_explanation(show_edges=False)
   plt.savefig('astar_large_steps.png', dpi=300, bbox_inches='tight')
   plt.show()
   print("✓ Static diagram saved: astar_large_steps.png")
   
   # Create animation (with fewer frames for efficiency)
   print("\nGenerating GIF animation (this may take several minutes)...")
   viz.create_astar_animation('astar_large.gif', fps=5, show_edges=False)
   print("✓ Done!")


def demo_comparison_maze():
   """Comparison: Different maze configurations"""
   print("\n" + "=" * 60)
   print("Example 3: Maze Comparison - Different Configurations")
   print("=" * 60)
   
   # Configuration 1: Sparse maze
   print("\n3a. Sparse Maze Configuration")
   viz1 = AStarVisualizer(width=50, height=50)
   
   # Simple vertical barriers
   for x in range(10, 45, 12):
       viz1.add_obstacles_rect(x, 5, 2, 20)
       viz1.add_obstacles_rect(x, 30, 2, 15)
   
   viz1.set_start(2, 25)
   viz1.set_goal(47, 25)
   
   fig1 = viz1.visualize_static_explanation(show_edges=False)
   plt.savefig('astar_maze_sparse.png', dpi=300, bbox_inches='tight')
   plt.show()
   viz1.create_astar_animation('astar_maze_sparse.gif', fps=3, show_edges=False)
   
   # Configuration 2: Dense maze
   print("\n3b. Dense Maze Configuration")
   viz2 = AStarVisualizer(width=50, height=50)
   
   # Create denser maze
   for i in range(0, 50, 5):
       for j in range(0, 50, 8):
           if (i + j) % 2 == 0:
               viz2.add_obstacles_rect(j, i, 3, 3)
   
   # Clear path regions
   for i in range(5, 45, 10):
       for j in range(5, 45, 10):
           viz2.grid[i:i+2, j:j+2] = 0
   
   viz2.set_start(1, 1)
   viz2.set_goal(48, 48)
   
   fig2 = viz2.visualize_static_explanation(show_edges=False)
   plt.savefig('astar_maze_dense.png', dpi=300, bbox_inches='tight')
   plt.show()
   viz2.create_astar_animation('astar_maze_dense.gif', fps=3, show_edges=False)
   
   print("✓ Comparison demonstrations completed!")


def demo_corridor_scenario():
   """Narrow corridor scenario to show A* efficiency"""
   print("\n" + "=" * 60)
   print("Example 4: Narrow Corridor - A* Optimization Showcase")
   print("=" * 60)
   
   viz = AStarVisualizer(width=100, height=30)
   
   # Create narrow winding corridor
   # Top wall
   viz.add_obstacles_rect(0, 0, 100, 5)
   # Bottom wall
   viz.add_obstacles_rect(0, 25, 100, 5)
   
   # Zigzag barriers
   for x in range(10, 90, 15):
       if (x // 15) % 2 == 0:
           viz.add_obstacles_rect(x, 5, 2, 12)
       else:
           viz.add_obstacles_rect(x, 13, 2, 12)
   
   viz.set_start(5, 15)
   viz.set_goal(95, 15)
   
   print("Generating static diagram...")
   fig = viz.visualize_static_explanation(show_edges=False)
   plt.savefig('astar_corridor.png', dpi=300, bbox_inches='tight')
   plt.show()
   
   print("Generating animation...")
   viz.create_astar_animation('astar_corridor.gif', fps=4, show_edges=False)
   print("✓ Done!")


def demo_open_space_astar():
   """Open space with scattered obstacles"""
   print("\n" + "=" * 60)
   print("Example 5: Open Space with Scattered Obstacles")
   print("=" * 60)
   
   viz = AStarVisualizer(width=60, height=60)
   
   # Scattered circular-ish obstacles
   obstacle_centers = [
       (15, 15, 4), (30, 10, 5), (45, 15, 4),
       (10, 30, 5), (25, 25, 6), (40, 30, 4),
       (15, 45, 5), (35, 45, 4), (50, 50, 5)
   ]
   
   for cx, cy, radius in obstacle_centers:
       for i in range(max(0, cy - radius), min(60, cy + radius)):
           for j in range(max(0, cx - radius), min(60, cx + radius)):
               if (i - cy) ** 2 + (j - cx) ** 2 <= radius ** 2:
                   viz.add_obstacle(j, i)
   
   viz.set_start(2, 2)
   viz.set_goal(57, 57)
   
   print("Generating static diagram...")
   fig = viz.visualize_static_explanation(show_edges=False)
   plt.savefig('astar_open_space.png', dpi=300, bbox_inches='tight')
   plt.show()
   
   print("Generating animation...")
   viz.create_astar_animation('astar_open_space.gif', fps=4, show_edges=False)
   print("✓ Done!")


# ==================== Main Function ====================

if __name__ == "__main__":
   print("🎓 A* Path Planning Algorithm Visualization")
   print("=" * 60)
   print()
   
   # Example 1: Maze environment (50x50)
   demo_maze_astar()
   
   # Example 2: Large environment (500x1500)
   demo_large_environment_astar()
   
   # Example 3: Different maze configurations
   demo_comparison_maze()
   
   # Example 4: Corridor scenario
   demo_corridor_scenario()
   
   # Example 5: Open space scenario
   demo_open_space_astar()
   
   print("\n" + "=" * 60)
   print("✓ All A* demonstrations completed!")
   print("=" * 60)
   print("\nGenerated files:")
   print("  📊 astar_maze_steps.png - Maze step explanation")
   print("  📊 astar_large_steps.png - Large environment steps")
   print("  📊 astar_maze_sparse.png - Sparse maze")
   print("  📊 astar_maze_dense.png - Dense maze")
   print("  📊 astar_corridor.png - Corridor scenario")
   print("  📊 astar_open_space.png - Open space scenario")
   print("  🎬 astar_maze.gif - Maze animation")
   print("  🎬 astar_large.gif - Large environment animation")
   print("  🎬 astar_maze_sparse.gif - Sparse maze animation")
   print("  🎬 astar_maze_dense.gif - Dense maze animation")
   print("  🎬 astar_corridor.gif - Corridor animation")
   print("  🎬 astar_open_space.gif - Open space animation")
   print("\n💡 A* Algorithm Features:")
   print("  • Uses f(n) = g(n) + h(n) heuristic")
   print("  • Guarantees optimal path (with admissible heuristic)")
   print("  • More efficient than BFS for large spaces")
   print("  • Explores fewer nodes by using goal-directed search")