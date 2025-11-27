import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import deque
from typing import List, Tuple, Dict, Set
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
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1

    def add_obstacles_rect(self, x: int, y: int, w: int, h: int):
        for i in range(y, min(y + h, self.height)):
            for j in range(x, min(x + w, self.width)):
                self.grid[i, j] = 1

    def set_start(self, x: int, y: int):
        self.start = (x, y)

    def set_goal(self, x: int, y: int):
        self.goal = (x, y)

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        # 4-connected: up, right, down, left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] == 0):
                neighbors.append((nx, ny))
        return neighbors

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star_step_by_step(self, start: Tuple[int,int], goal: Tuple[int,int]):
        """
        A* step-by-step. Returns list of steps where each step is a dict:
        {
            'current': node or None,
            'open': [(node, g, h, f), ...],
            'closed': set(...),
            'g': {node: g_value, ...},
            'f': {node: f_value, ...},
            'parent': {node: parent_node, ...},
            'path': list or None,
            'found': bool,
            'exploring': [neighbor nodes just added],
            'step_type': 'init'|'pop'|'expand'|'found'|'no_path'
        }
        """
        open_heap = []
        open_set = set()
        closed_set: Set[Tuple[int,int]] = set()
        parent: Dict[Tuple[int,int], Tuple[int,int] or None] = {}
        g_score: Dict[Tuple[int,int], float] = {}
        f_score: Dict[Tuple[int,int], float] = {}

        # initialize
        g_score[start] = 0
        h0 = self.heuristic(start, goal)
        f_score[start] = h0
        heapq.heappush(open_heap, (f_score[start], h0, start))  # tie-break by h
        open_set.add(start)
        parent[start] = None

        steps = []
        # initial snapshot
        steps.append({
            'current': None,
            'open': [(start, g_score[start], h0, f_score[start])],
            'closed': closed_set.copy(),
            'g': g_score.copy(),
            'f': f_score.copy(),
            'parent': parent.copy(),
            'path': None,
            'found': False,
            'exploring': [],
            'step_type': 'init'
        })

        while open_heap:
            # pop lowest f (tie break by h)
            f_cur, h_cur, current = heapq.heappop(open_heap)
            if current in closed_set:
                # stale entry (we may have pushed an updated duplicate)
                continue

            # Record pop action
            open_set.discard(current)
            steps.append({
                'current': current,
                'open': [(n, g_score.get(n, float('inf')), 
                          self.heuristic(n, goal), f_score.get(n, float('inf'))) for (_,_,n) in open_heap if n not in closed_set] + \
                        [(n, g_score[n], self.heuristic(n, goal), f_score[n]) for n in open_set],
                'closed': closed_set.copy(),
                'g': g_score.copy(),
                'f': f_score.copy(),
                'parent': parent.copy(),
                'path': None,
                'found': False,
                'exploring': [],
                'step_type': 'pop'
            })

            # Found?
            if current == goal:
                # reconstruct path
                path = []
                n = current
                while n is not None:
                    path.append(n)
                    n = parent.get(n)
                path.reverse()
                steps.append({
                    'current': current,
                    'open': [],
                    'closed': closed_set.copy(),
                    'g': g_score.copy(),
                    'f': f_score.copy(),
                    'parent': parent.copy(),
                    'path': path,
                    'found': True,
                    'exploring': [],
                    'step_type': 'found'
                })
                return steps

            # Expand current
            closed_set.add(current)
            neighbors = self.get_neighbors(*current)
            new_added = []
            for nbr in neighbors:
                if nbr in closed_set:
                    continue
                tentative_g = g_score[current] + 1  # grid cost = 1 per move

                if tentative_g < g_score.get(nbr, float('inf')):
                    parent[nbr] = current
                    g_score[nbr] = tentative_g
                    h_nbr = self.heuristic(nbr, goal)
                    f_score[nbr] = tentative_g + h_nbr
                    heapq.heappush(open_heap, (f_score[nbr], h_nbr, nbr))
                    open_set.add(nbr)
                    new_added.append(nbr)

            # record expansion step
            steps.append({
                'current': current,
                'open': [(n, g_score.get(n, float('inf')), self.heuristic(n, goal), f_score.get(n, float('inf'))) for n in open_set],
                'closed': closed_set.copy(),
                'g': g_score.copy(),
                'f': f_score.copy(),
                'parent': parent.copy(),
                'path': None,
                'found': False,
                'exploring': new_added,
                'step_type': 'expand'
            })

        # no path
        steps.append({
            'current': None,
            'open': [],
            'closed': closed_set.copy(),
            'g': g_score.copy(),
            'f': f_score.copy(),
            'parent': parent.copy(),
            'path': None,
            'found': False,
            'exploring': [],
            'step_type': 'no_path'
        })
        return steps

    # ---------------- Drawing helpers ----------------
    def _draw_grid_and_edges(self, ax):
        # draw light graph edges for free cells
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 0:
                    neighbors = self.get_neighbors(j, i)
                    for nx, ny in neighbors:
                        ax.plot([j + 0.5, nx + 0.5], [i + 0.5, ny + 0.5],
                                'lightgray', linewidth=1, alpha=0.25, zorder=1)

        # draw obstacles
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 1:
                    rect = patches.Rectangle((j, i), 1, 1,
                                             linewidth=1, edgecolor='black',
                                             facecolor='#2C3E50', zorder=2)
                    ax.add_patch(rect)

    def _draw_a_star_state(self, ax, step_info, title):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

        current = step_info['current']
        open_list = step_info['open']  # list of tuples (node, g, h, f)
        closed = step_info['closed']
        g = step_info['g']
        f = step_info['f']
        parent = step_info['parent']
        path = step_info['path']
        exploring = step_info['exploring']

        self._draw_grid_and_edges(ax)

        # closed (visited)
        for node in closed:
            if node != self.start and node != self.goal:
                x, y = node
                ax.plot(x + 0.5, y + 0.5, 'o', color='#AED6F1', markersize=14, zorder=5, alpha=0.8)

        # open nodes (fringe) -- square marker and annotate g/h/f
        for (node, gval, hval, fval) in open_list:
            x, y = node
            if node != self.start and node != self.goal:
                ax.plot(x + 0.5, y + 0.5, 's', color='#F9E79F', markersize=14, zorder=6, alpha=0.95)
                # annotate values (small)
                txt = f"g={gval}\nh={hval}\nf={int(fval)}"
                ax.text(x + 0.5, y + 0.35, txt, ha='center', va='top', fontsize=7, zorder=15,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        # exploring neighbors highlight (diamond)
        for nb in exploring:
            x, y = nb
            ax.plot(x + 0.5, y + 0.5, 'D', color='#82E0AA', markersize=12, zorder=7, alpha=0.95)

        # current node
        if current:
            cx, cy = current
            ax.plot(cx + 0.5, cy + 0.5, 'o', color='#FF6B6B', markersize=20, zorder=8,
                    markeredgecolor='darkred', markeredgewidth=2)
            # annotate current g/h/f if available
            gcur = g.get(current, None)
            fcur = f.get(current, None)
            hcur = None
            if gcur is not None and fcur is not None:
                hcur = int(fcur - gcur)
                ax.text(cx + 0.5, cy - 0.15, f"g={gcur}, h={hcur}, f={int(fcur)}",
                        ha='center', va='top', fontsize=9, fontweight='bold', zorder=16,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

        # start and goal
        sx, sy = self.start
        ax.plot(sx + 0.5, sy + 0.5, 'o', color='#2ECC71', markersize=22, zorder=10,
               markeredgecolor='darkgreen', markeredgewidth=3)
        ax.text(sx + 0.5, sy + 0.5, 'S', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')

        gx, gy = self.goal
        ax.plot(gx + 0.5, gy + 0.5, 's', color='#E74C3C', markersize=22, zorder=10,
               markeredgecolor='darkred', markeredgewidth=3)
        ax.text(gx + 0.5, gy + 0.5, 'G', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')

        # path if found - draw as red thick line
        if path:
            px = [x + 0.5 for x, y in path]
            py = [y + 0.5 for x, y in path]
            ax.plot(px, py, 'r-', linewidth=4, alpha=0.8, zorder=12)
            for x, y in path[1:-1]:
                ax.plot(x + 0.5, y + 0.5, 'o', color='#FF6B6B', markersize=12, zorder=13)

        ax.set_xlim(-0.5, self.width + 0.5)
        ax.set_ylim(-0.5, self.height + 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)

        # Legend
        legend_elements = [
            mlines.Line2D([], [], color='#2ECC71', marker='o', linestyle='None', markersize=12, label='Start', markeredgecolor='darkgreen', markeredgewidth=2),
            mlines.Line2D([], [], color='#E74C3C', marker='s', linestyle='None', markersize=12, label='Goal', markeredgecolor='darkred', markeredgewidth=2),
            mlines.Line2D([], [], color='#FF6B6B', marker='o', linestyle='None', markersize=12, label='Current', markeredgecolor='darkred', markeredgewidth=2),
            mlines.Line2D([], [], color='#F9E79F', marker='s', linestyle='None', markersize=10, label='Open (fringe)'),
            mlines.Line2D([], [], color='#AED6F1', marker='o', linestyle='None', markersize=10, label='Closed (visited)'),
            mlines.Line2D([], [], color='#82E0AA', marker='D', linestyle='None', markersize=8, label='Newly Discovered'),
        ]
        if path:
            legend_elements.append(mlines.Line2D([], [], color='red', linewidth=3, label=f'Path ({len(path)-1} steps)'))

        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # Info box showing counts and a small table of top open nodes with g/h/f (sorted by f)
        top_open = sorted(open_list, key=lambda t: t[3])[:6]  # top 6 by f
        info_lines = [f"A* Status:",
                      f"━━━━━━━━━━━━━━━━━━",
                      f"Closed size: {len(closed)}",
                      f"Open size: {len(open_list)}"]
        if current:
            info_lines.append(f"Current: {current}")
        if exploring:
            info_lines.append(f"New discovered: {len(exploring)}")
        info_lines.append("Top Open (node: g,h,f):")
        for (n, gv, hv, fv) in top_open:
            info_lines.append(f"{n}: {gv},{hv},{int(fv)}")
        if path:
            info_lines.append("━━━━━━━━━━━━━━━━━━")
            info_lines.append(f"Path length: {len(path)-1} steps")

        info_text = "\n".join(info_lines)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='orange', linewidth=1))

    # ---------------- Public visualizers ----------------
    def visualize_static_explanation(self):
        """Create static diagram with 4 snapshots from the A* steps"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        steps = self.a_star_step_by_step(self.start, self.goal)
        # pick 4 representative frames: init, early, mid, final
        init = steps[0]
        if len(steps) > 3:
            early = steps[min(2, len(steps)-1)]
            mid = steps[len(steps)//2]
            final = steps[-1]
        else:
            early = mid = final = steps[-1]

        self._draw_a_star_state(axes[0,0], init, "Step 1: Initialization")
        self._draw_a_star_state(axes[0,1], early, f"Early Step")
        self._draw_a_star_state(axes[1,0], mid, f"Mid Step")
        self._draw_a_star_state(axes[1,1], final, f"Final Step")
        plt.tight_layout()
        return fig

    def create_a_star_animation(self, filename='astar_animation.gif', fps=2):
        steps = self.a_star_step_by_step(self.start, self.goal)
        fig, ax = plt.subplots(figsize=(12, 10))

        def animate(frame_num):
            ax.clear()
            step_info = steps[frame_num]
            step_type = step_info['step_type']
            title_map = {
                'init': f"Step {frame_num+1}/{len(steps)}: Initialize",
                'pop': f"Step {frame_num+1}/{len(steps)}: Pop lowest f",
                'expand': f"Step {frame_num+1}/{len(steps)}: Expand current",
                'found': f"Step {frame_num+1}/{len(steps)}: ✓ Goal Found!",
                'no_path': f"Step {frame_num+1}/{len(steps)}: ✗ No Path"
            }
            ax.set_title(title_map.get(step_type, f"Step {frame_num+1}/{len(steps)}"), fontsize=16, fontweight='bold', pad=18)
            self._draw_a_star_state(ax, step_info, "")
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)

        anim = FuncAnimation(fig, animate, frames=len(steps), interval=1000/fps, repeat=True)
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=100)
        plt.close()
        print(f"✓ Animation saved as: {filename}")
        print(f"  Total steps: {len(steps)}")
        print(f"  Frame rate: {fps} FPS")
        return anim

# ================= Usage Examples =================
def demo_simple_astar():
    print("="*60)
    print("Example A*: Simple Scenario")
    print("="*60)
    viz = AStarVisualizer(width=10, height=8)
    viz.add_obstacles_rect(3, 2, 2, 4)
    viz.add_obstacles_rect(6, 1, 1, 3)
    viz.add_obstacle(7, 5)
    viz.add_obstacle(8, 5)
    viz.set_start(1, 3)
    viz.set_goal(8, 3)
    print("Generating static explanation figure...")
    fig = viz.visualize_static_explanation()
    plt.savefig('astar_steps_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Generating GIF animation...")
    viz.create_a_star_animation('astar_simple.gif', fps=2)
    print("Done.")

def demo_maze_astar():
    print("="*60)
    print("Example A*: Maze Scenario")
    print("="*60)
    viz = AStarVisualizer(width=12, height=10)
    viz.add_obstacles_rect(2, 1, 1, 6)
    viz.add_obstacles_rect(4, 3, 1, 6)
    viz.add_obstacles_rect(6, 1, 1, 5)
    viz.add_obstacles_rect(8, 4, 1, 5)
    viz.add_obstacles_rect(10, 2, 1, 4)
    viz.set_start(0,0)
    viz.set_goal(11,9)
    print("Generating GIF animation...")
    viz.create_a_star_animation('astar_maze.gif', fps=3)
    print("Done.")

def demo_no_path_astar():
    print("="*60)
    print("Example A*: No-Path Scenario")
    print("="*60)
    viz = AStarVisualizer(width=10, height=8)
    viz.add_obstacles_rect(4, 2, 1, 4)
    viz.add_obstacles_rect(5, 2, 3, 1)
    viz.add_obstacles_rect(5, 5, 3, 1)
    viz.add_obstacles_rect(7, 3, 1, 2)
    viz.set_start(1, 3)
    viz.set_goal(6, 4)
    print("Generating GIF animation (no path)...")
    viz.create_a_star_animation('astar_no_path.gif', fps=2)
    print("Done.")

def demo_complex_astar():
    print("="*60)
    print("Example A*: Complex Scenario")
    print("="*60)
    viz = AStarVisualizer(width=15, height=12)
    viz.add_obstacles_rect(3, 2, 3, 2)
    viz.add_obstacles_rect(3, 6, 3, 2)
    viz.add_obstacles_rect(8, 3, 2, 5)
    viz.add_obstacles_rect(11, 1, 2, 4)
    viz.add_obstacles_rect(11, 7, 2, 4)
    viz.add_obstacle(6, 4)
    viz.add_obstacle(6, 5)
    viz.set_start(1,1)
    viz.set_goal(13,10)
    print("Generating static explanation figure...")
    fig = viz.visualize_static_explanation()
    plt.savefig('astar_complex_steps.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Generating GIF animation...")
    viz.create_a_star_animation('astar_complex.gif', fps=2.5)
    print("Done.")

if __name__ == "__main__":
    print("🎓 A* Path Planning Algorithm Visualization")
    print("="*60)
    demo_simple_astar()
    demo_maze_astar()
    demo_complex_astar()
    demo_no_path_astar()
    print("All demos completed.")
