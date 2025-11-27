import heapq
from bfs_vis import BFSVisualizer
class AStarVisualizer(BFSVisualizer):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)

    def heuristic(self, a, b):
        # Using Manhattan distance as heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star_step_by_step(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))  # (f_cost, node)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        steps = []

        while open_list:
            current_f, current = heapq.heappop(open_list)
            if current == goal:
                steps.append({
                    'current': current,
                    'queue': [],
                    'visited': g_score,
                    'parent': came_from,
                    'path': self.reconstruct_path(came_from, current),
                    'found': True,
                    'exploring': [],
                    'step_type': 'found'
                })
                return steps

            # Explore neighbors
            neighbors = self.get_neighbors(*current)
            for neighbor in neighbors:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
            
            steps.append({
                'current': current,
                'queue': open_list,
                'visited': g_score,
                'parent': came_from,
                'path': None,
                'found': False,
                'exploring': neighbors,
                'step_type': 'explore'
            })

        steps.append({
            'current': None,
            'queue': [],
            'visited': g_score,
            'parent': came_from,
            'path': None,
            'found': False,
            'exploring': [],
            'step_type': 'no_path'
        })

        return steps

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        path.reverse()
        return path
