import random
from bfs_vis import BFSVisualizer
import numpy as np
class RRTVisualizer(BFSVisualizer):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self.tree = []

    def rrt_step_by_step(self, start, goal, max_iterations=1000):
        self.tree = [start]
        steps = []

        for i in range(max_iterations):
            # Randomly sample a point
            rand_point = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            nearest_point = self.nearest(self.tree, rand_point)
            new_point = self.steer(nearest_point, rand_point)

            # If the new point is valid (not hitting an obstacle)
            if self.is_valid(new_point):
                self.tree.append(new_point)

                steps.append({
                    'current': new_point,
                    'queue': self.tree,
                    'visited': set(self.tree),
                    'parent': nearest_point,
                    'path': None,
                    'found': False,
                    'exploring': [rand_point],
                    'step_type': 'expand'
                })

            if self.distance(new_point, goal) < 1:
                steps.append({
                    'current': new_point,
                    'queue': self.tree,
                    'visited': set(self.tree),
                    'parent': nearest_point,
                    'path': self.reconstruct_rrt_path(goal),
                    'found': True,
                    'exploring': [],
                    'step_type': 'found'
                })
                break

        return steps

    def nearest(self, tree, point):
        return min(tree, key=lambda p: self.distance(p, point))

    def steer(self, from_point, to_point):
        return to_point

    def is_valid(self, point):
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x] == 0

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def reconstruct_rrt_path(self, goal):
        # Construct path from the goal to start
        path = [goal]
        return path
