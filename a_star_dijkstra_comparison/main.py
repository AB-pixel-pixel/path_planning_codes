from bfs_vis import *
from a_stra import AStarVisualizer
from rrt import RRTVisualizer
import numpy as np
def compare_algorithms():
    width, height = 15, 12
    viz_a_star = AStarVisualizer(width, height)
    viz_rrt = RRTVisualizer(width, height)

    # Set up obstacles, start, and goal
    viz_a_star.add_obstacles_rect(3, 2, 3, 2)
    viz_a_star.add_obstacles_rect(8, 3, 2, 5)
    viz_a_star.set_start(1, 1)
    viz_a_star.set_goal(13, 10)

    # Visualize A* Algorithm
    print("Running A* Algorithm...")
    a_star_steps = viz_a_star.a_star_step_by_step((1, 1), (13, 10))
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    viz_a_star._draw_bfs_state(axes[0], a_star_steps[-1], "A* Path Finding")
    
    # Visualize RRT Algorithm
    print("Running RRT Algorithm...")
    rrt_steps = viz_rrt.rrt_step_by_step((1, 1), (13, 10))
    viz_rrt._draw_bfs_state(axes[1], rrt_steps[-1], "RRT Path Finding")
    
    plt.tight_layout()
    plt.show()

compare_algorithms()
