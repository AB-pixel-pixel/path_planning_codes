# Path Planning Codes

A comprehensive collection of path planning and robot navigation algorithms with extensive visualizations and demonstrations for AIE1902-Embodied AI Exploration.

## 📋 Overview

This repository contains implementations of various path planning algorithms commonly used in robotics, autonomous navigation, and mobile robot applications. Each algorithm includes detailed visualizations, step-by-step demonstrations, and comparison studies to help understand their behavior and performance characteristics.

## ✨ Key Features

- **Multiple Algorithm Implementations**: Classic search-based and sampling-based path planning algorithms
- **Rich Visualizations**: Animated GIFs and step-by-step images for each algorithm
- **Comparative Analysis**: Direct comparisons between different algorithms
- **Real-world Applications**: Robot manipulation, patrol, frontier exploration, and following behaviors
- **Educational Resource**: Clear code structure with visual explanations

## 🗂️ Repository Structure

### Core Path Planning Algorithms

#### 📁 `a_star/`
Implementation of the A* (A-Star) heuristic search algorithm for optimal pathfinding.

**Contents:**
- `a_stra.py` - Basic A* implementation
- `a_stra_large.py` - A* for large-scale environments
- Visualizations:
  - `astar_simple.gif` - Simple scenario demonstration
  - `astar_maze.gif` - Maze solving with A*
  - `astar_complex.gif` - Complex environment with multiple obstacles
  - `astar_no_path.gif` - Demonstration when no path exists
  - `astar_steps_explanation.png` - Step-by-step algorithm execution

**Features:**
- Heuristic-guided optimal path search
- Manhattan and Euclidean distance heuristics
- Efficient node exploration with priority queue

---

#### 📁 `dijkstra/` & `dijkstra2/`
Dijkstra's algorithm implementation for shortest path finding in weighted graphs.

**Contents:**
- `dijkstra.py` - Core Dijkstra implementation
- `dijkstra2/` includes:
  - 4-neighbor and 8-neighbor connectivity options
  - Comparative visualizations showing different connectivity patterns
  
**Visualizations:**
- `dijkstra_simple.gif` - Basic pathfinding
- `dijkstra_maze.gif` - Maze navigation
- `dijkstra_complex.gif` - Complex obstacle scenarios
- `dijkstra_4neighbor.gif` vs `dijkstra_8neighbor.gif` - Connectivity comparison
- `dijkstra_no_path.gif` - No solution scenario

**Features:**
- Guaranteed optimal path without heuristics
- Support for both 4-connected and 8-connected grids
- Uniform-cost search implementation

---

#### 📁 `bfs/` & `bfs_2/`
Breadth-First Search (BFS) implementation for unweighted graph traversal and shortest path finding.

**Contents:**
- `bfs_vis.py` - BFS with visualization
- `bfs_detail.py` - Detailed step-by-step BFS
- `bfs_detail_larger.py` - BFS on larger graphs

**Visualizations:**
- `bfs_simple.gif` - Basic BFS demonstration
- `bfs_maze.gif` - Maze solving
- `bfs_complex.gif` - Complex environment navigation
- `graph_bfs_detailed.gif` - Detailed graph traversal animation
- `tree_bfs_detailed.gif` - BFS tree expansion visualization
- `path_reconstruction.gif` - Path backtracking demonstration
- `shortest_path_ac.gif` & `shortest_path_ag.gif` - Different graph configurations
- Frame-by-frame image sequences in `graph_bfs_frames/` and `tree_bfs_frames/`

**Features:**
- Level-by-level exploration
- Optimal for unweighted graphs
- Clear visualization of frontier expansion

---

#### 📁 `rrt/`
Rapidly-exploring Random Tree (RRT) algorithm for sampling-based path planning.

**Contents:**
- `rrt.py` - Core RRT implementation
- `flow/rrt.py` - Enhanced RRT with flowchart visualization

**Visualizations:**
- `rrt_simple.gif` - Basic RRT tree growth
- `rrt_maze.gif` - RRT in maze environments
- `rrt_complex.gif` - Complex obstacle scenarios
- `rrt_narrow_passage.gif` - RRT handling narrow passages
- `rrt_enhanced_with_flow.gif` - Algorithm flow demonstration
- `rrt_steps_explanation.png` - Detailed step breakdown
- `rrt_vs_search_comparison.png` - RRT vs search-based algorithms

**Features:**
- Probabilistic completeness
- Efficient in high-dimensional spaces
- Good for complex obstacle environments
- Tree-based exploration strategy

---

#### 📁 `RPM/` (Probabilistic Roadmap Method)
Multi-query path planning using roadmap construction.

**Contents:**
- `main.py` - PRM implementation
- `prm_roadmap.png` - Roadmap visualization (in root directory)

**Features:**
- Learning phase: roadmap construction
- Query phase: fast path retrieval
- Efficient for multiple queries in same environment

---

### Comparison Studies

#### 📁 `a_star_dijkstra_comparison/`
Side-by-side comparison of A* and Dijkstra's algorithms.

**Contents:**
- `a_stra.py` - A* implementation for comparison
- `bfs_vis.py` - BFS reference implementation
- `rrt.py` - RRT for multi-algorithm comparison
- `main.py` - Comparison execution script

---

#### 📁 `a_star_rrt/`
Comparative analysis between A* (search-based) and RRT (sampling-based) approaches.

**Contents:**
- `main.py` & `main1.py` - Comparison implementations
- Visualizations:
  - `astar_vs_rrt.gif` - Animated comparison
  - `a_rrt_display.png` - Side-by-side results
  - `rrt_a_star.png` - Performance comparison
  - `3d.png` - 3D visualization

**Insights:**
- A* is optimal but requires complete environment knowledge
- RRT is probabilistic but handles complex spaces efficiently
- Trade-offs between optimality and computational cost

---

#### 📁 `dijkstra_bfs_compare/`
Detailed comparison between Dijkstra and BFS algorithms.

**Contents:**
- `main.py` - Comparison script
- Performance visualizations:
  - `comparison_simple.png` - Simple environment comparison
  - `comparison_obstacles.png` - Obstacle-rich scenarios
  - `comparison_extreme.png` - Extreme case analysis

---

### Advanced Applications

#### 📁 `graph_grid/`
Utilities for converting between grid-based and graph-based representations.

**Contents:**
- `generate_grid.py` - Grid environment generation
- `grid_to_graph.py` - Conversion utilities
- `grid_to_graph_large.py` - Large-scale conversion
- `planning_on_graph.py` - Path planning on graph structures
- `multiple_algorithm_result.py` - Multi-algorithm testing

**Visualizations:**
- `grid.png` - Grid environment
- `grid_to_graph_simple.png` - Basic conversion
- `grid_to_graph_4connected.png` - 4-connectivity graph
- `grid_to_graph_8connected_highlight.png` - 8-connectivity with highlights
- `grid_to_graph_comparison.png` - Connectivity comparison
- `grid_to_graph_large.png` - Large-scale graph
- `display_final.png` - Planning result display

---

#### 📁 `frontier_exploration/`
Autonomous exploration using frontier-based strategies.

**Contents:**
- `main.py` - Main exploration loop
- `algorithm_flow.py` - Algorithm workflow
- `detailed_steps.py` - Step-by-step execution

**Visualizations:**
- `frontier_exploration.mp4` - Full exploration video
- Step-by-step images:
  - `step1.png` - Initial state
  - `sensing.png` - Sensor data acquisition
  - `detect_frontiers.png` - Frontier detection
  - `cluster.png` - Frontier clustering
  - `select_target.png` - Target selection
  - `plan_path.png` - Path planning to target
  - `move.png` - Robot movement execution

**Features:**
- Unknown environment mapping
- Frontier detection and clustering
- Autonomous exploration strategy
- Real-time map building

---

#### 📁 `potential_field/`
Artificial Potential Field (APF) method for real-time navigation.

**Contents:**
- `main.py` - APF implementation
- Visualizations:
  - `failed_2d.png` - 2D local minimum scenario
  - `failed_3d.png` & `failed_3d_1.png` - 3D potential field visualization

**Features:**
- Attractive force from goal
- Repulsive force from obstacles
- Real-time reactive navigation
- Local minimum problem demonstration

---

### Robot Behavior Implementations

#### 📁 `arm/`
Robotic arm manipulation and path planning.

**Contents:**
- `a*.py` - A* for arm configuration space
- `manipulate.py` - Manipulation primitives
- `sorting_demo.gif` & `sorting_demo.mp4` - Object sorting demonstration

**Features:**
- Configuration space planning
- Pick-and-place operations
- Collision-free arm motion

---

#### 📁 `following/`
Robot following behavior with obstacle avoidance.

**Contents:**
- `following.py` - Basic following implementation
- `following_v2.py` - Enhanced version with safety features
- `main.py` - Execution script

**Visualizations:**
- `robot_following.gif` - Basic following behavior
- `robot_following_collision_safe.gif` - Safe navigation with obstacles
- `robot_following_enhanced1.gif` - Enhanced tracking performance

**Features:**
- Target tracking
- Collision avoidance
- Dynamic obstacle handling
- Smooth trajectory following

---

#### 📁 `patrol/`
Autonomous patrol and inspection behavior.

**Contents:**
- `main.py` - Patrol logic implementation
- `robot_patrol_inspection.gif` - Patrol demonstration

**Features:**
- Waypoint-based patrol
- Area coverage
- Periodic inspection
- Multi-point navigation

---

#### 📁 `object_goal_finding/`
Object detection and goal-directed navigation.

**Contents:**
- `main.py` - Object search and navigation

---

### Documentation

#### 📁 `doc/`
Project documentation and design documents.

**Contents:**
- `outlines.md` - Project structure and outlines

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
numpy
matplotlib
scipy (optional)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/AB-pixel-pixel/path_planning_codes.git
cd path_planning_codes

# Install dependencies
pip install numpy matplotlib scipy
```

### Running Examples

Each directory contains standalone implementations. Navigate to any algorithm directory and run:

```bash
# Example: Run A* algorithm
cd a_star
python a_stra.py

# Example: Run RRT
cd rrt
python rrt.py

# Example: Run comparison study
cd a_star_rrt
python main.py

# Example: Run frontier exploration
cd frontier_exploration
python main.py
```

## 📊 Algorithm Comparison

| Algorithm | Type | Completeness | Optimality | Time Complexity | Best Use Case |
|-----------|------|--------------|------------|-----------------|---------------|
| **A*** | Search-based | Complete | Optimal | O(b^d) | Known environments with good heuristics |
| **Dijkstra** | Search-based | Complete | Optimal | O(V²) or O(E log V) | Weighted graphs, no heuristic available |
| **BFS** | Search-based | Complete | Optimal (unweighted) | O(V + E) | Unweighted graphs, shortest hop count |
| **RRT** | Sampling-based | Probabilistically complete | Not optimal | O(n log n) | High-dimensional spaces, complex obstacles |
| **PRM** | Sampling-based | Probabilistically complete | Not optimal | O(n log n) | Multi-query scenarios |
| **APF** | Reactive | Not complete | Not optimal | O(1) per step | Real-time local navigation |

## 🎯 Key Insights

### Search-Based vs Sampling-Based

**Search-Based (A*, Dijkstra, BFS):**
- ✅ Guarantees optimal paths (when applicable)
- ✅ Systematic exploration
- ❌ Requires discretized environment
- ❌ Struggles in high-dimensional spaces

**Sampling-Based (RRT, PRM):**
- ✅ Handles high-dimensional configuration spaces
- ✅ No need for complete discretization
- ✅ Fast in complex environments
- ❌ Paths are not optimal
- ❌ Probabilistic completeness only

### 4-Connected vs 8-Connected Grids

As demonstrated in `dijkstra2/`:
- **4-connected**: Only horizontal/vertical moves (Manhattan distance)
- **8-connected**: Includes diagonal moves (more natural paths, shorter distances)

### Frontier Exploration Strategy

The `frontier_exploration/` module demonstrates a complete autonomous exploration pipeline:
1. Sense environment
2. Detect frontiers (boundaries between known and unknown)
3. Cluster frontiers
4. Select best target frontier
5. Plan path to target
6. Execute movement
7. Repeat until fully explored

## 🛠️ Implementation Details

### Grid Representation
- 0: Free space
- 1: Obstacle
- 2: Start position (in some implementations)
- 3: Goal position (in some implementations)

### Visualization Features
- **Animated GIFs**: Show algorithm execution over time
- **Step-by-step images**: Detailed breakdowns of each algorithm phase
- **Color coding**: 
  - Blue: Unexplored
  - Green: Frontier/Open set
  - Yellow: Visited/Closed set
  - Red: Obstacles
  - Cyan: Path
  - Magenta: Start/Goal

## 📚 Learning Path

Recommended order for understanding path planning:

1. **Start with BFS** (`bfs/`, `bfs_2/`) - Understand basic graph traversal
2. **Move to Dijkstra** (`dijkstra/`) - Learn weighted graph search
3. **Learn A*** (`a_star/`) - Understand heuristic-guided search
4. **Compare Search Methods** (`dijkstra_bfs_compare/`, `a_star_dijkstra_comparison/`)
5. **Explore RRT** (`rrt/`) - Sampling-based approach
6. **Compare Paradigms** (`a_star_rrt/`) - Search vs Sampling
7. **Advanced Topics**:
   - Grid-Graph conversion (`graph_grid/`)
   - Frontier exploration (`frontier_exploration/`)
   - Potential fields (`potential_field/`)
8. **Applications**:
   - Robot behaviors (`following/`, `patrol/`)
   - Manipulation (`arm/`)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional algorithms (D*, Theta*, JPS, etc.)
- 3D path planning
- Dynamic obstacle handling
- ROS integration
- Performance benchmarks

## 📖 References

### Key Papers
1. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*
2. LaValle, S. M. (1998). *Rapidly-Exploring Random Trees: A New Tool for Path Planning*
3. Karaman, S., & Frazzoli, E. (2011). *Sampling-based Algorithms for Optimal Motion Planning*
4. Yamauchi, B. (1997). *A Frontier-Based Approach for Autonomous Exploration*

### Books
- LaValle, S. M. (2006). *Planning Algorithms*. Cambridge University Press
- Choset, H., et al. (2005). *Principles of Robot Motion*. MIT Press

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**AB-pixel-pixel**
- GitHub: [@AB-pixel-pixel](https://github.com/AB-pixel-pixel)

## 🙏 Acknowledgments

This repository is designed for educational purposes to help students and researchers understand path planning algorithms through visual demonstrations and practical implementations.

---

**Note**: This is an educational implementation. For production robotics systems, consider using established libraries like OMPL, MoveIt!, or Navigation2.

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

*Last Updated: 2025*
