"""
Main CLI interface for the autonomous delivery agent.
Provides command-line access to different pathfinding algorithms and testing.
"""

import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

from environment import GridEnvironment, CellType
from agent import Package
from algorithms import BreadthFirstSearch, UniformCostSearch, AStarSearch, HillClimbingSearch, DynamicReplanner
from agent import DeliveryAgent, MultiAlgorithmAgent, DeliveryTask
from test_maps import load_test_map, save_test_maps
from visualization import visualize_path, visualize_environment, create_performance_comparison


def run_single_algorithm(args):
    """Run a single pathfinding algorithm."""
    print(f"Loading map: {args.map}")
    env = load_test_map(args.map)
    
    # Initialize algorithm
    algorithms_map = {
        'bfs': BreadthFirstSearch(env),
        'ucs': UniformCostSearch(env),
        'astar': AStarSearch(env),
        'hill': HillClimbingSearch(env, max_restarts=args.restarts)
    }
    
    if args.algorithm not in algorithms_map:
        print(f"Unknown algorithm: {args.algorithm}")
        print(f"Available algorithms: {list(algorithms_map.keys())}")
        return
    
    algorithm = algorithms_map[args.algorithm]
    
    # Get start and goal positions
    start = env.start_pos or (1, 1)
    goal = env.goal_pos or (env.width-2, env.height-2)
    
    if args.start:
        start = tuple(map(int, args.start.split(',')))
    if args.goal:
        goal = tuple(map(int, args.goal.split(',')))
    
    print(f"Running {args.algorithm.upper()} from {start} to {goal}")
    print("Environment:")
    env.print_grid()
    
    # Run algorithm
    result = algorithm.find_path(start, goal)
    
    # Display results
    print(f"\\nResults:")
    print(f"Algorithm: {result.algorithm_name}")
    print(f"Path found: {result.found_solution}")
    print(f"Path cost: {result.cost}")
    print(f"Nodes expanded: {result.nodes_expanded}")
    print(f"Time taken: {result.time_taken:.4f} seconds")
    
    if result.found_solution:
        print(f"Path length: {len(result.path)}")
        print(f"Path: {result.path}")
        
        # Show path on grid
        print("\\nPath visualization:")
        for i, pos in enumerate(result.path):
            env.print_grid(pos)
            if i < len(result.path) - 1:
                print("↓")
        
        # Create matplotlib visualization if requested
        if args.visualize:
            visualize_path(env, result.path, f"{args.algorithm.upper()} Path")
    
    return result


def run_comparison(args):
    """Compare all algorithms on the same problem."""
    print(f"Loading map: {args.map}")
    env = load_test_map(args.map)
    
    # Initialize all algorithms
    algorithms = {
        'BFS': BreadthFirstSearch(env),
        'UCS': UniformCostSearch(env),
        'A*': AStarSearch(env),
        'Hill-Climbing': HillClimbingSearch(env, max_restarts=10)
    }
    
    # Get start and goal positions
    start = env.start_pos or (1, 1)
    goal = env.goal_pos or (env.width-2, env.height-2)
    
    if args.start:
        start = tuple(map(int, args.start.split(',')))
    if args.goal:
        goal = tuple(map(int, args.goal.split(',')))
    
    print(f"Comparing algorithms from {start} to {goal}")
    print("Environment:")
    env.print_grid()
    
    results = {}
    print(f"\\nRunning comparisons...")
    
    for name, algorithm in algorithms.items():
        print(f"\\nTesting {name}...")
        result = algorithm.find_path(start, goal)
        results[name] = result
        
        print(f"  Solution found: {result.found_solution}")
        print(f"  Path cost: {result.cost}")
        print(f"  Nodes expanded: {result.nodes_expanded}")
        print(f"  Time taken: {result.time_taken:.4f}s")
    
    # Display comparison table
    print(f"\\n{'='*60}")
    print("ALGORITHM COMPARISON")
    print(f"{'='*60}")
    print(f"{'Algorithm':<15} {'Found':<8} {'Cost':<8} {'Nodes':<8} {'Time (s)':<10}")
    print(f"{'-'*60}")
    
    for name, result in results.items():
        found = "Yes" if result.found_solution else "No"
        cost = f"{result.cost:.1f}" if result.found_solution else "∞"
        nodes = result.nodes_expanded
        time_taken = f"{result.time_taken:.4f}"
        
        print(f"{name:<15} {found:<8} {cost:<8} {nodes:<8} {time_taken:<10}")
    
    # Create performance visualization
    if args.visualize:
        create_performance_comparison(results, f"Algorithm Comparison - {args.map.title()} Map")
    
    return results


def run_delivery_task(args):
    """Run a complete delivery task simulation."""
    print(f"Loading map: {args.map}")
    env = load_test_map(args.map)
    
    # Initialize multi-algorithm agent
    algorithms = {
        'BFS': BreadthFirstSearch(env),
        'UCS': UniformCostSearch(env),
        'A*': AStarSearch(env),
        'Hill-Climbing': HillClimbingSearch(env)
    }
    
    agent = MultiAlgorithmAgent(env, algorithms)
    agent.set_algorithm(args.algorithm or 'astar')
    
    # Create delivery task from packages in environment
    packages = []
    for y in range(env.height):
        for x in range(env.width):
            if env.grid[y][x].cell_type == CellType.PACKAGE:
                # For simplicity, deliver packages to goal location
                package = Package(
                    pickup_location=(x, y),
                    delivery_location=env.goal_pos or (env.width-1, env.height-1),
                    priority=1
                )
                packages.append(package)
    
    if not packages:
        print("No packages found in the map!")
        return
    
    # Create delivery task
    task = DeliveryTask(
        packages=packages,
        start_location=env.start_pos or (0, 0),
        time_limit=args.time_limit,
        fuel_limit=args.fuel_limit
    )
    
    print(f"\\nStarting delivery task with {len(packages)} packages")
    print(f"Algorithm: {agent.current_algorithm_name}")
    print(f"Constraints: Time={args.time_limit}, Fuel={args.fuel_limit}")
    
    # Execute delivery task
    start_time = time.time()
    result = agent.execute_delivery_task(task)
    execution_time = time.time() - start_time
    
    # Display results
    print(f"\\n{'='*50}")
    print("DELIVERY TASK RESULTS")
    print(f"{'='*50}")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Successful deliveries: {result['successful_deliveries']}")
    print(f"Failed deliveries: {result['failed_deliveries']}")
    print(f"Total path cost: {result['total_cost']}")
    print(f"Fuel consumed: {result['fuel_consumed']}")
    print(f"Time elapsed: {result['time_elapsed']} steps")
    print(f"Final position: {result['final_position']}")
    
    # Show delivery log
    if result['delivery_log']:
        print(f"\\nDelivery Log:")
        for i, log_entry in enumerate(result['delivery_log'], 1):
            package = log_entry['package']
            status = log_entry['status']
            print(f"  {i}. Package ({package.pickup_location} → {package.delivery_location}): {status}")
            if 'reason' in log_entry:
                print(f"     Reason: {log_entry['reason']}")
    
    # Visualize path if requested
    if args.visualize:
        visualize_path(env, result['path_history'], "Delivery Task Path", agent_path=True)
    
    return result


def run_dynamic_replanning(args):
    """Run dynamic replanning simulation."""
    print(f"Loading dynamic map: {args.map}")
    env = load_test_map(args.map)
    
    if not env.moving_obstacles:
        print("Warning: This map has no moving obstacles. Adding some for demonstration.")
        # Add a simple moving obstacle
        env.add_moving_obstacle((5, 5), [(5, 5), (6, 5), (7, 5), (6, 5)], speed=2)
    
    # Initialize algorithms
    primary_algorithm = AStarSearch(env)
    fallback_algorithm = HillClimbingSearch(env, max_restarts=5)
    
    # Create dynamic replanner
    replanner = DynamicReplanner(primary_algorithm, fallback_algorithm, env)
    
    # Set start and goal
    start = env.start_pos or (1, 1)
    goal = env.goal_pos or (env.width-2, env.height-2)
    
    if args.start:
        start = tuple(map(int, args.start.split(',')))
    if args.goal:
        goal = tuple(map(int, args.goal.split(',')))
    
    print(f"\\nRunning dynamic replanning from {start} to {goal}")
    print(f"Moving obstacles: {len(env.moving_obstacles)}")
    
    # Execute with replanning
    result = replanner.execute_plan_with_replanning(start, goal, max_execution_steps=100)
    
    # Display results
    print(f"\\n{'='*50}")
    print("DYNAMIC REPLANNING RESULTS")
    print(f"{'='*50}")
    print(f"Reached goal: {result['reached_goal']}")
    print(f"Total path cost: {result['cost']}")
    print(f"Time steps taken: {result['time_steps']}")
    print(f"Replan events: {result['replan_count']}")
    print(f"Final path length: {len(result['path'])}")
    
    # Show replan events
    if result['replan_events']:
        print(f"\\nReplanning Events:")
        for i, event in enumerate(result['replan_events'], 1):
            print(f"  {i}. Time {event['time_step']}: {event['reason']} at {event['position']}")
            print(f"     Blocked position: {event['blocked_position']}")
    
    # Visualize if requested
    if args.visualize:
        visualize_path(env, result['path'], "Dynamic Replanning Path", 
                      show_obstacles_at_time=result['time_steps'])
    
    return result


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Autonomous Delivery Agent")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create maps command
    create_parser = subparsers.add_parser('create-maps', help='Create test maps')
    
    # Single algorithm command
    single_parser = subparsers.add_parser('run', help='Run single algorithm')
    single_parser.add_argument('algorithm', choices=['bfs', 'ucs', 'astar', 'hill'],
                              help='Algorithm to use')
    single_parser.add_argument('map', help='Map to use (small, medium, large, dynamic)')
    single_parser.add_argument('--start', help='Start position (x,y)')
    single_parser.add_argument('--goal', help='Goal position (x,y)')
    single_parser.add_argument('--restarts', type=int, default=10,
                              help='Random restarts for hill-climbing')
    single_parser.add_argument('--visualize', action='store_true',
                              help='Show matplotlib visualization')
    
    # Comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare all algorithms')
    compare_parser.add_argument('map', help='Map to use')
    compare_parser.add_argument('--start', help='Start position (x,y)')
    compare_parser.add_argument('--goal', help='Goal position (x,y)')
    compare_parser.add_argument('--visualize', action='store_true',
                               help='Show performance charts')
    
    # Delivery task command
    delivery_parser = subparsers.add_parser('deliver', help='Run delivery task')
    delivery_parser.add_argument('map', help='Map to use')
    delivery_parser.add_argument('--algorithm', choices=['bfs', 'ucs', 'astar', 'hill'],
                                default='astar', help='Algorithm to use')
    delivery_parser.add_argument('--time-limit', type=int, help='Time limit for task')
    delivery_parser.add_argument('--fuel-limit', type=int, help='Fuel limit for task')
    delivery_parser.add_argument('--visualize', action='store_true',
                                help='Show path visualization')
    
    # Dynamic replanning command
    dynamic_parser = subparsers.add_parser('dynamic', help='Run dynamic replanning')
    dynamic_parser.add_argument('map', help='Map to use (preferably dynamic)')
    dynamic_parser.add_argument('--start', help='Start position (x,y)')
    dynamic_parser.add_argument('--goal', help='Goal position (x,y)')
    dynamic_parser.add_argument('--visualize', action='store_true',
                               help='Show path visualization')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'create-maps':
            save_test_maps()
        elif args.command == 'run':
            run_single_algorithm(args)
        elif args.command == 'compare':
            run_comparison(args)
        elif args.command == 'deliver':
            run_delivery_task(args)
        elif args.command == 'dynamic':
            run_dynamic_replanning(args)
        else:
            parser.print_help()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()