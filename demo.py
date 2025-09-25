"""
Demo script for the autonomous delivery agent system.
Runs various demonstrations of the pathfinding algorithms and delivery capabilities.
"""

import time
from environment import GridEnvironment
from algorithms import BreadthFirstSearch, UniformCostSearch, AStarSearch, HillClimbingSearch, DynamicReplanner
from agent import DeliveryAgent, MultiAlgorithmAgent, Package, DeliveryTask
from test_maps import load_test_map


def demo_algorithm_comparison():
    """Demonstrate algorithm comparison on different maps."""
    print("=" * 60)
    print("AUTONOMOUS DELIVERY AGENT DEMONSTRATION")
    print("=" * 60)
    print()
    
    maps_to_test = ['small', 'medium', 'large']
    
    for map_name in maps_to_test:
        print(f"Testing on {map_name.upper()} map:")
        print("-" * 40)
        
        try:
            env = load_test_map(map_name)
            
            # Initialize algorithms
            algorithms = {
                'BFS': BreadthFirstSearch(env),
                'UCS': UniformCostSearch(env),
                'A*': AStarSearch(env),
                'Hill-Climbing': HillClimbingSearch(env, max_restarts=5)
            }
            
            start = env.start_pos or (1, 1)
            goal = env.goal_pos or (env.width-2, env.height-2)
            
            print(f"Path from {start} to {goal}")
            
            results = {}
            for name, algorithm in algorithms.items():
                result = algorithm.find_path(start, goal)
                results[name] = result
            
            # Display results
            print(f"{'Algorithm':<15} {'Found':<6} {'Cost':<8} {'Nodes':<8} {'Time(ms)':<10}")
            print("-" * 55)
            
            for name, result in results.items():
                found = "Yes" if result.found_solution else "No"
                cost = f"{result.cost:.1f}" if result.found_solution else "∞"
                nodes = result.nodes_expanded
                time_ms = result.time_taken * 1000
                print(f"{name:<15} {found:<6} {cost:<8} {nodes:<8} {time_ms:<10.2f}")
            
            print()
            
        except Exception as e:
            print(f"Error testing {map_name} map: {e}")
            print()


def demo_delivery_task():
    """Demonstrate complete delivery task execution."""
    print("DELIVERY TASK DEMONSTRATION")
    print("=" * 40)
    
    try:
        # Use medium map for delivery demo
        env = load_test_map('medium')
        
        # Create multi-algorithm agent
        algorithms = {
            'A*': AStarSearch(env),
            'UCS': UniformCostSearch(env),
            'BFS': BreadthFirstSearch(env)
        }
        
        agent = MultiAlgorithmAgent(env, algorithms)
        agent.set_algorithm('A*')
        
        # Create packages from map
        packages = []
        for y in range(env.height):
            for x in range(env.width):
                if env.grid[y][x].cell_type.name == 'PACKAGE':
                    package = Package(
                        pickup_location=(x, y),
                        delivery_location=env.goal_pos or (env.width-1, env.height-1),
                        priority=1
                    )
                    packages.append(package)
        
        if packages:
            task = DeliveryTask(
                packages=packages,
                start_location=env.start_pos or (0, 0),
                time_limit=200,
                fuel_limit=150
            )
            
            print(f"Executing delivery task with {len(packages)} packages")
            print(f"Start: {task.start_location}")
            print(f"Constraints: Time={task.time_limit}, Fuel={task.fuel_limit}")
            print()
            
            result = agent.execute_delivery_task(task)
            
            print("DELIVERY RESULTS:")
            print(f"Successful deliveries: {result['successful_deliveries']}")
            print(f"Failed deliveries: {result['failed_deliveries']}")
            print(f"Total cost: {result['total_cost']}")
            print(f"Fuel consumed: {result['fuel_consumed']}")
            print(f"Time elapsed: {result['time_elapsed']} steps")
            
        else:
            print("No packages found in the map for delivery demonstration.")
        
        print()
        
    except Exception as e:
        print(f"Error in delivery demo: {e}")
        print()


def demo_dynamic_replanning():
    """Demonstrate dynamic replanning with moving obstacles."""
    print("DYNAMIC REPLANNING DEMONSTRATION")
    print("=" * 40)
    
    try:
        env = load_test_map('dynamic')
        
        print(f"Dynamic map loaded with {len(env.moving_obstacles)} moving obstacles")
        
        # Initialize replanning system
        primary_algorithm = AStarSearch(env)
        fallback_algorithm = HillClimbingSearch(env, max_restarts=3)
        replanner = DynamicReplanner(primary_algorithm, fallback_algorithm, env)
        
        start = env.start_pos or (1, 1)
        goal = env.goal_pos or (env.width-2, env.height-2)
        
        print(f"Planning path from {start} to {goal}")
        print("Executing with dynamic replanning...")
        
        # Execute with replanning
        result = replanner.execute_plan_with_replanning(start, goal, max_execution_steps=50)
        
        print("REPLANNING RESULTS:")
        print(f"Reached goal: {result['reached_goal']}")
        print(f"Total cost: {result['cost']}")
        print(f"Time steps: {result['time_steps']}")
        print(f"Replan events: {result['replan_count']}")
        print(f"Final path length: {len(result['path'])}")
        
        if result['replan_events']:
            print("\\nReplanning events:")
            for i, event in enumerate(result['replan_events'], 1):
                print(f"  {i}. Step {event['time_step']}: {event['reason']} at {event['position']}")
        
        print()
        
    except Exception as e:
        print(f"Error in dynamic replanning demo: {e}")
        print()


def main():
    """Run all demonstrations."""
    print("Starting Autonomous Delivery Agent Demonstration...")
    print()
    time.sleep(1)
    
    # Run demonstrations
    demo_algorithm_comparison()
    time.sleep(2)
    
    demo_delivery_task()
    time.sleep(2)
    
    demo_dynamic_replanning()
    
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("To interact with the system, use the CLI:")
    print("  python main.py run astar small          # Run A* on small map")
    print("  python main.py compare medium           # Compare algorithms")
    print("  python main.py deliver large --visualize # Run delivery task")
    print("  python main.py dynamic dynamic          # Test dynamic replanning")


if __name__ == "__main__":
    main()