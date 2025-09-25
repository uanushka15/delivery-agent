"""
Autonomous delivery agent implementation.
Handles package delivery, path planning, and rational decision making.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from environment import GridEnvironment, CellType
from algorithms import PathfindingAlgorithm, SearchResult, DynamicReplanner


class AgentState(Enum):
    """States of the delivery agent."""
    IDLE = "idle"
    MOVING = "moving"
    COLLECTING_PACKAGE = "collecting"
    DELIVERING = "delivering"
    REPLANNING = "replanning"


@dataclass
class Package:
    """Represents a package to be delivered."""
    pickup_location: Tuple[int, int]
    delivery_location: Tuple[int, int]
    priority: int = 1
    collected: bool = False
    delivered: bool = False


@dataclass
class DeliveryTask:
    """Represents a delivery task with constraints."""
    packages: List[Package]
    time_limit: Optional[int] = None
    fuel_limit: Optional[int] = None
    start_location: Tuple[int, int] = (0, 0)


class DeliveryAgent:
    """Autonomous delivery agent for navigating 2D grid city."""
    
    def __init__(self, environment: GridEnvironment, algorithm: PathfindingAlgorithm):
        self.environment = environment
        self.algorithm = algorithm
        self.position = environment.start_pos or (0, 0)
        self.state = AgentState.IDLE
        self.packages_carried = []
        self.fuel_consumed = 0
        self.time_elapsed = 0
        self.path_history = []
        self.delivery_log = []
        
    def set_position(self, position: Tuple[int, int]):
        """Set the agent's current position."""
        if self.environment.is_valid_position(position[0], position[1]):
            self.position = position
        else:
            raise ValueError(f"Invalid position: {position}")
    
    def calculate_delivery_efficiency(self, packages: List[Package]) -> List[Package]:
        """Calculate optimal delivery order based on distance and priority."""
        if not packages:
            return []
        
        # Simple heuristic: sort by priority first, then by distance from current position
        def delivery_score(package: Package) -> float:
            pickup_dist = self.manhattan_distance(self.position, package.pickup_location)
            delivery_dist = self.manhattan_distance(package.pickup_location, package.delivery_location)
            total_dist = pickup_dist + delivery_dist
            
            # Higher priority packages have lower scores (delivered first)
            # Shorter distances also have lower scores
            return total_dist / max(package.priority, 1)
        
        return sorted(packages, key=delivery_score)
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def execute_delivery_task(self, task: DeliveryTask) -> Dict:
        """Execute a complete delivery task with multiple packages."""
        self.fuel_consumed = 0
        self.time_elapsed = 0
        self.path_history = [self.position]
        self.delivery_log = []
        
        # Set starting position
        if task.start_location != self.position:
            self.set_position(task.start_location)
        
        # Calculate optimal delivery order
        ordered_packages = self.calculate_delivery_efficiency(task.packages)
        
        total_cost = 0
        successful_deliveries = 0
        failed_deliveries = 0
        
        for package in ordered_packages:
            # Check constraints
            if task.fuel_limit and self.fuel_consumed >= task.fuel_limit:
                self.delivery_log.append({
                    'package': package,
                    'status': 'failed',
                    'reason': 'fuel_limit_exceeded',
                    'time': self.time_elapsed
                })
                failed_deliveries += 1
                continue
                
            if task.time_limit and self.time_elapsed >= task.time_limit:
                self.delivery_log.append({
                    'package': package,
                    'status': 'failed',
                    'reason': 'time_limit_exceeded',
                    'time': self.time_elapsed
                })
                failed_deliveries += 1
                continue
            
            # Execute pickup and delivery
            delivery_result = self.deliver_package(package)
            total_cost += delivery_result['cost']
            
            if delivery_result['success']:
                successful_deliveries += 1
                self.delivery_log.append({
                    'package': package,
                    'status': 'delivered',
                    'cost': delivery_result['cost'],
                    'path_length': len(delivery_result['path']),
                    'time': self.time_elapsed
                })
            else:
                failed_deliveries += 1
                self.delivery_log.append({
                    'package': package,
                    'status': 'failed',
                    'reason': 'path_not_found',
                    'time': self.time_elapsed
                })
        
        return {
            'total_cost': total_cost,
            'successful_deliveries': successful_deliveries,
            'failed_deliveries': failed_deliveries,
            'fuel_consumed': self.fuel_consumed,
            'time_elapsed': self.time_elapsed,
            'path_history': self.path_history,
            'delivery_log': self.delivery_log,
            'final_position': self.position
        }
    
    def deliver_package(self, package: Package) -> Dict:
        """Deliver a single package from pickup to delivery location."""
        total_cost = 0
        full_path = []
        
        # Phase 1: Move to package pickup location
        if self.position != package.pickup_location:
            pickup_result = self.move_to_location(package.pickup_location)
            
            if not pickup_result['success']:
                return {
                    'success': False,
                    'cost': 0,
                    'path': [],
                    'reason': 'cannot_reach_pickup'
                }
            
            total_cost += pickup_result['cost']
            full_path.extend(pickup_result['path'][1:])  # Exclude starting position
        
        # Collect package
        self.state = AgentState.COLLECTING_PACKAGE
        package.collected = True
        self.packages_carried.append(package)
        
        # Phase 2: Move to delivery location
        delivery_result = self.move_to_location(package.delivery_location)
        
        if not delivery_result['success']:
            # Remove package from carried list if delivery fails
            self.packages_carried.remove(package)
            package.collected = False
            return {
                'success': False,
                'cost': total_cost,
                'path': full_path,
                'reason': 'cannot_reach_delivery'
            }
        
        total_cost += delivery_result['cost']
        full_path.extend(delivery_result['path'][1:])  # Exclude starting position
        
        # Deliver package
        self.state = AgentState.DELIVERING
        package.delivered = True
        self.packages_carried.remove(package)
        self.state = AgentState.IDLE
        
        return {
            'success': True,
            'cost': total_cost,
            'path': full_path,
            'pickup_location': package.pickup_location,
            'delivery_location': package.delivery_location
        }
    
    def move_to_location(self, target: Tuple[int, int]) -> Dict:
        """Move agent to target location using current pathfinding algorithm."""
        if self.position == target:
            return {'success': True, 'cost': 0, 'path': [self.position]}
        
        self.state = AgentState.MOVING
        
        # Find path using current algorithm
        search_result = self.algorithm.find_path(self.position, target, self.time_elapsed)
        
        if not search_result.found_solution:
            self.state = AgentState.IDLE
            return {
                'success': False,
                'cost': 0,
                'path': [],
                'algorithm_result': search_result
            }
        
        # Execute path
        path_cost = 0
        for i in range(1, len(search_result.path)):
            current_pos = search_result.path[i]
            terrain_cost = self.environment.get_terrain_cost(current_pos[0], current_pos[1])
            path_cost += terrain_cost
            self.fuel_consumed += terrain_cost
            self.time_elapsed += 1
            
            # Update position
            self.position = current_pos
            self.path_history.append(current_pos)
        
        self.state = AgentState.IDLE
        
        return {
            'success': True,
            'cost': path_cost,
            'path': search_result.path,
            'algorithm_result': search_result
        }
    
    def get_status(self) -> Dict:
        """Get current status of the agent."""
        return {
            'position': self.position,
            'state': self.state.value,
            'packages_carried': len(self.packages_carried),
            'fuel_consumed': self.fuel_consumed,
            'time_elapsed': self.time_elapsed,
            'path_length': len(self.path_history)
        }
    
    def reset(self):
        """Reset agent to initial state."""
        self.position = self.environment.start_pos or (0, 0)
        self.state = AgentState.IDLE
        self.packages_carried = []
        self.fuel_consumed = 0
        self.time_elapsed = 0
        self.path_history = []
        self.delivery_log = []


class MultiAlgorithmAgent(DeliveryAgent):
    """Delivery agent that can use multiple pathfinding algorithms."""
    
    def __init__(self, environment: GridEnvironment, algorithms: Dict[str, PathfindingAlgorithm]):
        # Use the first algorithm as default
        default_algorithm = list(algorithms.values())[0]
        super().__init__(environment, default_algorithm)
        self.algorithms = algorithms
        self.current_algorithm_name = list(algorithms.keys())[0]
        self.performance_history = {}
    
    def set_algorithm(self, algorithm_name: str):
        """Switch to a different pathfinding algorithm."""
        if algorithm_name in self.algorithms:
            self.algorithm = self.algorithms[algorithm_name]
            self.current_algorithm_name = algorithm_name
        else:
            raise ValueError(f"Algorithm {algorithm_name} not available")
    
    def compare_algorithms(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Dict[str, SearchResult]:
        """Compare all available algorithms on the same pathfinding problem."""
        results = {}
        
        for name, algorithm in self.algorithms.items():
            result = algorithm.find_path(start, goal, self.time_elapsed)
            results[name] = result
            
            # Store performance history
            if name not in self.performance_history:
                self.performance_history[name] = []
            self.performance_history[name].append(result)
        
        return results
    
    def get_best_algorithm_for_scenario(self, start: Tuple[int, int], goal: Tuple[int, int]) -> str:
        """Determine the best algorithm for a given scenario based on historical performance."""
        if not self.performance_history:
            # No history available, use default
            return self.current_algorithm_name
        
        # Simple heuristic: choose algorithm with best average performance
        best_algorithm = self.current_algorithm_name
        best_score = float('inf')
        
        for name, history in self.performance_history.items():
            if history:
                # Calculate average cost/time ratio
                avg_cost = sum(r.cost for r in history if r.found_solution) / len([r for r in history if r.found_solution])
                avg_time = sum(r.time_taken for r in history) / len(history)
                
                # Lower is better
                score = avg_cost * 0.7 + avg_time * 0.3
                
                if score < best_score:
                    best_score = score
                    best_algorithm = name
        
        return best_algorithm