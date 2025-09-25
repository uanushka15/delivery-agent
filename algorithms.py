"""
Pathfinding algorithms for the autonomous delivery agent.
Includes BFS, Uniform-cost search, A*, and local search methods.
"""

import heapq
import time
import random
from collections import deque
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

from environment import GridEnvironment


@dataclass
class SearchResult:
    """Result of a pathfinding algorithm."""
    path: List[Tuple[int, int]]
    cost: float
    nodes_expanded: int
    time_taken: float
    algorithm_name: str
    found_solution: bool


class PathfindingAlgorithm(ABC):
    """Abstract base class for pathfinding algorithms."""
    
    def __init__(self, environment: GridEnvironment):
        self.environment = environment
        self.nodes_expanded = 0
        
    @abstractmethod
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], time_step: int = 0) -> SearchResult:
        """Find path from start to goal."""
        pass
    
    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary."""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return list(reversed(path))
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class BreadthFirstSearch(PathfindingAlgorithm):
    """Breadth-First Search algorithm."""
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], time_step: int = 0) -> SearchResult:
        start_time = time.time()
        self.nodes_expanded = 0
        
        if start == goal:
            return SearchResult([start], 0, 0, 0, "BFS", True)
        
        queue = deque([start])
        visited = {start}
        came_from = {}
        
        while queue:
            current = queue.popleft()
            self.nodes_expanded += 1
            
            # Get neighbors at current time step
            neighbors = self.environment.get_neighbors(current[0], current[1], time_step)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    
                    if neighbor == goal:
                        path = self.reconstruct_path(came_from, neighbor)
                        cost = len(path) - 1  # Number of moves
                        time_taken = time.time() - start_time
                        return SearchResult(path, cost, self.nodes_expanded, time_taken, "BFS", True)
                    
                    queue.append(neighbor)
        
        # No path found
        time_taken = time.time() - start_time
        return SearchResult([], float('inf'), self.nodes_expanded, time_taken, "BFS", False)


class UniformCostSearch(PathfindingAlgorithm):
    """Uniform-Cost Search algorithm."""
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], time_step: int = 0) -> SearchResult:
        start_time = time.time()
        self.nodes_expanded = 0
        
        if start == goal:
            return SearchResult([start], 0, 0, 0, "UCS", True)
        
        # Priority queue: (cost, position)
        pq: List[Tuple[int, Tuple[int, int]]] = [(0, start)]
        visited = set()
        came_from = {}
        cost_so_far = {start: 0}
        
        while pq:
            current_cost, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            self.nodes_expanded += 1
            
            if current == goal:
                path = self.reconstruct_path(came_from, current)
                time_taken = time.time() - start_time
                return SearchResult(path, current_cost, self.nodes_expanded, time_taken, "UCS", True)
            
            # Get neighbors at current time step
            neighbors = self.environment.get_neighbors(current[0], current[1], time_step)
            
            for neighbor in neighbors:
                terrain_cost = self.environment.get_terrain_cost(neighbor[0], neighbor[1])
                new_cost = current_cost + terrain_cost
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(pq, (int(new_cost), neighbor))
        
        # No path found
        time_taken = time.time() - start_time
        return SearchResult([], float('inf'), self.nodes_expanded, time_taken, "UCS", False)


class AStarSearch(PathfindingAlgorithm):
    """A* Search algorithm with Manhattan distance heuristic."""
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], time_step: int = 0) -> SearchResult:
        start_time = time.time()
        self.nodes_expanded = 0
        
        if start == goal:
            return SearchResult([start], 0, 0, 0, "A*", True)
        
        # Priority queue: (f_score, position)
        pq = [(self.manhattan_distance(start, goal), start)]
        visited = set()
        came_from = {}
        g_score = {start: 0}
        
        while pq:
            _, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            self.nodes_expanded += 1
            
            if current == goal:
                path = self.reconstruct_path(came_from, current)
                cost = g_score[current]
                time_taken = time.time() - start_time
                return SearchResult(path, cost, self.nodes_expanded, time_taken, "A*", True)
            
            # Get neighbors at current time step
            neighbors = self.environment.get_neighbors(current[0], current[1], time_step)
            
            for neighbor in neighbors:
                terrain_cost = self.environment.get_terrain_cost(neighbor[0], neighbor[1])
                tentative_g_score = g_score[current] + terrain_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.manhattan_distance(neighbor, goal)
                    heapq.heappush(pq, (f_score, neighbor))
        
        # No path found
        time_taken = time.time() - start_time
        return SearchResult([], float('inf'), self.nodes_expanded, time_taken, "A*", False)


class HillClimbingSearch(PathfindingAlgorithm):
    """Hill-climbing local search with random restarts."""
    
    def __init__(self, environment: GridEnvironment, max_restarts: int = 10, max_steps: int = 100):
        super().__init__(environment)
        self.max_restarts = max_restarts
        self.max_steps = max_steps
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], time_step: int = 0) -> SearchResult:
        start_time = time.time()
        self.nodes_expanded = 0
        
        best_path = []
        best_cost = float('inf')
        
        for restart in range(self.max_restarts):
            path, cost = self._hill_climb_attempt(start, goal, time_step)
            self.nodes_expanded += len(path) if path else self.max_steps
            
            if cost < best_cost:
                best_cost = cost
                best_path = path
                
                # If we found the goal, stop searching
                if path and path[-1] == goal:
                    break
        
        time_taken = time.time() - start_time
        found_solution = bool(best_path) and len(best_path) > 0 and best_path[-1] == goal
        
        return SearchResult(best_path, best_cost, self.nodes_expanded, time_taken, "Hill-Climbing", found_solution)
    
    def _hill_climb_attempt(self, start: Tuple[int, int], goal: Tuple[int, int], time_step: int) -> Tuple[List[Tuple[int, int]], float]:
        """Single hill-climbing attempt."""
        current = start
        path = [current]
        total_cost = 0
        
        for step in range(self.max_steps):
            if current == goal:
                return path, total_cost
            
            # Get all valid neighbors
            neighbors = self.environment.get_neighbors(current[0], current[1], time_step + step)
            
            if not neighbors:
                break
            
            # Find neighbor that minimizes distance to goal
            best_neighbor = None
            best_distance = float('inf')
            
            for neighbor in neighbors:
                distance = self.manhattan_distance(neighbor, goal)
                if distance < best_distance:
                    best_distance = distance
                    best_neighbor = neighbor
            
            # If no improvement, try random neighbor (to escape local optima)
            if best_neighbor is None or self.manhattan_distance(best_neighbor, goal) >= self.manhattan_distance(current, goal):
                best_neighbor = random.choice(neighbors) if neighbors else None
            
            if best_neighbor is None:
                break
                
            # Move to best neighbor
            terrain_cost = self.environment.get_terrain_cost(best_neighbor[0], best_neighbor[1])
            total_cost += terrain_cost
            current = best_neighbor
            path.append(current)
            
            # Avoid cycles by preventing revisiting recent positions
            if len(path) > 4 and current in path[-4:-1]:
                break
        
        return path, total_cost


class DynamicReplanner:
    """Handles dynamic replanning when obstacles appear or environment changes."""
    
    def __init__(self, primary_algorithm: PathfindingAlgorithm, fallback_algorithm: PathfindingAlgorithm, environment: GridEnvironment):
        self.primary_algorithm = primary_algorithm
        self.fallback_algorithm = fallback_algorithm
        self.environment = environment
        self.replan_count = 0
    
    def execute_plan_with_replanning(self, start: Tuple[int, int], goal: Tuple[int, int], 
                                   max_execution_steps: int = 1000) -> Dict:
        """Execute plan with dynamic replanning when obstacles are encountered."""
        current_pos = start
        full_path = [current_pos]
        total_cost = 0
        time_step = 0
        replan_events = []
        
        while current_pos != goal and time_step < max_execution_steps:
            # Plan from current position to goal
            result = self.primary_algorithm.find_path(current_pos, goal, time_step)
            
            if not result.found_solution:
                # Try fallback algorithm
                result = self.fallback_algorithm.find_path(current_pos, goal, time_step)
                
                if not result.found_solution:
                    break
            
            planned_path = result.path[1:]  # Exclude current position
            
            # Execute planned path step by step
            for i, next_pos in enumerate(planned_path):
                time_step += 1
                
                # Check if path is still valid (no new obstacles)
                if self.environment.is_obstacle(next_pos[0], next_pos[1], time_step):
                    # Obstacle detected, need to replan
                    replan_events.append({
                        'time_step': time_step,
                        'position': current_pos,
                        'blocked_position': next_pos,
                        'reason': 'dynamic_obstacle'
                    })
                    self.replan_count += 1
                    break
                
                # Move to next position
                terrain_cost = self.environment.get_terrain_cost(next_pos[0], next_pos[1])
                total_cost += terrain_cost
                current_pos = next_pos
                full_path.append(current_pos)
                self.environment.update_time_step()
                
                if current_pos == goal:
                    break
        
        return {
            'path': full_path,
            'cost': total_cost,
            'time_steps': time_step,
            'replan_events': replan_events,
            'replan_count': self.replan_count,
            'reached_goal': current_pos == goal
        }