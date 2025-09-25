# delivery-agent
Autonomous Delivery Agent – CSA2001

An autonomous delivery agent that navigates a 2D grid city to deliver packages efficiently while handling static obstacles, varying terrain costs, and dynamic moving obstacles.

 Key Features

Environment: Static/dynamic obstacles, terrain with different movement costs.

Algorithms:

Uninformed: BFS, Uniform-Cost

Informed: A* with admissible heuristics

Local Search: Hill-Climbing / Simulated Annealing for replanning

Agent Design: Rational agent that replans under constraints (time, fuel).

Experiments: Compare algorithms on multiple maps (cost, nodes expanded, runtime).

Deliverables

Python source code with CLI

4 test maps (small, medium, large, dynamic)

Proof of dynamic replanning
