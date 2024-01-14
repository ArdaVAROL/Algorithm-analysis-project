"""
CENG 319
ALGORITHM ANALYSIS
Fall 2023-2024
SEMESTER PROJECT
Project Delivery:
January 11, 2024 at 23:59
"""

# Importing libraries
import heapq

import networkx as nx
import numpy as np


class Project:
    def __init__(self, input):
        self.input = input

    # Function to read the input file
    def read_input(self):
        # Splitting the input file into 4 matrices
        self.input = self.input.split("\n\n")
        self.neighborhood = self.input[0]
        self.bandwidth = self.input[1]
        self.delay = self.input[2]
        self.reliability = self.input[3]

        # Splitting the matrices into rows
        self.neighborhood = self.neighborhood.split("\n")
        self.bandwidth = self.bandwidth.split("\n")
        self.delay = self.delay.split("\n")
        self.reliability = self.reliability.split("\n")

        # Splitting the rows into columns
        for i in range(len(self.neighborhood)):
            self.neighborhood[i] = self.neighborhood[i].split(":")
            self.bandwidth[i] = self.bandwidth[i].split(":")
            self.delay[i] = self.delay[i].split(":")
            self.reliability[i] = self.reliability[i].split(":")

        # Converting the matrices into numpy arrays
        self.neighborhood = np.array(self.neighborhood)
        self.bandwidth = np.array(self.bandwidth)
        self.delay = np.array(self.delay)
        self.reliability = np.array(self.reliability)

        # Converting the matrices into float
        self.neighborhood = self.neighborhood.astype(float)
        self.bandwidth = self.bandwidth.astype(float)
        self.delay = self.delay.astype(float)
        self.reliability = self.reliability.astype(float)

        # Printing the matrices
        print("Neighborhood Matrix:\n", self.neighborhood)
        print("Bandwidth Matrix:\n", self.bandwidth)
        print("Delay Matrix:\n", self.delay)
        print("Reliability Matrix:\n", self.reliability)

        # Returning the matrices
        return self.neighborhood, self.bandwidth, self.delay, self.reliability

        # Modified function to find the shortest path with constraints

    def shortest_path(self, source, destination, min_bandwidth, max_delay, min_reliability):
        # Create a graph with additional properties
        self.graph = nx.Graph()
        for i in range(len(self.neighborhood)):
            for j in range(len(self.neighborhood[i])):
                if self.neighborhood[i][j] != 0:
                    self.graph.add_edge(i, j, weight=self.neighborhood[i][j],
                                        bandwidth=self.bandwidth[i][j],
                                        delay=self.delay[i][j],
                                        reliability=self.reliability[i][j])

        # Initialize distances and priority queue
        distances = {node: float('infinity') for node in self.graph.nodes}
        distances[source] = 0
        priority_queue = [(0, source, 1.0, [])]  # distance, node, reliability, path

        while priority_queue:
            current_distance, current_node, current_reliability, path = heapq.heappop(priority_queue)
            path = path + [current_node]

            if current_node == destination:
                return path, current_distance

            for neighbor, edge_data in self.graph[current_node].items():
                if edge_data['bandwidth'] >= min_bandwidth and edge_data['reliability'] >= min_reliability:
                    new_distance = current_distance + edge_data['weight']
                    new_reliability = current_reliability * edge_data['reliability']

                    if new_distance < distances[neighbor] and new_distance <= max_delay:
                        distances[neighbor] = new_distance
                        heapq.heappush(priority_queue, (new_distance, neighbor, new_reliability, path))

        return None, float('infinity')

    # Modified Dijkstra's algorithm considering the objective function
    def shortest_path_with_objective(self, source, destination, min_bandwidth, max_delay, min_reliability,
                                     bandwidth_demand):
        # Create a graph with additional properties
        self.graph = nx.Graph()
        for i in range(len(self.neighborhood)):
            for j in range(len(self.neighborhood[i])):
                if self.neighborhood[i][j] != 0:
                    self.graph.add_edge(i, j, weight=self.neighborhood[i][j],
                                        bandwidth=self.bandwidth[i][j],
                                        delay=self.delay[i][j],
                                        reliability=self.reliability[i][j])

        # Initialize distances and priority queue
        distances = {node: float('infinity') for node in self.graph.nodes}
        distances[source] = 0
        priority_queue = [(0, source, 1.0, [])]  # distance, node, reliability, path

        while priority_queue:
            current_distance, current_node, current_reliability, path = heapq.heappop(priority_queue)
            path = path + [current_node]

            if current_node == destination:
                return path, current_distance

            for neighbor, edge_data in self.graph[current_node].items():
                if edge_data['bandwidth'] >= min_bandwidth and edge_data['reliability'] >= min_reliability:
                    new_distance = current_distance + (edge_data['weight'] * bandwidth_demand)
                    new_reliability = current_reliability * edge_data['reliability']

                    if new_distance < distances[neighbor] and new_distance <= max_delay:
                        distances[neighbor] = new_distance
                        heapq.heappush(priority_queue, (new_distance, neighbor, new_reliability, path))

        return None, float('infinity')

    # Modified Bellman-Ford algorithm considering the objective function
    def bellman_ford_with_objective(self, source, bandwidth_demand):
        # Initialize distances and predecessors
        distances = {v: float('infinity') for v in self.graph.nodes()}
        distances[source] = 0

        # Relax edges repeatedly
        for _ in range(len(self.graph.nodes()) - 1):
            for u, v, data in self.graph.edges(data=True):
                new_cost = distances[u] + (data['weight'] * bandwidth_demand)
                if new_cost < distances[v]:
                    distances[v] = new_cost

        # Check for negative weight cycles
        for u, v, data in self.graph.edges(data=True):
            if distances[u] + (data['weight'] * bandwidth_demand) < distances[v]:
                raise ValueError("Graph contains a negative weight cycle")

        return distances

    # Heuristic function (example: Euclidean distance)
    def heuristic(self, node, goal):
        # Implement the heuristic logic (Euclidean, Manhattan, etc.)
        # This is a placeholder logic. Replace with actual heuristic based on your graph structure.
        return abs(node - goal)

    # Modified A* algorithm considering the objective function
    def a_star_with_objective(self, start, goal, bandwidth_demand):
        open_set = [(0, start, 0, [])]  # (f-score, node, g-score, path)
        closed_set = set()

        while open_set:
            _, current, g, path = heapq.heappop(open_set)
            path = path + [current]

            if current == goal:
                return path

            closed_set.add(current)

            for neighbor, data in self.graph[current].items():
                if neighbor in closed_set:
                    continue

                g_score = g + (data['weight'] * bandwidth_demand)
                f_score = g_score + self.heuristic(neighbor, goal)

                heapq.heappush(open_set, (f_score, neighbor, g_score, path))

        return None  # Path not found

    def process_adjacency_matrix_only(self):
        # Assuming the input is a string representing the adjacency matrix
        matrix = [row.split(':') for row in self.input.strip().split('\n')]
        self.adjacency_matrix = np.array(matrix, dtype=float)
        return self.adjacency_matrix

    def find_shortest_path_with_adjacency_matrix_only(self, adjacency_matrix, source, destination):
        num_nodes = len(adjacency_matrix)
        distances = [float('inf')] * num_nodes
        distances[source] = 0
        priority_queue = [(0, source)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_node == destination:
                break

            for neighbor, weight in enumerate(adjacency_matrix[current_node]):
                if weight > 0:  # There is an edge
                    distance = current_distance + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(priority_queue, (distance, neighbor))

        # Reconstruct path (optional, remove if not needed)
        path = []
        current = destination
        while current != source:
            path.insert(0, current)
            current = min(range(len(adjacency_matrix)), key=lambda i: (
                float('inf') if adjacency_matrix[i][current] == 0 else distances[i] + adjacency_matrix[i][current]))
        path.insert(0, source)

        return path, distances[destination]


# Main function
def main():
    # Path to the input file - Update this with the actual file path
    input_file_path = 'USNET.txt'

    # Read the contents of the input file
    with open(input_file_path, 'r') as file:
        input_data = file.read()

    # Instantiate the Project class
    project = Project(input_data)

    # Read the input matrices
    neighborhood, bandwidth, delay, reliability = project.read_input()

    # Define source, destination, and constraints
    source_node = 0  # Example source node
    destination_node = 4  # Example destination node
    min_bandwidth = 5  # Example minimum bandwidth
    max_delay = 40  # Example maximum delay
    min_reliability = 0.70  # Example minimum reliability
    bandwidth_demand = 5  # Example bandwidth demand

    # Find the shortest path using different algorithms
    path_dijkstra, distance_dijkstra = project.shortest_path(source_node, destination_node, min_bandwidth, max_delay,
                                                             min_reliability)
    path_bellman_ford = project.bellman_ford_with_objective(source_node, bandwidth_demand)
    path_a_star = project.a_star_with_objective(source_node, destination_node, bandwidth_demand)

    # Display the results
    print("Dijkstra's Algorithm Path:", path_dijkstra, "Distance:", distance_dijkstra)
    print("Bellman-Ford Algorithm Distances:", path_bellman_ford)
    print("A* Algorithm Path:", path_a_star)


def main2():
    # Path to the adjacency matrix file
    adjacency_matrix_file_path = "USNET_AjdMatrix.txt"

    # Read the contents of the adjacency matrix file
    with open(adjacency_matrix_file_path, 'r') as file:
        adjacency_matrix_data = file.read()

    # Instantiate the Project class with the adjacency matrix data
    project = Project(adjacency_matrix_data)

    # Process the adjacency matrix
    adjacency_matrix = project.process_adjacency_matrix_only()

    # Define source and destination nodes for the pathfinding
    source_node = 0  # Example: start from node 0
    destination_node = 4  # Example: destination is node 4

    # Find the shortest path using a suitable algorithm
    path, distance = project.find_shortest_path_with_adjacency_matrix_only(adjacency_matrix, source_node,
                                                                           destination_node)

    # Display the results
    print(f"Shortest path from node {source_node} to node {destination_node}: {path}")
    print(f"Total distance: {distance}")


if __name__ == "__main__":
    main()
    print("\n")
    main2()

