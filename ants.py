import random

import numpy as np

from test_data_parser import CVRPTestParser

SEED = 42
NUM_ANTS = 50
NUMBER_OF_ITERATIONS = 300
ALPHA = 1
BETA = 7
EVAPORATE_FACTOR = 0.1
Q = 10


class AcoSolver:
    def __init__(
            self,
            number_of_trucks,
            max_capacity,
            demand,
            distances,
            depot,
            optimal
    ):
        self.number_of_trucks = number_of_trucks
        self.max_capacity = max_capacity
        self.demand = demand
        self.distances = distances
        self.depot = depot
        self.optimal = optimal

        self.num_nodes = len(self.demand)

        self.num_ants = self.num_nodes if self.num_nodes > NUM_ANTS else NUM_ANTS
        self.num_iterations = NUMBER_OF_ITERATIONS
        self.alpha = ALPHA
        self.beta = BETA
        self.evaporate = EVAPORATE_FACTOR
        self.Q = Q
        self.seed = SEED

        self.pheromones = self._init_pheromones_by_distance()
        self.best_solution = None
        self.best_cost = float('inf')

    def _init_pheromones(self):
        return np.ones((self.num_nodes, self.num_nodes))

    def _init_pheromones_by_distance(self):
        avg_distance = np.mean(self.distances)
        return avg_distance / (self.distances + 1e-6)

    def _calculate_probability(self, current_node, unvisited):
        probabilities = []
        for node in unvisited:
            probabilities.append(
                self.pheromones[current_node][node] ** self.alpha *
                (1 / (self.distances[current_node][node] + 1e-6)) ** self.beta
            )

        probabilities = np.array(probabilities)
        total = probabilities.sum()
        if total == 0 or np.isnan(total):
            return np.ones(len(unvisited)) / len(unvisited)

        probabilities /= total
        return probabilities

    def _construct_ant_solution(self):
        num_nodes = len(self.demand)
        unvisited = set(range(num_nodes)) - {self.depot}
        routes = []

        while unvisited:
            route = [self.depot]
            load = 0
            current_node = self.depot

            while unvisited:
                probabilities = self._calculate_probability(
                    current_node=current_node,
                    unvisited=list(unvisited)
                )
                if len(probabilities) == 0:
                    break

                next_node = random.choices(list(unvisited), probabilities, k=1)[0]

                if load + self.demand[next_node] > self.max_capacity:
                    break

                route.append(next_node)
                load += self.demand[next_node]
                unvisited.remove(next_node)
                current_node = next_node

            route.append(self.depot)
            routes.append(route)

        return routes

    def _calculate_cost(self, solution):
        total_cost = 0
        for routes in solution:
            for route in routes:
                cost = 0
                for i in range(len(route) - 1):
                    cost += self.distances[route[i]][route[i + 1]]
                total_cost += cost
        return total_cost

    def _update_pheromones(self, solutions, costs):
        self.pheromones *= (1 - self.evaporate)

        for routes, cost in zip(solutions, costs):
            for route in routes:
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    self.pheromones[from_node][to_node] += self.Q / (cost + 1e-6)
                    self.pheromones[to_node][from_node] \
                        = self.pheromones[from_node][to_node]

    def solve(self):
        for _ in range(self.num_iterations):
            solutions = []

            for _ in range(self.num_ants):
                solution = self._construct_ant_solution()
                solutions.append(solution)

            costs = [self._calculate_cost([solution]) for solution in solutions]

            # top_k = max(1, len(solutions) // 2)
            top_k = 10
            best_solutions_i = np.argsort(costs)[:top_k]
            new_cost = []
            new_sol = []
            for i in best_solutions_i:
                new_cost.append(costs[i])
                new_sol.append(solutions[i])

            self._update_pheromones(new_sol, new_cost)

            min_cost = min(costs)
            if min_cost < self.best_cost:
                self.best_cost = min_cost
                self.best_solution = solutions[costs.index(min_cost)]

        return self.best_solution, self.best_cost

    def get_result(self):
        diff = round((self.best_cost - self.optimal) / self.optimal, 2)
        return self.best_cost, self.optimal, diff


def run_new_test(bench_dir, test):
    test_set = CVRPTestParser.parse(bench_dir, test)
    solver = AcoSolver(
        number_of_trucks=test_set.truck_count,
        max_capacity=test_set.capacity,
        demand=test_set.demand,
        distances=test_set.distances,
        depot=test_set.depot,
        optimal=test_set.optimal
    )
    solver.solve()
    route_path, optimal, diff = solver.get_result()
    print(f"{test=} {route_path=} {optimal=} {diff=}")
    return bench_dir, diff
