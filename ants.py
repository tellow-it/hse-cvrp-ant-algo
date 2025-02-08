import sys

import numpy as np
import vrplib
from dataclasses import dataclass
from random import Random
import numpy
import os


MAX_RANGE = 1000
NUMBER_OF_ITERATIONS = 300
ALPHA = 1
BETA = 7
EVAPORATE_FACTOR = 0.1
PHEROMONES_FACTOR = 20

MAX_SIZE = sys.maxsize


@dataclass
class TestData:
    truck_count: int
    capacity: int
    demand: list
    distances: list[list]
    optimal: int


class CVRPTestParser:
    @classmethod
    def parse(cls, bench_dir: str, filename: str) -> TestData:
        instance = vrplib.read_instance(os.path.join(bench_dir, filename))

        truck_count = int(instance["name"].split("-")[-1][1:])
        capacity = int(instance["capacity"])
        distances = instance["edge_weight"]
        demand = instance["demand"]

        sol_file = filename.replace(".vrp", ".sol")
        solution = vrplib.read_solution(os.path.join(bench_dir, sol_file))
        optimal = solution["cost"]

        test_data = TestData(int(truck_count), int(capacity), demand, distances, int(optimal))
        return test_data


class _AntSolution:
    def __init__(self) -> None:
        self.current_route = [0]
        self.current_route_length = -1
        self.best_route = []
        self.best_route_length = MAX_SIZE

    def check_current_route(self) -> bool:
        if self.current_route_length == -1 or self.current_route_length >= self.best_route_length:
            return False

        self.best_route = self.current_route
        self.best_route_length = self.current_route_length
        return True

    def reset(self) -> None:
        self.current_route = [0]
        self.current_route_length = -1


class ACOSolver:
    def __init__(self,
                 number_of_trucks: int,
                 max_capacity: int,
                 demand: list,
                 distances: list[list],
                 optimal: int,
                 max_range: int,
                 number_of_ants: int,
                 alpha: float,
                 beta: float,
                 pheromones_factor: float,
                 evaporate_factor: float,
                 number_of_iterations: int,
                 seed: int = None,
                 ) -> None:
        self.distances = distances
        self.demand = demand
        self.max_capacity = max_capacity
        self.max_range = max_range
        self.number_of_trucks = number_of_trucks
        self.was_visited = [-self.number_of_trucks + 1] + [0] * (len(self.demand) - 1)
        self.rem_capacity = self.max_capacity
        self.rem_range = self.max_range
        self.current_id = 0
        self.waiting = self.demand[1:]
        self.route = []
        self.route_length = MAX_SIZE
        self.result = None
        self.seed = seed
        self.random = Random(self.seed)
        self.optimal = optimal
        self.number_of_ants = number_of_ants
        self.alpha = alpha
        self.beta = beta
        self.pheromones_factor = pheromones_factor
        self.evaporate_factor = evaporate_factor
        self.number_of_iterations = number_of_iterations
        self.pheromones = self._init_pheromones_by_distance_matrix()
        self.current_ant_id = 0
        self.ants = [_AntSolution() for _ in range(number_of_ants)]
        self.optimal = optimal
        self.diff = None

    def _init_pheromones(self):
        return numpy.ones((len(self.demand), len(self.demand)))

    def _init_pheromones_by_distance_matrix(self):
        avg_distance = np.mean(self.distances)
        return (avg_distance / (self.distances + 1e-6)) * 10

    def _can_visit(self, target_id: int) -> bool:
        range_fulfilled = (
                (target_id == 0 and self._get_distance_to(target_id) <= self.rem_range) or
                self._get_distance_to(target_id) + self.distances[target_id][0] <= self.rem_range
        )
        return (self.current_id != target_id
                and self.was_visited[target_id] < 1
                and self.demand[target_id] <= self.rem_capacity
                and range_fulfilled)

    def _visit(self, target_id: int, route: list[int]) -> float:
        self.was_visited[target_id] += 1
        self.rem_capacity -= self.demand[target_id]
        self.rem_range -= self._get_distance_to(target_id)
        distance = self._get_distance_to(target_id)
        self.current_id = target_id
        route.append(self.current_id)
        if target_id == 0:
            self.rem_capacity = self.max_capacity
            self.rem_range = self.max_range
        return distance

    def _check_all_visited(self) -> bool:
        return all(self.was_visited[1:])

    def _get_distance_to(self, target_id: int) -> float:
        return self.distances[self.current_id][target_id]

    def _find_route(self) -> tuple[list[int], float]:
        route_length = 0
        route = [0]
        while not self._check_all_visited():
            to_visit = list(filter(self._can_visit, range(0, len(self.demand))))
            if len(to_visit) == 0:
                return [], -1
            if 0 in to_visit and len(to_visit) > 1:
                to_visit.remove(0)
            target_id = self._get_target_id(to_visit)
            route_length += self._visit(target_id, route)
        route_length = (route_length + self._visit(0, route) if self._can_visit(0) else -1)
        return route, route_length

    def _update_result(self, route: list[int], route_length: float) -> None:
        if route_length < self.route_length:
            self.route = route
            self.route_length = route_length
            self.result = True

    def _get_route_length(self, route: list[int]) -> float:
        length = 0
        for i, city_from in enumerate(route[:-1]):
            city_to = route[i + 1]
            length += self.distances[city_from][city_to]
        return length

    def _get_target_id(self, allowed_cities: list[int]) -> int:
        weights = []
        for city in allowed_cities:
            if self._get_distance_to(city) == 0:
                return city

            pheromon_factor = self.pheromones[self.current_id, city]
            heuristic_factor = 1 / self._get_distance_to(city)
            weights.append((pheromon_factor ** self.alpha) * (heuristic_factor ** self.beta))
        if sum(weights) <= 0.0:
            weights = None
        return self.random.choices(allowed_cities, weights, k=1)[0]

    def _lay_pheromones(self, route: list[int], factor: float = None) -> None:
        if factor is None:
            factor = self.pheromones_factor
        if len(route) <= 0:
            return
        for i in range(len(route) - 1):
            city_from = route[i]
            city_to = route[i + 1]
            if city_from != city_to:
                self.pheromones[city_from, city_to] += factor / self._get_route_length(route)

    def _update_pheromones(self) -> None:
        for i in range(len(self.demand)):
            for j in range(len(self.demand)):
                if i != j:
                    self.pheromones[i, j] *= (1 - self.evaporate_factor)

    def solve(self) -> None:
        for i in range(self.number_of_iterations):
            for ant in self.ants:
                ant.reset()
                self.was_visited = [-self.number_of_trucks + 1] + [0] * (len(self.demand) - 1)
                ant.current_route, ant.current_route_length = self._find_route()
                self._lay_pheromones(ant.current_route)
                if ant.check_current_route():
                    self._update_result(ant.best_route, ant.best_route_length)
            self._update_pheromones()

    def get_result(self):
        self.diff = round(abs(self.route_length - self.optimal) / self.optimal, 2)
        return self.route_length, self.optimal, self.diff





def run_test(bench_dir, test):
    test_set = CVRPTestParser.parse(bench_dir, test)
    solver = ACOSolver(
        number_of_trucks=test_set.truck_count,
        max_capacity=test_set.capacity,
        demand=test_set.demand,
        distances=test_set.distances,
        optimal=test_set.optimal,
        max_range=MAX_RANGE,
        number_of_ants=len(test_set.demand),
        alpha=ALPHA,
        beta=BETA,
        pheromones_factor=PHEROMONES_FACTOR,
        evaporate_factor=EVAPORATE_FACTOR,
        number_of_iterations=NUMBER_OF_ITERATIONS,
        seed=None,
    )
    solver.solve()
    route_path, optimal, diff = solver.get_result()
    print(f"{test=} {route_path=} {optimal=} {diff=}")
    return bench_dir, diff
