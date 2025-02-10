import os
from dataclasses import dataclass

import vrplib


@dataclass
class TestData:
    truck_count: int
    capacity: int
    demand: list
    distances: list[list]
    depot: int
    optimal: int


class CVRPTestParser:
    @classmethod
    def parse(cls, bench_dir: str, filename: str) -> TestData:
        instance = vrplib.read_instance(os.path.join(bench_dir, filename))

        truck_count = int(instance["name"].split("-")[-1][1:])
        capacity = int(instance["capacity"])
        distances = instance["edge_weight"]
        demand = instance["demand"]
        depot = int(instance['depot'][0])

        sol_file = filename.replace(".vrp", ".sol")
        solution = vrplib.read_solution(os.path.join(bench_dir, sol_file))
        optimal = int(solution["cost"])

        test_data = TestData(
            truck_count=truck_count,
            capacity=capacity,
            demand=demand,
            distances=distances,
            depot=depot,
            optimal=optimal
        )
        return test_data
