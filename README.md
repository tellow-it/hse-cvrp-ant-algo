# CVRP Ant Algorithm

## Best params
- SEED = 42
- NUM_ANTS = max(50, num_cities)
- NUMBER_OF_ITERATIONS = 300
- ALPHA = 1
- BETA = 7
- EVAPORATE_FACTOR = 0.1
- Q = 10

## Features
- init pheromones by weighted average reverse distances
- update pheromones only for top 10 best solutions
- num ants = max(50, num_cities)

## Benchmark result

- bench_dir='A' Mean diff 0.087
- bench_dir='M' Mean diff 0.140
- bench_dir='P' Mean diff 0.066

### Total results: 0.097

