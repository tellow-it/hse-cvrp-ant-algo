import multiprocessing
import os

from ants import run_new_test


def run_new_benchmark():
    bench_dirs = ["A", "M", "P"]
    results = {bench_dir: [] for bench_dir in bench_dirs}

    tasks = []
    for bench_dir in bench_dirs:
        tests = [test for test in os.listdir(bench_dir) if test.endswith(".vrp")]
        tasks.extend((bench_dir, test) for test in tests)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for bench_dir, diff in pool.starmap(run_new_test, tasks):
            results[bench_dir].append(diff)

    total_res = 0
    for bench_dir in bench_dirs:
        tests = [diff for diff in results[bench_dir] if diff]
        total_res += sum(tests) / len(tests)
        print(f"{bench_dir=} Mean diff {sum(tests) / len(tests)}")
    print(f"Total results: {round(total_res / 3, 2)}")
