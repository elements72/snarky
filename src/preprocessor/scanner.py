import multiprocessing
from typing import Callable
import os

class Scanner:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._dirs = self.get_dirs()
        self._results = []

    def get_dirs(self) -> list:
        return sorted(os.listdir(self._dataset_path))

    def get_dirs_queue(self, num_workers: int) -> multiprocessing.Queue:
        manager = multiprocessing.Manager()
        queue = manager.Queue()

        count = 1
        for directory in self.get_dirs():
            queue.put((directory, count))
            count = count + 1

        for _ in range(num_workers):
            queue.put(None)

        return queue

    def scan(self, analysis_function: Callable[..., any]) -> any:
        queue = self.get_dirs_queue(1)
        print(
            f"""Starting database scan of {self._dataset_path}""")

        results = worker(analysis_function, queue)

        return results

    def scan_multi(self, analysis_function: Callable[..., any]) -> any:
        num_workers = min(len(self._dirs), multiprocessing.cpu_count() - 1)
        pool = multiprocessing.Pool(processes=num_workers)
        queue = self.get_dirs_queue(num_workers)

        print(
            f"""Starting database scan of {self._dataset_path}, started {num_workers} processes""")

        workers = [pool.apply_async(worker, (analysis_function, queue))
                   for _ in range(num_workers)]
        results = []
        for w in workers:
            results.extend(w.get())

        pool.close()
        pool.join()

        print("Database scan finished")

        return results


def worker(analysis_function: Callable[..., any], queue: multiprocessing.Queue) -> any:
    results = []
    for (file, count) in iter(queue.get, None):
        result = analysis_function(file, results, count)
        results = result
    return results
