import threading
import queue
import time
from collections import defaultdict

class TimeAligner:
    def __init__(self, sample_rates, ready_count, callback):
        self.sample_rates = sample_rates
        self.queues = [queue.Queue() for _ in sample_rates]
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ready_count = ready_count
        self.callback = callback
        self.aligned_data = defaultdict(list)

    def start(self):
        self.base = self._compute_miu_time(time.time() * 1e6, self.sample_rates[0])
        self.threads = [threading.Thread(target=self._process_data, args=(i,)) for i in range(len(self.sample_rates))]
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()

    def add_data(self, source_idx, timestamp, data):
        self.queues[source_idx].put((timestamp, data))

    def _process_data(self, source_idx):
        while not self.stop_event.is_set():
            try:
                timestamp, data = self.queues[source_idx].get(timeout=1)
                miu_time = self._compute_miu_time(timestamp, self.sample_rates[source_idx], self.base)

                # Process data, e.g., check if data points are MIU aligned, and store or perform calculations
                with self.lock:
                    self._handle_data(source_idx, miu_time, data)

            except queue.Empty:
                continue

    def _compute_miu_time(self, data_time, sample_rate, base=0):
        sample_time = 1e6 / sample_rate
        miu = data_time // sample_time - base
        print(f"current: {data_time}, miu: {miu}")
        return miu

    def _handle_data(self, source_idx, miu_time, data):
        # Store data if MIU aligned
        self.aligned_data[miu_time].append((source_idx, data))

        # Check if the data is ready for processing (X samples have been aligned)
        if len(self.aligned_data[miu_time]) == self.ready_count:
            self.callback(miu_time, self.aligned_data[miu_time])
            del self.aligned_data[miu_time]


