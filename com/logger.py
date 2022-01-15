import os
import csv
import numpy as np


class CSVLogger(object):
    def __init__(self, log_dir, filename="progress.csv"):
        self.csvfile = open(os.path.join(log_dir, filename), "w")
        self.writer = None

    def init_writer(self, keys):
        if self.writer is None:
            self.writer = csv.DictWriter(self.csvfile, fieldnames=list(keys))
            self.writer.writeheader()

    def log_epoch(self, data):
        if "stats" in data:
            for key, values in data["stats"].items():
                data["mean_" + key] = np.mean(values)
                data["median_" + key] = np.median(values)
                data["min_" + key] = np.min(values)
                data["max_" + key] = np.max(values)
        del data["stats"]

        self.init_writer(data.keys())
        self.writer.writerow(data)
        self.csvfile.flush()

    def __del__(self):
        self.csvfile.close()


class ConsoleCSVLogger(CSVLogger):
    def __init__(self, console_log_interval=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console_log_interval = console_log_interval

    def log_epoch(self, data):
        super().log_epoch(data)

        flush = data["iter"] % self.console_log_interval == 0
        print(
            (
                f'Updates {data["iter"]}, '
                f'timesteps {data["total_num_steps"]}, '
                f'curriculum {data.get("curriculum", -1)}, '
                f'metric {data.get("curriculum_metric", -1):.1f}, '
                f'FPS {data["fps"]}, '
                f'mean_rew {data["mean_rew"]:.1f}, '
                f'min_rew {data["min_rew"]:.1f}, '
                f'max_rew {data["max_rew"]:.1f}'
            ),
            flush=flush,
        )
