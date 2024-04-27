# Lots of assumptions for how my specific model works.

import cProfile
import glob
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
import obspy
import scipy
import torch

CLASSES = ["earthquake", "explosion", "surface event"]

batch_size = 64
window_len_s = 60
sampling_rate = 100  # Hz
window_len = window_len_s * sampling_rate
step = 10 * sampling_rate
threshold = 0.1
min_sample = 2 * sampling_rate
max_sample = 15 * sampling_rate


cpu_model = torch.load("model1.pt")
cpu_model.eval()


def write_pick(trace_stats, source_type, pick_time, pick_prob, picks_file):
    network = trace_stats.network
    station = trace_stats.station
    location = trace_stats.location
    channel = trace_stats.channel
    picks_file.write(
        f"{network},{station},{location},{channel},{source_type},{pick_time},{pick_prob}\n"
    )


def normalize(waveform):
    normalized = scipy.signal.detrend(waveform, axis=-1)
    return normalized / np.std(normalized, axis=-1)[:, None]


def apply_batch(worker_n, model, X, batch_starttime, trace_stats, picks_file):
    code = ".".join(
        [trace_stats[k] for k in ("network", "station", "location", "channel")]
    )
    batch_endtime = batch_starttime + window_len_s * X.shape[0]
    print(f"worker {worker_n} {code} {batch_starttime} {batch_endtime}")
    sys.stdout.flush()
    with torch.no_grad():
        X = torch.tensor(X[:, None, :], dtype=torch.float32)
        y = model(X).cpu().numpy()
        for batchi in range(y.shape[0]):
            for classi, cls in enumerate(CLASSES):
                # Assumes a single peak.
                peaki = np.argmax(y[batchi, classi])
                if (
                    min_sample < peaki < max_sample
                    and y[batchi, classi, peaki] > threshold
                ):
                    pick_time = (
                        batch_starttime + window_len_s * batchi + peaki / sampling_rate
                    )
                    write_pick(
                        trace_stats,
                        cls,
                        pick_time,
                        y[batchi, classi, peaki],
                        picks_file,
                    )


def apply_trace(worker_n, model, tr, picks_file):
    print(f"worker {worker_n} working on trace")
    sys.stdout.flush()
    # Trim data to multiple of window_len. Doing this before resampling
    # makes Fourier transform much faster. See benchmark_resample.py.
    tr.data = tr.data[: window_len * (len(tr.data) // window_len)]
    if tr.stats.sampling_rate != sampling_rate:
        print(f"worker {worker_n} resampling trace")
        sys.stdout.flush()
        tr.resample(sampling_rate)
        assert tr.stats.sampling_rate == sampling_rate
        print(f"worker {worker_n} done resampling trace")
        sys.stdout.flush()
    XX = tr.data
    for batch_start in range(0, len(XX), batch_size * window_len):
        X_batch = XX[batch_start : batch_start + (batch_size + 1) * window_len]
        for start in range(0, window_len, step):
            X_shift = X_batch[start : -(window_len - start)].reshape(-1, window_len)
            # Normalize after each shift.
            X_norm = normalize(X_shift)
            apply_batch(
                worker_n,
                model,
                X_norm,
                tr.stats.starttime + (batch_start + start) / sampling_rate,
                tr.stats,
                picks_file,
            )


# Need to define this for multiprocessing.
def apply_mseed(worker_n, model, args):
    mseed_path, picks_path = args
    print(f"worker {worker_n} reading {mseed_path}")
    sys.stdout.flush()
    st = obspy.read(mseed_path)
    print(f"worker {worker_n} done reading {mseed_path}")
    sys.stdout.flush()
    for tr in st:
        with open(picks_path, "a") as picks_file:
            apply_trace(worker_n, model, tr, picks_file)


def apply_gpu(n, queue):
    pr = cProfile.Profile()
    pr.enable()
    with torch.device(f"cuda:{n}"):
        model = cpu_model.to(torch.device(f"cuda:{n}"))
        while True:
            print(f"worker {n} waiting...")
            sys.stdout.flush()
            args = queue.get()
            print(f"worker {n} got args {args}")
            sys.stdout.flush()
            if args is None:
                print(f"worker {n} breaking!")
                sys.stdout.flush()
                break  # Received stop value, stop worker.
            apply_mseed(n, model, args)
    pr.disable()
    prof_path = f"worker_{n}.pstats"
    print(f"dumping profile to {prof_path}...")
    sys.stdout.flush()
    pr.dump_stats(prof_path)


def run_inference(data_dir, picks_path):
    mseed_paths = glob.glob(str(Path(data_dir) / "*.mseed"))
    n_gpus = 4
    q = multiprocessing.Queue()
    processes = []
    for n in range(n_gpus):
        p = multiprocessing.Process(target=apply_gpu, args=(n, q))
        p.start()
        processes.append(p)
    for path in mseed_paths:
        q.put((path, picks_path))
    # Add stop value for each process to queue.
    for _ in processes:
        q.put(None)
    # Wait for all processes to finish.
    for p in processes:
        p.join()


def write_header(picks_path):
    with open(picks_path, "w") as f:
        f.write("network,station,location,channel,source_type,pick_time,pick_prob\n")


if __name__ == "__main__":
    data_dir, picks_path = sys.argv[1], sys.argv[2]
    assert not os.path.exists(picks_path), f"{picks_path} already exists."
    write_header(picks_path)
    run_inference(data_dir, picks_path)
