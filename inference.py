# Lots of assumptions for how my specific model works.

import glob
import sqlite3
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


model = torch.load("model1.pt")
model.eval()


def write_pick(trace_stats, source_type, pick_time, pick_prob):
    network = trace_stats.network
    station = trace_stats.station
    location = trace_stats.location
    channel = trace_stats.channel
    insert_query = """
    INSERT INTO picks (network, station, location, channel, source_type, pick_time, pick_prob) VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    assert type(pick_time) == obspy.UTCDateTime
    pick_time = str(pick_time)  # Convert to str for insertion.
    pick_prob = float(pick_prob)  # sqlite3 doesn't like numpy floats.
    row = (network, station, location, channel, source_type, pick_time, pick_prob)
    with sqlite3.connect("picks.db") as cur:
        cur.execute(insert_query, row)


def normalize(waveform):
    normalized = scipy.signal.detrend(waveform, axis=-1)
    return normalized / np.std(normalized, axis=-1)[:, None]


def apply_batch(X, batch_starttime, trace_stats):
    print(batch_starttime)
    with torch.no_grad():
        X = torch.tensor(X[:, None, :], dtype=torch.float32)
        y = model(X).numpy()
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
                    write_pick(trace_stats, cls, pick_time, y[batchi, classi, peaki])


def apply_trace(tr):
    if tr.stats.sampling_rate != sampling_rate:
        tr = tr.resample(sampling_rate)
    XX = tr.data
    # Trim off extra samples.
    XX = XX[: window_len * (len(XX) // window_len)]
    for batch_start in range(0, len(XX), batch_size * window_len):
        X_batch = XX[batch_start : batch_start + (batch_size + 1) * window_len]
        for start in range(0, window_len, step):
            X_shift = X_batch[start : -(window_len - start)].reshape(-1, window_len)
            # Normalize after each shift.
            X_norm = normalize(X_shift)
            apply_batch(
                X_norm,
                tr.stats.starttime + (batch_start + start) / sampling_rate,
                tr.stats,
            )


def run_inference(data_dir):
    mseed_paths = glob.glob(str(Path(data_dir) / "*.mseed"))
    for mseed_path in mseed_paths:
        st = obspy.read(mseed_path)
        for tr in st:
            apply_trace(tr)


if __name__ == "__main__":
    run_inference(sys.argv[1])
