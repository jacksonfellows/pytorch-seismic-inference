import datetime
import glob

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import obspy
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

picks = pd.read_csv("mt_st_helens_test.csv", parse_dates=["pick_time"])


def plot_times(clss, threshold):
    picks_clss = picks[
        (picks["source_type"] == clss) & (picks["pick_prob"] > threshold)
    ]
    # pick_times = picks_clss["pick_time"].dt.tz_convert("US/Pacific").dt.time
    pick_times = picks_clss["pick_time"].dt.tz_convert("US/Pacific")
    pick_probs = picks_clss["pick_prob"]
    pick_times.hist(bins=365)
    # plt.plot(pick_times, pick_probs, "be", alpha=0.01)
    plt.show()


def find_events(clss, threshold):
    picks_ = picks[(picks["source_type"] == clss) & (picks["pick_prob"] > threshold)]
    times = picks_["pick_time"].to_numpy()
    station_codes = (picks_["network"] + "." + picks_["station"]).to_numpy()
    I = np.argsort(times)
    times = times[I]
    station_codes = station_codes[I]
    dt = np.timedelta64(10, "s")
    event_times = []
    event_num_stations = []
    i = 0
    while i < len(times):
        indices = [i]
        j = i + 1
        while j < len(times) and (times[j] - times[i]) < dt:
            if station_codes[i] != station_codes[j]:
                indices.append(j)
            j += 1
        if len(indices) > 1:
            # TODO try to fit the most stations into each event.
            event_times.append(np.min(times[indices]))
            event_num_stations.append(len(set(station_codes[indices])))
            i = j + 1
        else:
            i += 1

    return np.array(event_times), np.array(event_num_stations)


def plot_su(su_times_, su_num_stations, ns, days, num_stations):
    fig, axs = plt.subplots(
        layout="tight", figsize=(19, 8), nrows=len(ns) + 1, sharex=True
    )
    plt.suptitle("Mount St. Helens, 2023")
    for i, n in enumerate(ns):
        su_times = su_times_[su_num_stations >= n]
        axs[i].set_title(f"{n} or more stations (n={len(su_times)})")
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].xaxis.set_major_locator(mdates.MonthLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axs[i].hist(su_times, bins=365)
        axs[i].set_ylabel("# of Surface Events")
    axs[-1].set_xlabel("Day")
    # plt.show()
    axs[-1].set_ylabel("# of Stations")
    axs[-1].bar(days, num_stations, width=1, color="grey")
    # plt.show()
    axs[-1].spines["top"].set_visible(False)
    axs[-1].spines["right"].set_visible(False)
    plt.savefig("su_st_helens_2023.png", dpi=300)


def find_station_availability(files_list, start_time, end_time):
    with open(files_list, "r") as f:
        mseed_files = f.readlines()
    print("n =", len(mseed_files))
    t = start_time
    days = []
    num_stations = []
    # TODO: Sort files first.
    while t < end_time:
        next_t = t + datetime.timedelta(days=1)
        date_str = (
            f"{t.strftime('%Y%m%d')}T000000Z__{next_t.strftime('%Y%m%d')}T000000Z"
        )
        day_files = [path for path in mseed_files if date_str in path]
        days.append(t)
        num_stations.append(len(day_files))
        t = next_t
    return np.array(days), np.array(num_stations)


def plot_station_availability(files_list, start_time, end_time):
    days, num_stations = find_station_availability(files_list, start_time, end_time)
    plt.xlabel("Day")
    plt.ylabel("# of stations")
    plt.bar(days, num_stations, width=1, color="grey")
    # TODO: Show 2,3,4,... more clearly.
    plt.show()
