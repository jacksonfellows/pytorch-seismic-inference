import calendar
import datetime
import glob
import re
from collections import Counter

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import obspy
import pandas as pd
from matplotlib import colormaps
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

# picks = pd.read_csv("mt_st_helens_2000_2024_picks.csv", parse_dates=["pick_time"])


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


def find_events(clss, threshold, networks=None):
    if networks is not None:
        picks_ = picks[picks["network"].isin(networks)]
    else:
        picks_ = picks
    picks_ = picks_[(picks_["source_type"] == clss) & (picks_["pick_prob"] > threshold)]
    print(f"n={len(picks_)}")
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

    df = pd.DataFrame(index=event_times, data={"nstations": event_num_stations})
    # df = df.tz_convert("US/Pacific")
    return df


def save_events():
    networks = ["CC", "UW"]
    for cls in ["earthquake", "explosion", "surface event"]:
        print(f"{cls=}")
        events_df = find_events(cls, threshold=0.5, networks=networks)
        cls_filename = cls.replace(" ", "_")
        # WARNING: Assumes dates are in UTC
        iso = "%Y-%m-%dT%H:%M:%SZ"
        events_df.to_csv(
            f"mt_st_helens_events/{cls_filename}_events.csv", date_format=iso
        )


def load_event_dfs():
    local_time = "US/Pacific"
    df_eq, df_ex, df_su = [
        pd.read_csv(x, parse_dates=[0], index_col=0).tz_convert(local_time)
        for x in [
            "mt_st_helens_events/earthquake_events.csv",
            "mt_st_helens_events/explosion_events.csv",
            "mt_st_helens_events/surface_event_events.csv",
        ]
    ]
    return dict(eq=df_eq, ex=df_ex, su=df_su)


def plot2(dfs):
    fig, axs = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        sharey=False,
        layout="constrained",
        figsize=(8.5, 5),
    )
    nminstats = 3
    plt.xlim(pd.Timestamp("2000"), pd.Timestamp("2024"))
    plt.suptitle(f"Mt. St. Helens 2000–2024, detections at ≥{nminstats} stations")
    for i, (cls, clsl) in enumerate(
        zip(["Earthquakes", "Explosions", "Surface events"], ["eq", "ex", "su"])
    ):
        axs[i].set_title(cls)
        axs[i].set_ylabel("# of events/day")
        axs[i].set_xticks([pd.Timestamp(str(year)) for year in range(2000, 2025, 2)])
        axs[i].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        df = dfs[clsl]
        day_counts = (
            df[(df["nstations"] >= nminstats) & (df.index.year >= 2000)]
            .resample("1D")
            .count()
        )
        axs[i].plot(
            day_counts.index, day_counts.to_numpy()[:, 0], color="k", linewidth=0.5
        )
    # plt.show()
    plt.savefig("figures/mt_st_helens_2000_2024.pdf")
    plt.close()


def plot_seasonal(dfs):
    nminstats = 3
    fig, axs = plt.subplots(
        layout="constrained",
        figsize=(6, 8.5),
        nrows=3,
        ncols=1,
        sharex=True,
        sharey=False,
    )
    plt.suptitle(
        f"Mt. St. Helens 2000–2004 & 2008–2024, detections at ≥{nminstats} stations"
    )
    x = [
        datetime.datetime.strptime(f"2000-{day:03d}", "%Y-%j")
        for day in range(1, 365 + 1)
    ]
    for i, (clsl, cls) in enumerate(
        zip(["Earthquakes", "Explosions", "Surface events"], ["eq", "ex", "su"])
    ):
        df = dfs[cls]
        day_counts = df[df["nstations"] >= nminstats].resample("1D").count()
        axs[i].set_title(clsl)
        axs[i].set_ylabel("Normalized # of events/day")
        axs[i].set_xlim(x[0], x[-1])
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        cm = matplotlib.colormaps["plasma"]
        years = list(range(2000, 2004)) + list(range(2008, 2024))
        mat = np.zeros((len(years), 365))
        for j, year in enumerate(years):
            y = day_counts[day_counts.index.year == year]
            yy = y.to_numpy()[:, 0].astype("float")
            L = min(len(yy), 365)
            mat[j, :L] = yy[:L]
        mat -= np.mean(mat, axis=1).reshape((-1, 1))
        mat /= np.std(mat, axis=1).reshape((-1, 1))
        for j in range(len(years)):
            axs[i].plot(x, mat[j], color="k", linewidth=0.2)
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)
        axs[i].fill_between(
            x, mean - std, mean + std, color="r", alpha=0.2, edgecolor="none"
        )
    axs[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    # axs[-1].set_xlabel("Day of year")
    # plt.show()
    plt.savefig("figures/mt_st_helens_seasonal.pdf")
    plt.close()


def plot_su_day(dfs):
    nminstats = 3
    fig = plt.figure(layout="constrained", figsize=(13, 8.5))
    subfigs = fig.subfigures(nrows=1, ncols=3)
    labels = ["Earthquakes", "Explosions", "Surface events"]
    keys = ["eq", "ex", "su"]
    plt.suptitle(
        f"Mt. St. Helens 2000–2004 & 2008–2024, detections at ≥{nminstats} stations"
    )
    for subfigi, subfig in enumerate(subfigs):
        axs = subfig.subplots(nrows=4, ncols=3, sharex=True, sharey=True)
        axs[0, 0].set_xlim(1, 24)
        for i in range(4):
            axs[i, 0].set_ylabel("Normalized # of events")
        for i in range(3):
            axs[-1, i].set_xlabel("Hour of day")
            axs[-1, i].set_xticks([6, 12, 18, 24])
        df = dfs[keys[subfigi]]
        # TODO only non-volcanic years?
        subfig.suptitle(labels[subfigi])
        for month in range(1, 13):
            row, col = divmod(month - 1, 3)
            axs[row, col].set_title(calendar.month_name[month])
            binned = df[
                (df["nstations"] >= nminstats)
                & (df.index.year >= 2000)
                & (df.index.month == month)
            ].resample("1Y")
            bin2size = 60
            L = 24 * 60 // bin2size
            mat = np.zeros((len(binned), L))
            for i, (_, bin_) in enumerate(binned):
                bin2id = (bin_.index.hour * 60 + bin_.index.minute) // bin2size
                counts = bin_.groupby(bin2id).count()
                for j in range(len(counts)):
                    mat[i, counts.index[j]] = int(counts.iloc[j].iloc[0])
            mat -= np.mean(mat, axis=1).reshape((-1, 1))
            mat /= np.std(mat, axis=1).reshape((-1, 1))
            x = bin2size * np.arange(L) / 60 + 1
            for i in range(len(binned)):
                axs[row, col].plot(x, mat[i], color="k", linewidth=0.2)
            mean = np.mean(mat, axis=0)
            std = np.std(mat, axis=0)
            axs[row, col].fill_between(
                x, mean - std, mean + std, color="r", alpha=0.2, edgecolor="none"
            )
    plt.savefig("figures/mt_st_helens_diurnal.pdf")
    plt.close()


# def find_station_availability(files_list, start_time, end_time, networks="*"):
#     with open(files_list, "r") as f:
#         mseed_files = f.readlines()
#     # Create regular expression to match network codes.
#     if networks == "*":
#         network_regexp = "[A-Z]+"
#     else:
#         network_regexp = "(" + "|".join(networks) + ")"
#     path_regexp = f"^{network_regexp}\."
#     mseed_files = [path for path in mseed_files if re.match(path_regexp, path)]
#     print("n =", len(mseed_files))
#     date_counts = Counter()
#     for path in mseed_files:
#         if path.endswith(".mseed\n"):
#             date_str = path.split("__")[1]
#             date_counts[date_str] += 1
#     # Convert map of date -> # of stations to arrays I can easily plot.
#     days = []
#     num_stations = []
#     t = start_time
#     while t < end_time:
#         s = t.strftime("%Y%m%d") + "T000000Z"
#         days.append(t)
#         num_stations.append(date_counts[s])
#         t += datetime.timedelta(days=1)
#     return days, num_stations


# def plot_station_availability(files_list, start_time, end_time, networks="*"):
#     days, num_stations = find_station_availability(
#         files_list, start_time, end_time, networks
#     )
#     plt.title(f"Networks = {networks}")
#     plt.xlabel("Day")
#     plt.ylabel("# of stations")
#     plt.bar(days, num_stations, width=1, color="grey")
#     # TODO: Show 2,3,4,... more clearly.
#     plt.show()
