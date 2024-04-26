import sqlite3

import matplotlib
import matplotlib.patches as mpatches
import obspy
from matplotlib import pyplot as plt

# TODO: This approach won't work for lots of data.
st = obspy.read("test/*.mseed")


def load_picks():
    with sqlite3.connect("picks.db") as cur:
        res = cur.execute("SELECT * FROM picks;")
        return res.fetchall()


picks = load_picks()


def plot_pick(pick_row):
    network, station, location, channel, source_type, pick_time, pick_prob = pick_row
    tr = st.select(network=network, station=station)[0]
    pick_time = obspy.UTCDateTime(pick_time)
    X = tr.slice(pick_time - 30, pick_time + 30).data
    plt.plot(X, "k", linewidth=0.5)
    plt.vlines(30 * tr.stats.sampling_rate, X.min(), X.max(), "r")
    plt.show()


class Colorer:
    def __init__(self):
        self.color_map = dict()
        self.n = 0

    def __call__(self, x):
        if x in self.color_map:
            return self.color_map[x]
        # Use golden ratio
        c = matplotlib.colors.hsv_to_rgb(((1.618 * self.n) % 1, 0.7, 1))
        self.n += 1
        self.color_map[x] = c
        return c

    def get_legend_handles(self):
        return [mpatches.Patch(color=c, label=l) for l, c in self.color_map.items()]


def plot_times(clss):
    C = Colorer()
    su = [row for row in picks if row[4] == clss]
    print(len(su))
    start = obspy.UTCDateTime("2023-01-01")
    pick_times = [obspy.UTCDateTime(pick_row[5]) - start for pick_row in su]
    pick_probs = [pick_row[6] for pick_row in su]
    station_colors = [C(".".join((pick_row[0], pick_row[1]))) for pick_row in su]
    plt.legend(handles=C.get_legend_handles())
    plt.scatter(pick_times, pick_probs, c=station_colors)
    plt.show()
