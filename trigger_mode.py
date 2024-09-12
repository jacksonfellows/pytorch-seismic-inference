import matplotlib.pyplot as plt


def load():
    sizes_kb = []
    years = []
    with open("cc_uw_mseed_sizes", "r") as f:
        for line in f:
            size, path = line.split()
            year = path[-22:-18]
            sizes_kb.append(int(size))
            years.append(year)
    return years, sizes_kb
