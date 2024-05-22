import obspy

tr = obspy.read("test/CC.SEP..BHZ__20230101T000000Z__20230102T000000Z.mseed")[0]


def resample_original(tr):
    # Watch out: resample modifies original trace.
    tr.resample(100)


def resample_trim(tr):
    L = int(60 * 60 * 24 * tr.stats.sampling_rate)
    tr.data = tr.data[:L]
    tr.resample(100)
