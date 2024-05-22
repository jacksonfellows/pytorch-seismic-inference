from obspy import UTCDateTime

starttime = UTCDateTime(2023, 5, 1)
endtime = UTCDateTime(2023, 6, 1)

download_dir = "/Users/jackson/pytorch-seismic-inference/test_04"

# Mt. St. Helens crater
lat = 46.203880
lon = -122.190498

radius_deg = 0.25
channel_priorities = ("HHZ", "BHZ")

# How many times to try to download missing traces.
n_attempts = 5
