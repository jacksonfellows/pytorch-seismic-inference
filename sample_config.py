from obspy import UTCDateTime

starttime = UTCDateTime(2023, 1, 1)
endtime = UTCDateTime(2023, 1, 2)
download_dir = "./test"
lat = 46.203880
lon = -122.190498
radius_deg = 0.4
channel_priorities = ("HHZ", "BHZ")
