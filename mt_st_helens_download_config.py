from obspy import UTCDateTime

starttime = UTCDateTime(2000, 1, 1)
endtime = UTCDateTime(2024, 1, 1)

download_dir = "~/mt_st_helens"

# Mt. St. Helens crater
lat = 46.203880
lon = -122.190498

radius_deg = 0.25
channel_priorities = ("HHZ", "BHZ")