from obspy import UTCDateTime

startyear = 2000
endyear = 2024

download_dir = "/share/barcheck/data/seismic/pnw_surface_events/mt_st_helens_2000_2024"

# Mt. St. Helens crater
lat = 46.203880
lon = -122.190498

radius_deg = 0.25
channel_priorities = ("EHZ", "HHZ", "BHZ")

# How many times to try to download missing traces.
n_attempts = 5
