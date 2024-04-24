import numbers
import sys

import obspy
import obspy.clients.fdsn.mass_downloader as mass_downloader


def download_area(
    download_dir, starttime, endtime, lat, lon, radius_deg, channel_priorities
):
    domain = mass_downloader.CircularDomain(
        latitude=lat, longitude=lon, minradius=0, maxradius=radius_deg
    )
    restrictions = mass_downloader.Restrictions(
        starttime=starttime,
        endtime=endtime,
        channel_priorities=channel_priorities,
        minimum_interstation_distance_in_m=0,  # Default is 1000 m!
        chunklength_in_sec=60 * 60 * 24,  # 1 day.
    )

    mdl = mass_downloader.MassDownloader(providers=["IRIS"])
    mdl.download(
        domain=domain,
        restrictions=restrictions,
        mseed_storage=download_dir,
        stationxml_storage=download_dir,
        threads_per_client=8,
        download_chunk_size_in_mb=32,
    )


def parse_config_file(path):
    config_globals = dict()
    with open(path, "r") as f:
        exec(f.read(), config_globals)

    config = dict()
    config_keys = (
        "download_dir",
        "starttime",
        "endtime",
        "lat",
        "lon",
        "radius_deg",
        "channel_priorities",
    )
    for k in config_keys:
        assert k in config_globals, f"Parameter {k} missing!"
        config[k] = config_globals[k]

    # Validate config.
    assert (
        type(config["starttime"])
        == type(config["endtime"])
        == obspy.core.utcdatetime.UTCDateTime
    )
    for num in (config["lat"], config["lon"], config["radius_deg"]):
        assert isinstance(num, numbers.Number)

    return config


if __name__ == "__main__":
    config = parse_config_file(sys.argv[1])
    print(f"config = {config}")
    download_area(**config)
