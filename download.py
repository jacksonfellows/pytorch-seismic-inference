import numbers
import sys

import obspy
import obspy.clients.fdsn.mass_downloader as mass_downloader


def download_area(
    download_dir, starttime, endtime, lat, lon, radius_deg, channel_priorities, **kwargs
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
        # Try to get as much data as possible:
        reject_channels_with_gaps=False,
        minimum_length=0.0,  # Min. length as fraction of time frame.
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
    config_vars = (
        ("download_dir", str),
        ("starttime", obspy.core.utcdatetime.UTCDateTime),
        ("endtime", obspy.core.utcdatetime.UTCDateTime),
        ("lat", numbers.Number),
        ("lon", numbers.Number),
        ("radius_deg", numbers.Number),
        ("channel_priorities", object),  # Any type.
        ("n_attempts", int),
    )
    # Validate config.
    for k, type_ in config_vars:
        assert k in config_globals, f"Parameter {k} missing!"
        assert isinstance(config_globals[k], type_)
        config[k] = config_globals[k]

    return config


if __name__ == "__main__":
    config = parse_config_file(sys.argv[1])
    print(f"config = {config}")
    for n in range(config["n_attempts"]):
        print(f"Attempt {n}:")
        try:
            download_area(**config)
        except Exception as e:
            print(f"Got exception {e}")
            pass
