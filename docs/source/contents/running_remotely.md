# Running NWB widgets remotely

## Using Docker
You can easily deploy a webapp for visualizing NWB files using NWB widgets, [Voila](https://voila.readthedocs.io/en/stable/using.html) and Docker. Start by cloning the repository:
```bash
$ git clone https://github.com/NeurodataWithoutBorders/nwbwidgets.git
$ cd nwbwidgets/docker
```

Build a Docker image:
```bash
$ docker build -t nwbwidgets-voila .
```

Finally, run a container:
```bash
$ docker run -p 8866:8866 nwbwidgets-voila
```

By default, the Panel will run with `enable_local_source=False` (see [Panel](https://nwb-widgets.readthedocs.io/en/latest/contents/quickstart.html#basic-usage-with-panel)). To allow for local files browsering, run the container with:
```bash
$ docker run -p 8866:8866 -e ENABLE_LOCAL_SOURCE=True nwbwidgets-voila
```

If you want to skip the default Panel and instead run it for a specific NWB file, pass the s3 url as an ENV variable to the container:
```bash
$ docker run -p 8866:8866 -e S3_URL_NWBFILE=<s3_url_to_nwb_file> nwbwidgets-voila
```


## Using Binder
