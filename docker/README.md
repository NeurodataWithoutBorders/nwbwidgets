## Build Docker Image

From `/docker` directory, run:
```bash
$ DOCKER_BUILDKIT=1 docker build -t nwbwidgets-panel .
```

## Run Docker Image

To run with `Panel`:
```bash
$ docker run -p 8866:8866 nwbwidgets-panel
```

By default, the container will run to access remote files only, using `enable_local_source=False` (see [Panel](https://nwb-widgets.readthedocs.io/en/latest/contents/running_on_jupyter.html#using-panel)). To allow for local files browsering, run the container with:
```bash
$ docker run -p 8866:8866 -e ENABLE_LOCAL_SOURCE=True -v "$(pwd):/app/local_files:ro" nwbwidgets-panel
```

To run for a specific remote NWB file, pass the s3 url as an ENV variable to the container:
```bash
$ docker run -p 8866:8866 -e S3_URL_NWBFILE=<s3_url_to_nwb_file> nwbwidgets-panel
```
