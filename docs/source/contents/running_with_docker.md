# Running with Docker

## Use the pre-built image
You can easily run NWB widgets Panel using Docker.

```bash
$ docker pull ghcr.io/NeurodataWithoutBorders/nwbwidgets-panel:latest
$ docker run -p 8866:8866 nwbwidgets-panel
```

By default, the container will run to access remote files only, using `enable_local_source=False` (see [Panel](https://nwb-widgets.readthedocs.io/en/latest/contents/quickstart.html#basic-usage-with-panel)). To allow for local files browsering, run the container with:
```bash
$ docker run -p 8866:8866 -e ENABLE_LOCAL_SOURCE=True -v "$(pwd):/app/local_files:ro" nwbwidgets-panel
```
where `$(pwd)` is the current working directly. Change it if needed to the path containing the NWB files you want to visualize.

If you want to skip the default Panel and instead run it just for a specific remote NWB file, pass the s3 url as an ENV variable to the container:
```bash
$ docker run -p 8866:8866 -e S3_URL_NWBFILE=<s3_url_to_nwb_file> nwbwidgets-panel
```


## Build from source
Start by cloning the repository:
```bash
$ git clone https://github.com/NeurodataWithoutBorders/nwbwidgets.git
$ cd nwbwidgets/docker
```

Build the Docker image:
```bash
$ DOCKER_BUILDKIT=1 docker build -t nwbwidgets-panel .
```
