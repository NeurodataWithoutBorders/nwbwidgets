## Build Docker Image
```bash
$ docker build -t nwbwidgets-voila .
```

## Run Docker Image

To run with `Panel`:
```bash
$ docker run -p 8866:8866 nwbwidgets-voila
```

To run for a specific NWB file, pass the s3 url as an ENV variable to the container:
```bash
$ docker run -p 8866:8866 nwbwidgets-voila -e S3_URL_NWBFILE=<s3_url_to_nwb_file>
```