## Build documentation

To run auto-doc for the API, from `/docs` run:
```bash
$ sphinx-apidoc -f -o ./source/api_reference ../nwbwidgets
```

To build the HTML, from `/docs` run:
```bash
$ sphinx-build -b html ./source ./build
```