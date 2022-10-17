FROM python:3.10

# Only needed for now, while there's no pynwb release with fsspec
RUN git clone --recurse-submodules https://github.com/NeurodataWithoutBorders/pynwb \
    && cd pynwb \
    && pip install -r requirements.txt \
    && pip install .

RUN cd ..
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install git+https://github.com/NeurodataWithoutBorders/nwbwidgets.git

COPY notebook.ipynb .
EXPOSE 8866

CMD ["voila", "notebook.ipynb", "--port=8866", "--Voila.ip=0.0.0.0", "--enable_nbextensions=True", "--autoreload=True", "--no-browser"]