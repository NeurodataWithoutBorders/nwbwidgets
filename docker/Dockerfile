FROM python:3.10

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install git+https://github.com/NeurodataWithoutBorders/nwbwidgets.git

WORKDIR /app
RUN mkdir local_files
RUN mkdir nwb-cache

COPY favicon.ico .
COPY NWB_Panel.ipynb .
COPY run.sh .

EXPOSE 8866
CMD ["./run.sh"]
# CMD ["voila", "notebook.ipynb", "--port=8866", "--Voila.ip=0.0.0.0", "--enable_nbextensions=True", "--autoreload=True", "--no-browser"]

# EXPOSE 8888
# CMD ["jupyter", "notebook", "notebook.ipynb", "--port=8888", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
