from pathlib import Path

import fsspec
import h5py
import ipywidgets as widgets
from dandi.dandiapi import DandiAPIClient
from fsspec.implementations.cached import CachingFileSystem
from pynwb import NWBHDF5IO
from tqdm.notebook import tqdm

from nwbwidgets import nwb2widget
from nwbwidgets.utils.dandi import (
    get_dandiset_metadata,
    get_file_url,
    has_nwb,
    list_dandiset_files,
)


class Panel(widgets.VBox):
    def __init__(
        self,
        stream_mode: str = "fsspec",
        cache_path: str = None,
        enable_dandi_source: bool = True,
        enable_s3_source: bool = True,
        enable_local_source: bool = True,
        **kwargs,
    ):
        """
        NWB widgets Panel for visualization of NWB files.

        Args:
            stream_mode (str, optional): Either "fsspec" or "ros3". Defaults to "fsspec".
            cache_path (str, optional): The path to cached data if streaming with "fsspec". If left as None, a directory "nwb-cache" is created under the current working directory. Defaults to None.
            enable_dandi_source (bool, optional): Enable DANDI source option. Defaults to True.
            enable_s3_source (bool, optional): Enable S3 source option. Defaults to True.
            enable_local_source (bool, optional): Enable local source option. Defaults to True.
        """
        super().__init__(children=[], **kwargs)

        self.stream_mode = stream_mode

        self.cache_path = cache_path
        if cache_path is None:
            self.cache_path = "nwb-cache"

        # Create a virtual filesystem based on the http protocol and use caching to save accessed data to RAM.
        if enable_dandi_source or enable_s3_source:
            self.cfs = CachingFileSystem(
                fs=fsspec.filesystem("http"),
                cache_storage=self.cache_path,  # Local folder for the cache
            )

        self.source_options_names = list()
        if enable_local_source:
            self.source_options_names.append("Local dir")
            self.source_options_names.append("Local file")
        if enable_dandi_source:
            self.source_options_names.append("DANDI")
        if enable_s3_source:
            self.source_options_names.append("S3")

        self.all_dandisets_metadata = None

        self.source_options_radio = widgets.RadioButtons(
            options=self.source_options_names,
            layout=widgets.Layout(width="100px", overflow=None),
        )
        self.source_options_label = widgets.Label("Source:", layout=widgets.Layout(width="100px", overflow=None))
        self.source_options = widgets.VBox(
            children=[self.source_options_radio],
            layout=widgets.Layout(width="120px", overflow=None, padding="5px 5px 5px 10px"),
        )

        self.source_changing_panel = widgets.Box()

        self.input_form = widgets.HBox(
            children=[
                self.source_options,
                self.source_changing_panel,
                # self.source_file_dandi_vbox
            ],
            layout={"border": "1px solid gray"},
        )
        self.widgets_panel = widgets.VBox([])

        self.children = [self.input_form, self.widgets_panel]
        self.source_options_radio.observe(self.updated_source, "value")

        if enable_local_source:
            self.source_options_radio.value = "Local dir"
            self.create_components_local_dir_source()
        elif enable_dandi_source:
            self.source_options_radio.value = "DANDI"
            self.create_components_dandi_source()

    def updated_source(self, args=None):
        """Update Panel components depending on chosen source."""
        if args["new"] == "DANDI":
            self.create_components_dandi_source()
        elif args["new"] == "S3":
            self.create_components_s3_source()
        elif args["new"] == "Local dir":
            self.create_components_local_dir_source()
        elif args["new"] == "Local file":
            self.create_components_local_file_source()

    def create_components_dandi_source(self, args=None):
        """Create widgets components for DANDI option"""
        if self.all_dandisets_metadata is None:
            self.all_dandisets_metadata = self.get_all_dandisets_metadata()

        dandiset_options = list()
        for m in self.all_dandisets_metadata:
            item_name = m.id.split(":")[1].split("/")[0] + " - " + m.name
            dandiset_options.append(item_name)

        self.source_dandi_id = widgets.Dropdown(
            options=dandiset_options,
            description="Dandiset:",
            layout=widgets.Layout(width="400px", overflow=None),
        )
        self.source_dandi_file_dropdown = widgets.Dropdown(
            options=[],
            description="File:",
            layout=widgets.Layout(width="400px", overflow=None),
        )
        self.source_dandi_file_button = widgets.Button(icon="check", description="Load file")

        self.source_dandi_vbox = widgets.VBox(
            children=[
                self.source_dandi_id,
                self.source_dandi_file_dropdown,
                self.source_dandi_file_button,
            ],
            layout=widgets.Layout(padding="5px 0px 5px 0px"),
        )

        self.dandi_summary = widgets.HTML(
            value="<style>p{word-wrap: break-word}</style> <p>" + "" + "</p>",
            layout=widgets.Layout(height="100px", width="700px", padding="5px 5px 5px 10px"),
        )

        self.dandi_panel = widgets.HBox([self.source_dandi_vbox, self.dandi_summary])

        self.source_changing_panel.children = [self.dandi_panel]

        self.source_dandi_id.observe(self.list_dandiset_files_dropdown, "value")
        self.source_dandi_file_button.on_click(self.stream_dandiset_file)
        self.list_dandiset_files_dropdown()

    def create_components_s3_source(self):
        """Create widgets components for S3 option"""
        self.source_s3_file_url = widgets.Text(
            value="",
            description="URL:",
        )
        self.source_s3_button = widgets.Button(icon="check", description="Load file")
        self.s3_panel = widgets.VBox(
            children=[self.source_s3_file_url, self.source_s3_button],
            layout=widgets.Layout(padding="5px 0px 5px 0px"),
        )
        self.source_changing_panel.children = [self.s3_panel]
        self.source_s3_button.on_click(self.stream_s3_file)

    def create_components_local_dir_source(self):
        """Create widgets components for Loca dir option"""
        self.local_dir_path = widgets.Text(
            value="",
            description="Dir path:",
            layout=widgets.Layout(width="400px", overflow=None),
        )
        self.local_dir_button = widgets.Button(description="Search")
        self.local_dir_top = widgets.HBox(
            children=[self.local_dir_path, self.local_dir_button],
            layout=widgets.Layout(padding="5px 0px 5px 0px"),
        )
        self.local_dir_files = widgets.Dropdown(
            options=[],
            description="Files:",
            layout=widgets.Layout(width="400px", overflow=None),
        )
        self.local_dir_file_button = widgets.Button(icon="check", description="Load file")
        self.local_dir_panel = widgets.VBox(
            children=[
                self.local_dir_top,
                self.local_dir_files,
                self.local_dir_file_button,
            ],
            layout=widgets.Layout(padding="5px 0px 5px 0px"),
        )
        self.source_changing_panel.children = [self.local_dir_panel]
        self.local_dir_button.on_click(self.list_local_dir_files)
        self.local_dir_file_button.on_click(self.load_local_dir_file)

    def create_components_local_file_source(self):
        """Create widgets components for Local file option"""
        self.local_file_path = widgets.Text(
            value="",
            description="File path:",
            layout=widgets.Layout(width="400px", overflow=None),
        )
        self.local_file_button = widgets.Button(icon="check", description="Load file")
        self.local_file_panel = widgets.VBox(
            children=[self.local_file_path, self.local_file_button],
            layout=widgets.Layout(padding="5px 0px 5px 0px"),
        )
        self.source_changing_panel.children = [self.local_file_panel]
        self.local_file_button.on_click(self.load_local_file)

    def list_dandiset_files_dropdown(self, args=None):
        """Populate dropdown with all files and text area with summary"""
        self.dandi_summary.value = "Loading dandiset info..."
        self.source_dandi_file_dropdown.options = []
        dandiset_id = self.source_dandi_id.value.split("-")[0].strip()
        self.source_dandi_file_dropdown.options = list_dandiset_files(dandiset_id=dandiset_id)

        metadata = get_dandiset_metadata(dandiset_id=dandiset_id)
        self.dandi_summary.value = "<style>p{word-wrap: break-word}</style> <p>" + metadata.description + "</p>"

    def list_local_dir_files(self, args=None):
        """List NWB files in local dir"""
        if Path(self.local_dir_path.value).is_dir():
            all_files = [f.name for f in Path(self.local_dir_path.value).glob("*.nwb")]
            self.local_dir_files.options = all_files
        else:
            print("Invalid local dir path")

    def stream_dandiset_file(self, args=None):
        """Stream NWB file from DANDI"""
        self.widgets_panel.children = [widgets.Label("loading...")]
        dandiset_id = self.source_dandi_id.value.split("-")[0].strip()
        file_path = self.source_dandi_file_dropdown.value
        s3_url = get_file_url(dandiset_id=dandiset_id, file_path=file_path)
        if self.stream_mode == "ros3":
            io = NWBHDF5IO(s3_url, mode="r", load_namespaces=True, driver="ros3")

        elif self.stream_mode == "fsspec":
            f = self.cfs.open(s3_url, "rb")
            file = h5py.File(f)
            io = NWBHDF5IO(file=file, load_namespaces=True)

        nwbfile = io.read()
        self.widgets_panel.children = [nwb2widget(nwbfile)]

    def stream_s3_file(self, args=None):
        """Stream NWB file from S3 url"""
        self.widgets_panel.children = [widgets.Label("loading...")]
        s3_url = self.source_s3_file_url.value
        if self.stream_mode == "ros3":
            io = NWBHDF5IO(s3_url, mode="r", load_namespaces=True, driver="ros3")
        elif self.stream_mode == "fsspec":
            f = self.cfs.open(s3_url, "rb")
            file = h5py.File(f)
            io = NWBHDF5IO(file=file, load_namespaces=True)

        nwbfile = io.read()
        self.widgets_panel.children = [nwb2widget(nwbfile)]

    def load_local_dir_file(self, args=None):
        """Load local NWB file"""
        full_file_path = str(Path(self.local_dir_path.value) / self.local_dir_files.value)
        io = NWBHDF5IO(full_file_path, mode="r", load_namespaces=True)
        nwb = io.read()
        self.widgets_panel.children = [nwb2widget(nwb)]

    def load_local_file(self, args=None):
        """Load local NWB file"""
        full_file_path = str(Path(self.local_file_path.value))
        io = NWBHDF5IO(full_file_path, mode="r", load_namespaces=True)
        nwb = io.read()
        self.widgets_panel.children = [nwb2widget(nwb)]

    def get_all_dandisets_metadata(self):
        with DandiAPIClient() as client:
            all_metadata = list()
            dandisets_iter = tqdm(list(client.get_dandisets()), desc="Loading dandiset metadata")
            self.source_changing_panel.children = [dandisets_iter.container]
            for ii, dandiset in enumerate(dandisets_iter):
                if 1 < ii < 560:
                    try:
                        metadata = dandiset.get_metadata()
                        if has_nwb(metadata):
                            all_metadata.append(metadata)
                    except:
                        pass
                else:
                    pass
        return all_metadata
