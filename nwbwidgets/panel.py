import concurrent.futures
import os
import warnings
from pathlib import Path

import fsspec
import h5py
import ipywidgets as widgets
from dandi.dandiapi import DandiAPIClient
from fsspec.implementations.cached import CachingFileSystem
from ipyfilechooser import FileChooser
from pynwb import NWBHDF5IO
from tqdm.notebook import tqdm

from .utils.dandi import (
    get_dandiset_metadata,
    get_file_url,
    has_nwb,
    list_dandiset_files,
)
from .view import nwb2widget


class Panel(widgets.VBox):
    def __init__(
        self,
        stream_mode: str = "fsspec",
        enable_dandi_source: bool = True,
        enable_s3_source: bool = True,
        enable_local_source: bool = True,
        enable_cache: bool = True,
        show_warnings: bool = False,
        **kwargs,
    ):
        """
        NWB widgets Panel for visualization of NWB files.

        Args:
            stream_mode : {'fsspec', 'ros3'}
            cache_path : str, optional
                The path to cached data if streaming with "fsspec". If left as None, a directory "nwb-cache" is
                created under the current working directory. Defaults to None.
            enable_dandi_source : bool, default: True
                Enable DANDI source option.
            enable_s3_source : bool, default: True
                Enable S3 source option.
            enable_local_source : bool, default: True
                Enable local source option.
            enable_cache : bool, default: True
                Enable caching data.
            show_warnings : bool, default: False
                Show warnings.
        """
        super().__init__(children=[], **kwargs)

        self.stream_mode = stream_mode
        self.io = None
        self.nwbfile = None

        if not show_warnings:
            warnings.filterwarnings("ignore")

        self.source_options_names = list()
        if enable_local_source:
            self.source_options_names.append("Local file")
        if enable_dandi_source:
            self.source_options_names.append("DANDI")
        if enable_s3_source:
            self.source_options_names.append("S3")

        self.enable_cache = enable_cache
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
            self.source_options_radio.value = "Local file"
            self.create_components_local_file_source()
        elif enable_dandi_source:
            self.source_options_radio.value = "DANDI"
            self.create_components_dandi_source()

    def updated_source(self, args=None):
        """Update Panel components depending on chosen source."""
        if args["new"] == "DANDI":
            self.create_components_dandi_source()
        elif args["new"] == "S3":
            self.create_components_s3_source()
        elif args["new"] == "Local file":
            self.create_components_local_file_source()

    def create_cache_row(self):
        """Create cache row"""
        if self.enable_cache:
            self.cache_checkbox = widgets.Checkbox(description="cache")
            self.cache_checkbox.observe(self.toggle_cache)
            self.cache_path_text = widgets.Text("nwb-cache")
            self.cache_path_text.layout.visibility = "hidden"
            self.cache_row = widgets.HBox([self.cache_checkbox, self.cache_path_text])
        else:
            self.cache_checkbox = None
            self.cache_path_text = None
            self.cache_row = None

    def create_components_dandi_source(self, args=None):
        """Create widgets components for DANDI option"""
        if self.all_dandisets_metadata is None:
            self.all_dandisets_metadata = self.get_all_dandisets_metadata()

        dandiset_options = list()
        for m in self.all_dandisets_metadata:
            item_name = m.id.split(":")[1].split("/")[0] + " - " + m.name
            dandiset_options.append(item_name)

        self.source_dandi_id = widgets.Dropdown(
            options=sorted(dandiset_options),
            description="Dandiset:",
            layout=widgets.Layout(width="400px", overflow=None),
        )
        self.source_dandi_file_dropdown = widgets.Dropdown(
            options=[],
            description="File:",
            layout=widgets.Layout(width="400px", overflow=None),
        )

        self.create_cache_row()
        self.source_dandi_file_button = widgets.Button(icon="check", description="Load file")

        children_list = [self.source_dandi_id, self.source_dandi_file_dropdown, self.source_dandi_file_button]
        if self.cache_row is not None:
            children_list.insert(2, self.cache_row)

        self.source_dandi_vbox = widgets.VBox(
            children=children_list,
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

    def toggle_cache(self, args):
        if isinstance(args["new"], dict):
            if args["new"].get("value") is True:
                self.cache_path_text.layout.visibility = "visible"
            elif args["new"].get("value") is False:
                self.cache_path_text.layout.visibility = "hidden"

    def create_components_s3_source(self):
        """Create widgets components for S3 option"""
        self.source_s3_file_url = widgets.Text(
            value="",
            description="URL:",
        )
        self.source_s3_button = widgets.Button(icon="check", description="Load file")

        self.create_cache_row()
        children_list = [
            self.source_s3_file_url,
            self.source_s3_button,
        ]
        if self.cache_row is not None:
            children_list.insert(1, self.cache_row)

        self.s3_panel = widgets.VBox(
            children=children_list,
            layout=widgets.Layout(padding="5px 0px 5px 0px"),
        )
        self.source_changing_panel.children = [self.s3_panel]
        self.source_s3_button.on_click(self.stream_s3_file)

    def create_components_local_file_source(self):
        """Create widgets components for Local file option"""
        self.local_file_chooser = FileChooser()
        self.local_file_chooser.sandbox_path = str(Path.cwd())
        self.local_file_chooser.filter_pattern = ["*.nwb"]
        self.local_file_chooser.title = "<b>Select local NWB file</b>"
        self.local_file_chooser.register_callback(self.load_local_file)
        self.local_file_panel = widgets.VBox(
            children=[self.local_file_chooser],
            layout=widgets.Layout(padding="5px 0px 5px 0px"),
        )
        self.source_changing_panel.children = [self.local_file_panel]

    def list_dandiset_files_dropdown(self, args=None):
        """Populate dropdown with all files and text area with summary"""
        self.dandi_summary.value = "Loading dandiset info..."
        self.source_dandi_file_dropdown.options = []
        dandiset_id = self.source_dandi_id.value.split("-")[0].strip()
        self.source_dandi_file_dropdown.options = list_dandiset_files(dandiset_id=dandiset_id)
        metadata = get_dandiset_metadata(dandiset_id=dandiset_id)
        self.dandi_summary.value = "<style>p{word-wrap: break-word}</style> <p>" + metadata.description + "</p>"

    def stream_dandiset_file(self, args=None):
        """Stream NWB file from DANDI"""
        self.widgets_panel.children = [widgets.Label("loading...")]
        dandiset_id = self.source_dandi_id.value.split("-")[0].strip()
        file_path = self.source_dandi_file_dropdown.value
        try:
            s3_url = get_file_url(dandiset_id=dandiset_id, file_path=file_path)
            self._stream_s3_file(s3_url)
        except Exception as e:
            self.widgets_panel.children = [widgets.Label(str(e))]

    def stream_s3_file(self, args=None):
        """Stream NWB file from S3 url"""
        self.widgets_panel.children = [widgets.Label("loading file...")]
        s3_url = self.source_s3_file_url.value
        try:
            self._stream_s3_file(s3_url)
        except Exception as e:
            self.widgets_panel.children = [widgets.Label(str(e))]

    def _stream_s3_file(self, s3_url):
        if self.stream_mode == "ros3":
            io_kwargs = {
                "path": s3_url,
                "mode": "r",
                "load_namespaces": True,
                "driver": "ros3",
            }
        elif self.stream_mode == "fsspec":
            fs = fsspec.filesystem("http")
            if not self.cache_checkbox:
                f = fs.open(s3_url, "rb")
            else:
                cfs = CachingFileSystem(
                    fs=fs,
                    cache_storage=self.cache_path_text.value,  # Local folder for the cache
                )
                f = cfs.open(s3_url, "rb")
            file = h5py.File(f)
            io_kwargs = {
                "file": file,
                "mode": "r",
                "load_namespaces": True,
            }
        # Close previous io
        if self.io:
            self.io.close()
        self.io = NWBHDF5IO(**io_kwargs)
        self.nwbfile = self.io.read()
        self.widgets_panel.children = [nwb2widget(self.nwbfile)]

    def load_local_file(self, args=None):
        """Load local NWB file"""
        full_file_path = str(Path(self.local_file_chooser.selected))
        if self.io:
            self.io.close()
        self.io = NWBHDF5IO(full_file_path, mode="r", load_namespaces=True)
        self.nwbfile = self.io.read()
        self.widgets_panel.children = [nwb2widget(self.nwbfile)]

    def process_dandiset(self, dandiset):
        try:
            metadata = dandiset.get_metadata()
            if has_nwb(metadata):
                return metadata
        except:
            pass
        return None

    def get_all_dandisets_metadata(self):
        with DandiAPIClient() as client:
            api_key = os.environ.get("DANDI_API_KEY", None)
            if api_key:
                client.dandi_authenticate()
            all_metadata = []
            dandisets = list(client.get_dandisets())
            total_dandisets = len(dandisets)
            pbar = tqdm(total=total_dandisets, desc="Loading dandiset metadata")
            self.source_changing_panel.children = [pbar.container]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_dandiset, dandiset) for dandiset in dandisets]

                for future in concurrent.futures.as_completed(futures):
                    metadata = future.result()
                    if metadata:
                        all_metadata.append(metadata)
                    pbar.update(1)
                pbar.close()
        return all_metadata
