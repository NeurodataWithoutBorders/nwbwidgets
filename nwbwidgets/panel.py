import ipywidgets as widgets
from pathlib import Path

from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget
from dandi.dandiapi import DandiAPIClient
import h5py


class Panel(widgets.VBox):

    def __init__(self, children: list = None, stream_mode: str = "fsspec", cache_path: str = None, **kwargs):
        if children is None:
            children = list()
        super().__init__(children, **kwargs)

        self.stream_mode = stream_mode
        if cache_path is None:
            cache_path = "nwb-cache"

        self.source_options_radio = widgets.RadioButtons(options=['dandi', 'local dir', 'local file'], value='dandi')
        self.source_options_label = widgets.Label('Source:')
        self.source_options = widgets.VBox(
            children=[self.source_options_label, self.source_options_radio], 
            layout=widgets.Layout(width='150px', overflow=None)
        )

        self.source_path_text = widgets.Text(value="", layout={'width': '300px'})
        self.source_path_dandi_button = widgets.Button(icon='angle-right', layout={'width': '50px'})
        self.source_path_lower = widgets.HBox(
            children=[self.source_path_text, self.source_path_dandi_button]
        )
        self.source_path_label = widgets.Label('Dandiset number:')
        self.source_path = widgets.VBox(
            children=[self.source_path_label, self.source_path_lower], 
            layout=widgets.Layout(width='400px', overflow=None)
        )

        self.source_dandiset_info = widgets.Label("")
        self.source_file_dandi_dropdown = widgets.Dropdown(options=[])
        self.source_file_dandi_button = widgets.Button(icon='angle-down', layout={'width': '50px'})
        self.source_file_dandi_hbox = widgets.HBox(children=[self.source_file_dandi_dropdown, self.source_file_dandi_button])
        self.source_file_dandi_vbox = widgets.VBox([self.source_dandiset_info, self.source_file_dandi_hbox], layout=widgets.Layout(width='400px', overflow=None))

        self.input_form = widgets.HBox([self.source_options, self.source_path, self.source_file_dandi_vbox], layout={"border": "1px solid gray"})
        self.widgets_panel = widgets.VBox([])
        
        self.children = [self.input_form, self.widgets_panel]

        def updated_source(args):
            if args['new'] == "dandi":
                self.source_path_label.value = "Dandiset number:"
                self.source_path_text.value = ""
                self.source_path_dandi_button.icon='angle-right'
                self.source_file_dandi_vbox.layout.display = None
            elif args['new'] == "local dir":
                self.source_path_label.value = "Path to local dir:"
                self.source_path_text.value = ""
                self.source_path_dandi_button.icon='angle-right'
                self.source_file_dandi_vbox.layout.display = None
            elif args['new'] == "local file":
                self.source_path_label.value = "Path to local file:"
                self.source_path_text.value = ""
                self.source_path_dandi_button.icon='angle-down'
                self.source_file_dandi_vbox.layout.display = 'none'

        self.source_options_radio.observe(updated_source, 'value')

        def list_dandiset_files(args):
            self.source_file_dandi_dropdown.options = []
            if self.source_path_label.value == "Dandiset number:":
                with DandiAPIClient() as client:
                    dandiset = client.get_dandiset(dandiset_id=self.source_path_text.value, version_id="draft")
                    try:
                        # Get dandiset info
                        self.source_dandiset_info.value = dandiset.json_dict()["version"]["name"]
                        # Populate dropdown with all files
                        all_files = [i.dict().get("path") for i in dandiset.get_assets()]
                        self.source_file_dandi_dropdown.options = all_files
                    except:
                        self.source_dandiset_info.value = "Invalid Dandiset number"
            elif self.source_path_label.value == "Path to local dir:":
                if Path(self.source_path_text.value).is_dir():
                    self.source_dandiset_info.value = str(Path(self.source_path_text.value))
                    all_files = [f.name for f in Path(self.source_path_text.value).glob("*.nwb")]
                    self.source_file_dandi_dropdown.options = all_files
                else:
                    self.source_dandiset_info.value = "Invalid local dir path"

        self.source_path_dandi_button.on_click(list_dandiset_files)

        def stream_dandiset_file(args):
            self.widgets_panel.children = [widgets.Label("loading...")]
            if self.source_path_label.value == "Dandiset number:":
                with DandiAPIClient() as client:
                    asset = client.get_dandiset(dandiset_id=self.source_path_text.value, version_id="draft").get_asset_by_path(self.source_file_dandi_dropdown.value)
                    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
                    
                if self.stream_mode == "ros3":
                    io = NWBHDF5IO(s3_url, mode='r', load_namespaces=True, driver='ros3')

                elif self.stream_mode == "fsspec":
                    import fsspec
                    from fsspec.implementations.cached import CachingFileSystem
                    
                    # Create a virtual filesystem based on the http protocol and use caching to save accessed data to RAM.
                    fs = CachingFileSystem(
                        fs=fsspec.filesystem("http"),
                        cache_storage=cache_path,  # Local folder for the cache
                    )
                    f = fs.open(s3_url, "rb")
                    file = h5py.File(f)
                    io = NWBHDF5IO(file=file, load_namespaces=True)

                nwbfile = io.read()
                self.widgets_panel.children = [nwb2widget(nwbfile)]

            elif self.source_path_label.value == "Path to local dir:":
                full_file_path = str(Path(self.source_path_text.value) / self.source_file_dandi_dropdown.value)
                io = NWBHDF5IO(full_file_path, mode='r', load_namespaces=True)
                nwb = io.read()
                self.widgets_panel.children = [nwb2widget(nwb)]

        self.source_file_dandi_button.on_click(stream_dandiset_file)