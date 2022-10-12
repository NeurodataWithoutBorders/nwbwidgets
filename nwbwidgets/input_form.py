import ipywidgets as widgets
from pathlib import Path

from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget
from dandi.dandiapi import DandiAPIClient


def make_input_panel():
    source_options_radio = widgets.RadioButtons(options=['dandi', 'local dir', 'local file'], value='dandi')
    source_options_label = widgets.Label('Source:')
    source_options = widgets.VBox([source_options_label, source_options_radio], layout=widgets.Layout(width='150px', overflow=None))

    source_path_text = widgets.Text(value="", layout={'width': '300px'})
    source_path_dandi_button = widgets.Button(icon='angle-right', layout={'width': '50px'})
    source_path_lower = widgets.HBox([source_path_text, source_path_dandi_button])
    source_path_label = widgets.Label('Dandiset number:')
    source_path = widgets.VBox([source_path_label, source_path_lower], layout=widgets.Layout(width='400px', overflow=None))

    source_dandiset_info = widgets.Label("")
    source_file_dandi_dropdown = widgets.Dropdown(options=[])
    source_file_dandi_button = widgets.Button(icon='angle-down', layout={'width': '50px'})
    source_file_dandi_hbox = widgets.HBox([source_file_dandi_dropdown, source_file_dandi_button])
    source_file_dandi_vbox = widgets.VBox([source_dandiset_info, source_file_dandi_hbox], layout=widgets.Layout(width='400px', overflow=None))

    input_form = widgets.HBox([source_options, source_path, source_file_dandi_vbox], layout={"border": "1px solid gray"})
    widgets_panel = widgets.VBox([])
    full_panel = widgets.VBox([input_form, widgets_panel])

    def updated_source(args):
        if args['new'] == "dandi":
            source_path_label.value = "Dandiset number:"
            source_path_text.value = ""
            source_path_dandi_button.icon='angle-right'
            source_file_dandi_vbox.layout.display = None
        elif args['new'] == "local dir":
            source_path_label.value = "Path to local dir:"
            source_path_text.value = ""
            source_path_dandi_button.icon='angle-right'
            source_file_dandi_vbox.layout.display = None
        elif args['new'] == "local file":
            source_path_label.value = "Path to local file:"
            source_path_text.value = ""
            source_path_dandi_button.icon='angle-down'
            source_file_dandi_vbox.layout.display = 'none'

    source_options_radio.observe(updated_source, 'value')

    def list_dandiset_files(args):
        source_file_dandi_dropdown.options = []
        if source_path_label.value == "Dandiset number:":
            with DandiAPIClient() as client:
                dandiset = client.get_dandiset(dandiset_id=source_path_text.value, version_id="draft")
                try:
                    # Get dandiset info
                    source_dandiset_info.value = dandiset.json_dict()["version"]["name"]
                    # Populate dropdown with all files
                    all_files = [i.dict().get("path") for i in dandiset.get_assets()]
                    source_file_dandi_dropdown.options = all_files
                except:
                    source_dandiset_info.value = "Invalid Dandiset number"
        elif source_path_label.value == "Path to local dir:":
            if Path(source_path_text.value).is_dir():
                source_dandiset_info.value = str(Path(source_path_text.value))
                all_files = [f.name for f in Path(source_path_text.value).glob("*.nwb")]
                source_file_dandi_dropdown.options = all_files
            else:
                source_dandiset_info.value = "Invalid local dir path"

    source_path_dandi_button.on_click(list_dandiset_files)

    def stream_dandiset_file(args):
        widgets_panel.children = [widgets.Label("loading...")]
        if source_path_label.value == "Dandiset number:":
            with DandiAPIClient() as client:
                asset = client.get_dandiset(dandiset_id=source_path_text.value, version_id="draft").get_asset_by_path(source_file_dandi_dropdown.value)
                s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
                io = NWBHDF5IO(s3_url, mode='r', load_namespaces=True, driver='ros3')
                nwb = io.read()
                widgets_panel.children = [nwb2widget(nwb)]
        elif source_path_label.value == "Path to local dir:":
            full_file_path = str(Path(source_path_text.value) / source_file_dandi_dropdown.value)
            io = NWBHDF5IO(full_file_path, mode='r', load_namespaces=True)
            nwb = io.read()
            widgets_panel.children = [nwb2widget(nwb)]

    source_file_dandi_button.on_click(stream_dandiset_file)

    return full_panel