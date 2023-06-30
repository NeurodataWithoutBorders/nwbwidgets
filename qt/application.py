import sys
import webbrowser
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStyle,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from qtvoila import QtVoila
from utils.dandi import (
    get_all_dandisets_metadata,
    get_dandiset_metadata,
    get_file_url,
    list_dandiset_files,
)


class MyApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(800, 800)
        self.setWindowTitle("Desktop DANDI Explorer")

        try:
            self.all_dandisets_metadata = get_all_dandisets_metadata()
        except BaseException as e:
            self.all_dandisets_metadata = list()
            print("Failed to fetch DANDI archive datasets: ", e)

        # Voila widget
        self.voila_widget = QtVoila(
            parent=self,
            strip_sources=True,
        )

        # Select source
        self.source_choice = QComboBox()
        self.source_choice.currentTextChanged.connect(self.change_data_source)
        # self.source_choice.addItem(QIcon(':/static/icon_dandi.svg'), "DANDI")
        self.source_choice.addItem("Local dir")
        self.source_choice.addItem("Local file")
        self.source_choice.addItem("DANDI archive")

        # Main Layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.source_choice, stretch=0)
        self.layout.addWidget(self.voila_widget, stretch=1)

        self.create_folder_layout()

        self.main_widget = QWidget(self)
        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)
        self.show()

    def change_data_source(self, value):
        self.delete_item_from_layout()
        if value == "DANDI archive":
            self.create_dandi_layout()
        elif value == "Local dir":
            self.create_folder_layout()
        elif value == "Local file":
            self.create_file_layout()

    def delete_item_from_layout(self):
        # Ref: https://stackoverflow.com/a/9899475/11483674
        child = self.layout.takeAt(1)
        child.widget().deleteLater()

    def browser_local_folder(self):
        folder_path = QFileDialog.getExistingDirectory(parent=self, caption="Open folder", dir=str(Path.home()))
        if folder_path:
            for f in Path(folder_path).glob("*.nwb"):
                self.all_folder_files.addItem(str(f))

    def browser_local_file(self):
        filename, filter = QFileDialog.getOpenFileName(
            parent=self, caption="Open file", dir=str(Path.home()), filter="NWB Files (*.nwb)"
        )
        if filename:
            self.chosen_file.setText(filename)

    def create_folder_layout(self):
        browser_folder_button = QPushButton()
        icon = self.style().standardIcon(QStyle.SP_DialogOpenButton)
        browser_folder_button.setIcon(icon)
        browser_folder_button.setToolTip("Browser local dir")
        browser_folder_button.clicked.connect(self.browser_local_folder)

        self.all_folder_files = QComboBox()

        accept_folder_file_button = QPushButton()
        icon_2 = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        accept_folder_file_button.setIcon(icon_2)
        accept_folder_file_button.setToolTip("Visualize NWB file")
        accept_folder_file_button.clicked.connect(self.pass_code_to_voila_widget)

        hbox = QHBoxLayout()
        hbox.addWidget(browser_folder_button, stretch=0)
        hbox.addWidget(self.all_folder_files, stretch=1)
        hbox.addWidget(accept_folder_file_button, stretch=0)
        w = QWidget()
        w.setLayout(hbox)

        self.layout.insertWidget(1, w)

    def create_file_layout(self):
        browser_file_button = QPushButton()
        icon = self.style().standardIcon(QStyle.SP_DialogOpenButton)
        browser_file_button.setIcon(icon)
        browser_file_button.setToolTip("Browser local dir")
        browser_file_button.clicked.connect(self.browser_local_file)

        self.chosen_file = QLineEdit()

        accept_file_button = QPushButton()
        icon_2 = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        accept_file_button.setIcon(icon_2)
        accept_file_button.setToolTip("Visualize NWB file")
        accept_file_button.clicked.connect(self.pass_code_to_voila_widget)

        hbox = QHBoxLayout()
        hbox.addWidget(browser_file_button, stretch=0)
        hbox.addWidget(self.chosen_file, stretch=1)
        hbox.addWidget(accept_file_button, stretch=0)
        w = QWidget()
        w.setLayout(hbox)

        self.layout.insertWidget(1, w)

    def create_dandi_layout(self):
        self.dandiset_choice = QComboBox()
        for m in self.all_dandisets_metadata:
            item_name = m.id.split(":")[1].split("/")[0] + " - " + m.name
            self.dandiset_choice.addItem(item_name)
        self.dandiset_choice.view().setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        accept_dandiset_choice = QPushButton()
        icon_1 = self.style().standardIcon(QStyle.SP_ArrowDown)
        accept_dandiset_choice.setIcon(icon_1)
        accept_dandiset_choice.setToolTip("Read DANDI set")
        accept_dandiset_choice.clicked.connect(self.list_dandiset_files)

        open_dandiset_choice = QPushButton()
        icon_2 = self.style().standardIcon(QStyle.SP_ComputerIcon)
        open_dandiset_choice.setIcon(icon_2)
        open_dandiset_choice.setToolTip("Open in DANDI Archive")
        open_dandiset_choice.clicked.connect(self.open_webbrowser)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.dandiset_choice, stretch=1)
        hbox1.addWidget(accept_dandiset_choice, stretch=0)
        hbox1.addWidget(open_dandiset_choice, stretch=0)
        hbox1_w = QWidget()
        hbox1_w.setLayout(hbox1)

        # Summary info
        self.info_summary = QTextBrowser()
        self.info_summary.setOpenExternalLinks(True)
        self.info_summary.setStyleSheet("font-size: 14px; background: rgba(0,0,0,0%);")
        self.info_summary.setFixedHeight(100)

        # Select file
        self.dandi_file_choice = QComboBox()
        self.dandi_file_choice.view().setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        accept_file_choice = QPushButton()
        icon_3 = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        accept_file_choice.setIcon(icon_3)
        accept_file_choice.setToolTip("Visualize NWB file")
        accept_file_choice.clicked.connect(self.pass_code_to_voila_widget)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.dandi_file_choice, stretch=1)
        hbox2.addWidget(accept_file_choice, stretch=0)
        hbox2_w = QWidget()
        hbox2_w.setLayout(hbox2)

        dandi_source_layout = QVBoxLayout()
        dandi_source_layout.addWidget(hbox1_w, stretch=0)
        dandi_source_layout.addWidget(self.info_summary, stretch=0)
        dandi_source_layout.addWidget(hbox2_w, stretch=0)
        w = QWidget()
        w.setLayout(dandi_source_layout)

        self.layout.insertWidget(1, w)

    def open_webbrowser(self):
        dandiset_id = self.dandiset_choice.currentText().split("-")[0].strip()
        metadata = get_dandiset_metadata(dandiset_id=dandiset_id)
        webbrowser.open(metadata.url)

    def list_dandiset_files(self):
        self.dandi_file_choice.clear()
        dandiset_id = self.dandiset_choice.currentText().split("-")[0].strip()
        self.info_summary.clear()
        metadata = get_dandiset_metadata(dandiset_id=dandiset_id)
        self.info_summary.append(metadata.description)
        all_files = list_dandiset_files(dandiset_id=dandiset_id)
        for f in all_files:
            self.dandi_file_choice.addItem(f)

    def pass_code_to_voila_widget(self):
        self.voila_widget.external_notebook = None
        self.voila_widget.clear()

        if self.source_choice.currentText() == "DANDI archive":
            file_url = get_file_url(
                dandiset_id=self.dandiset_choice.currentText().split("-")[0].strip(),
                file_path=self.dandi_file_choice.currentText().strip(),
            )
            code = f"""import fsspec
import pynwb
import h5py
from fsspec.implementations.cached import CachingFileSystem
from nwbwidgets import nwb2widget

fs = CachingFileSystem(
    fs=fsspec.filesystem("http"),
    cache_storage="nwb-cache",  # Local folder for the cache
)

# next, open the file
f = fs.open('{file_url}', "rb")
file = h5py.File(f)
io = pynwb.NWBHDF5IO(file=file, load_namespaces=True)
nwbfile = io.read()
nwb2widget(nwbfile)"""

        elif self.source_choice.currentText() == "Local dir":
            file_path = self.all_folder_files.currentText()
            code = f"""import pynwb
from nwbwidgets import nwb2widget

io = pynwb.NWBHDF5IO('{file_path}', load_namespaces=True)
nwbfile = io.read()
nwb2widget(nwbfile)"""

        elif self.source_choice.currentText() == "Local file":
            file_path = self.chosen_file.text()
            code = f"""import pynwb
from nwbwidgets import nwb2widget

io = pynwb.NWBHDF5IO('{file_path}', load_namespaces=True)
nwbfile = io.read()
nwb2widget(nwbfile)"""

        self.voila_widget.add_notebook_cell(code=code, cell_type="code")
        # Run Voila
        self.voila_widget.run_voila()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    my_app = MyApp()
    sys.exit(app.exec())
