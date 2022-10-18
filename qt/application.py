from PySide6.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QPushButton, 
    QWidget,
    QVBoxLayout, 
    QHBoxLayout, 
    QComboBox, 
    QStyle,
    QTextBrowser
)
from PySide6.QtCore import Qt
from qtvoila import QtVoila
import sys
import webbrowser

from utils.dandi import (
    get_all_dandisets_metadata, 
    get_dandiset_metadata,
    list_dandiset_files, 
    get_file_url
)


class MyApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(800, 800)
        self.setWindowTitle('Desktop DANDI Explorer')

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
        self.source_choice.addItem("DANDI archive")
        self.source_choice.addItem("Local dir")
        self.source_choice.addItem("Local file")
        self.source_choice.model().item(1).setEnabled(False)
        self.source_choice.model().item(2).setEnabled(False)

        # Select dandi set
        self.dandiset_choice = QComboBox()
        for m in self.all_dandisets_metadata:
            item_name = m.id.split(":")[1].split("/")[0] + " - " + m.name
            self.dandiset_choice.addItem(item_name)
        self.dandiset_choice.view().setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.accept_dandiset_choice = QPushButton()
        icon_1 = self.style().standardIcon(QStyle.SP_ArrowDown)
        self.accept_dandiset_choice.setIcon(icon_1)
        self.accept_dandiset_choice.setToolTip("Read DANDI set")
        self.accept_dandiset_choice.clicked.connect(self.list_dandiset_files)

        self.open_dandiset_choice = QPushButton()
        icon_2 = self.style().standardIcon(QStyle.SP_ComputerIcon)
        self.open_dandiset_choice.setIcon(icon_2)
        self.open_dandiset_choice.setToolTip("Open in DANDI Archive")
        self.open_dandiset_choice.clicked.connect(self.open_webbrowser)

        self.hbox1 = QHBoxLayout()
        self.hbox1.addWidget(self.source_choice, stretch=0)
        self.hbox1.addWidget(self.dandiset_choice, stretch=1)
        self.hbox1.addWidget(self.accept_dandiset_choice, stretch=0)
        self.hbox1.addWidget(self.open_dandiset_choice, stretch=0)
        self.hbox1_w = QWidget()
        self.hbox1_w.setLayout(self.hbox1)

        # Summary info
        self.info_summary = QTextBrowser()
        self.info_summary.setOpenExternalLinks(True)
        self.info_summary.setStyleSheet("font-size: 14px; background: rgba(0,0,0,0%);")
        self.info_summary.setFixedHeight(100)

        # Select file
        self.file_choice = QComboBox()
        self.file_choice.view().setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.accept_file_choice = QPushButton()
        icon_3 = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        self.accept_file_choice.setIcon(icon_3)
        self.accept_file_choice.setToolTip("Visualize NWB file")
        self.accept_file_choice.clicked.connect(self.pass_code_to_voila_widget)

        self.hbox2 = QHBoxLayout()
        self.hbox2.addWidget(self.file_choice, stretch=1)
        self.hbox2.addWidget(self.accept_file_choice, stretch=0)
        self.hbox2_w = QWidget()
        self.hbox2_w.setLayout(self.hbox2)

        layout = QVBoxLayout()
        layout.addWidget(self.hbox1_w, stretch=0)
        layout.addWidget(self.info_summary, stretch=0)
        layout.addWidget(self.hbox2_w, stretch=0)
        layout.addWidget(self.voila_widget, stretch=1)

        self.main_widget = QWidget(self)
        self.main_widget.setLayout(layout)
        self.setCentralWidget(self.main_widget)
        self.show()

    
    def change_data_source(self, value):
        if value == "DANDI archive":
            print("CHANGED: ", value)
        elif value == "Local dir":
            print("CHANGED: ", value)
        elif value == "Local file":
            print("CHANGED: ", value)
    

    def open_webbrowser(self):
        dandiset_id = self.dandiset_choice.currentText().split("-")[0].strip()
        metadata = get_dandiset_metadata(dandiset_id=dandiset_id)
        webbrowser.open(metadata.url)
    

    def list_dandiset_files(self):
        self.file_choice.clear()
        dandiset_id = self.dandiset_choice.currentText().split("-")[0].strip()
        self.info_summary.clear()
        metadata = get_dandiset_metadata(dandiset_id=dandiset_id)
        self.info_summary.append(metadata.description)
        all_files = list_dandiset_files(dandiset_id=dandiset_id)
        for f in all_files:
            self.file_choice.addItem(f)


    def pass_code_to_voila_widget(self):
        self.voila_widget.external_notebook = None
        self.voila_widget.clear()
        file_url = get_file_url(
            dandiset_id=self.dandiset_choice.currentText().split("-")[0].strip(), 
            file_path=self.file_choice.currentText().strip()
        )
        code1 = f"""import fsspec
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
        self.voila_widget.add_notebook_cell(code=code1, cell_type='code')
        # Run Voila
        self.voila_widget.run_voila()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_app = MyApp()
    sys.exit(app.exec())
