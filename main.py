import sys
from PyQt6 import QtWidgets as widgets
from ui.main_window import MainWindow

if __name__ == '__main__':
    app = widgets.QApplication(sys.argv)
    app.setStyle(widgets.QStyleFactory.create('Fusion'))
    app.setPalette(app.style().standardPalette())

    window = MainWindow()
    window.setMouseTracking(True)
    window.show()

    sys.exit(app.exec())