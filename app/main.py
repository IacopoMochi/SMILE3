import sys
from PyQt6 import QtWidgets

from app.controllers.main_window_controller import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exit(app.exec())


if __name__ == '__main__':
    main()
