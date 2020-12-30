from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QApplication
import sys
import hello_world



class Ui_MainWindow(QtWidgets.QMainWindow, hello_world.Ui_MainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.setupUi(self)



def main():
    app = QApplication(sys.argv)
    form = Ui_MainWindow()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()