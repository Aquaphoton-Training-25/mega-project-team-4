from PyQt6 import QtCore, QtGui, QtWidgets

class Bluetooth_Fend(QtWidgets.QWidget):
    def __init__(self, backend,):
        super().__init__()
        self.backend = backend
        self.setupUi()

    def setupUi(self):
        self.setObjectName("Form")
        self.resize(400, 300)
        self.Bluetooth_icon = QtWidgets.QLabel(self)
        self.Bluetooth_icon.setGeometry(QtCore.QRect(90, 10, 221, 201))
        self.Bluetooth_icon.setText("")
        self.Bluetooth_icon.setPixmap(QtGui.QPixmap("../../Bluetooth.png"))
        self.Bluetooth_icon.setScaledContents(True)
        self.Bluetooth_icon.setObjectName("Bluetooth_icon")

        self.Low = QtWidgets.QLabel(self)
        self.Low.setGeometry(QtCore.QRect(110, 220, 31, 41))
        self.Low.setText("")
        self.Low.setObjectName("Low")

        self.Moderate = QtWidgets.QLabel(self)
        self.Moderate.setGeometry(QtCore.QRect(180, 220, 31, 41))
        self.Moderate.setText("")
        self.Moderate.setObjectName("Moderate")

        self.Strong = QtWidgets.QLabel(self)
        self.Strong.setGeometry(QtCore.QRect(250, 220, 31, 41))
        self.Strong.setText("")
        self.Strong.setObjectName("Strong")

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        # Connect the backend signal to the slot to update the UI
        self.backend.connection_status_changed.connect(self.update_labels)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Form", "Form"))

    def update_labels(self, status):
        # Define images for connection statuses
        images = {
            'weak': 'Weak.JPG',
            'moderate': 'Moderate.JPG',
            'strong': 'Strong.JPG'
        }

        # Clear all labels first
        self.Low.setPixmap(QtGui.QPixmap())
        self.Moderate.setPixmap(QtGui.QPixmap())
        self.Strong.setPixmap(QtGui.QPixmap())

        if status == 'none':
            # No connection, all labels are empty
            pass
        elif status == 'weak':
            self.Low.setPixmap(QtGui.QPixmap(images['weak']))
        elif status == 'moderate':
            self.Low.setPixmap(QtGui.QPixmap(images['moderate']))
            self.Moderate.setPixmap(QtGui.QPixmap(images['moderate']))
        elif status == 'strong':
            self.Low.setPixmap(QtGui.QPixmap(images['strong']))
            self.Moderate.setPixmap(QtGui.QPixmap(images['strong']))
            self.Strong.setPixmap(QtGui.QPixmap(images['strong']))
