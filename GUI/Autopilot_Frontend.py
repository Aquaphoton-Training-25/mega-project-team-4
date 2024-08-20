from PyQt6 import QtCore, QtGui, QtWidgets

class Ui_Autopilot_SubW(object):
    def setupUi(self, Autopilot_SubW, switch_handler=None):
        # Setup main window
        Autopilot_SubW.setObjectName("Autopilot_SubW")
        Autopilot_SubW.resize(int(1504 * 1.6), int(542 * 1.5))

        scale_x = 1.6
        scale_y= 1.5

        # Screenshot label
        self.Screenshot = QtWidgets.QLabel(parent=Autopilot_SubW)
        self.Screenshot.setGeometry(QtCore.QRect(int(600 * 1.6), int(200 * 1.5), int(371 * 1.6), int(121 * 1.5)))
        self.Screenshot.setObjectName("Screenshot")

        # Webcam label
        self.Webcam = QtWidgets.QLabel(parent=Autopilot_SubW)
        self.Webcam.setGeometry(QtCore.QRect(int(0 * 1.6), int(0 * 1.5), int(500 * 1.6), int(400 * 1.5)))
        self.Webcam.setObjectName("Webcam")

        # Webcam button
        self.Webcam_button = QtWidgets.QPushButton(parent=Autopilot_SubW)
        self.Webcam_button.setGeometry(QtCore.QRect(int(200 * 1.6), int(291 * 1.5), int(51 * 1.6), int(41 * 1.5)))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../Documents/web3.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.Webcam_button.setIcon(icon)
        self.Webcam_button.setIconSize(QtCore.QSize(int(50 * 1.6), int(55 * 1.5)))
        self.Webcam_button.setObjectName("Webcam_button")

        # Screenshot button
        self.Screenshot_button = QtWidgets.QPushButton(parent=Autopilot_SubW)
        self.Screenshot_button.setGeometry(QtCore.QRect(int(260 * 1.6), int(291 * 1.5), int(51 * 1.6), int(41 * 1.5)))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("Screenshot.jpg"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.Screenshot_button.setIcon(icon1)
        self.Screenshot_button.setIconSize(QtCore.QSize(int(50 * 1.6), int(60 * 1.5)))
        self.Screenshot_button.setObjectName("Screenshot_button")

        # Switch button
        self.switch_button = QtWidgets.QPushButton(parent=Autopilot_SubW, clicked=switch_handler)
        self.switch_button.setGeometry(QtCore.QRect(int(140 * 1.6), int(340 * 1.5), int(251 * 1.6), int(35 * 1.5)))
        font = QtGui.QFont()
        font.setFamily("Georgia Pro Cond Semibold")
        font.setPointSize(int(10 * 1.5))
        font.setBold(True)
        font.setWeight(75)
        self.switch_button.setFont(font)
        self.switch_button.setObjectName("switch_button")

        # Label and Stop button
        self.label_2 = QtWidgets.QLabel(parent=Autopilot_SubW)
        self.label_2.setGeometry(QtCore.QRect(int(150 * 1.6), int(270 * 1.5), int(191 * 1.6), int(61 * 1.5)))
        self.label_2.setObjectName("label_2")

        self.Stop = QtWidgets.QPushButton(parent=Autopilot_SubW)
        self.Stop.setGeometry(QtCore.QRect(int(49 * 1.6), int(298 * 1.5), int(45 * 1.6), int(45 * 1.5)))
        self.Stop.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("Center.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.Stop.setIcon(icon6)
        self.Stop.setIconSize(QtCore.QSize(int(45 * 1.6), int(45 * 1.5)))
        self.Stop.setObjectName("Stop")

        # Additional labels
        self.label = QtWidgets.QLabel(parent=Autopilot_SubW)
        self.label.setGeometry(QtCore.QRect(int(40 * 1.6), int(30 * 1.5), int(51 * 1.6), int(21 * 1.5)))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("Strong.JPG"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")

        self.label_3 = QtWidgets.QLabel(parent=Autopilot_SubW)
        self.label_3.setGeometry(QtCore.QRect(int(40 * 1.6), int(30 * 1.5), int(51 * 1.6), int(21 * 1.5)))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("Weak.JPG"))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")

        # Raise widgets to ensure correct stacking order
        self.Screenshot.raise_()
        self.Webcam.raise_()
        self.switch_button.raise_()
        self.label_2.raise_()
        self.label.raise_()
        self.label_3.raise_()
        self.Webcam_button.raise_()
        self.Screenshot_button.raise_()
        self.Stop.raise_()

        # Switch button
        self.switch_button = QtWidgets.QPushButton(parent=Autopilot_SubW, clicked=switch_handler)
        self.switch_button.setGeometry(QtCore.QRect(
        int(140 * 1.6), int(340 * 1.5), int(251 * 1.6), int(35 * 1.5)))
        font = QtGui.QFont()
        font.setFamily("Georgia Pro Cond Semibold")
        font.setPointSize(int(10))
        font.setBold(True)
        font.setWeight(75)
        self.switch_button.setFont(font)
        self.switch_button.setObjectName("switch_button")

        # Line Edits
        self.lineSpeed = QtWidgets.QLineEdit(parent=Autopilot_SubW)
        self.lineSpeed.setGeometry(QtCore.QRect(int(440 * scale_x), int(310 * scale_y), int(41 * scale_x), int(20 * scale_y)))
        self.lineSpeed.setObjectName("lineSpeed")
        self.lineEdit_2 = QtWidgets.QLineEdit(parent=Autopilot_SubW)
        self.lineEdit_2.setGeometry(QtCore.QRect(int(440 * scale_x), int(332 * scale_y), int(41 * scale_x), int(20 * scale_y)))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(parent=Autopilot_SubW)
        self.lineEdit_3.setGeometry(QtCore.QRect(int(440 * scale_x), int(354 * scale_y), int(41 * scale_x), int(20 * scale_y)))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.Speed = QtWidgets.QLabel(parent=Autopilot_SubW)
        self.Speed.setGeometry(QtCore.QRect(int(398 * scale_x), int(316 * scale_y), int(21 * scale_x), int(10 * scale_y)))
        self.Speed.setWordWrap(True)
        self.Speed.setObjectName("Speed")
        self.Voltage = QtWidgets.QLabel(parent=Autopilot_SubW)
        self.Voltage.setGeometry(QtCore.QRect(int(397 * scale_x), int(338 * scale_y), int(35 * scale_x), int(10 * scale_y)))
        self.Voltage.setObjectName("Voltage")
        self.Current = QtWidgets.QLabel(parent=Autopilot_SubW)
        self.Current.setGeometry(QtCore.QRect(int(397 * scale_x), int(355 * scale_y), int(41 * scale_x), int(20 * scale_y)))
        self.Current.setScaledContents(False)
        self.Current.setObjectName("Current")

        

        # Connect the switch_mode button click to the switch_handler
        if switch_handler:
            self.switch_button.clicked.connect(switch_handler)

        # Setup the UI translations and connections
        self.retranslateUi(Autopilot_SubW)
        QtCore.QMetaObject.connectSlotsByName(Autopilot_SubW)

    def retranslateUi(self, Autopilot_SubW):
        _translate = QtCore.QCoreApplication.translate
        Autopilot_SubW.setWindowTitle(_translate("Autopilot_SubW", "Form"))
        self.switch_button.setText(_translate("Autopilot_SubW", "Switch Mode"))
        self.Speed.setText(_translate("Autopilot_SubW", "Speed"))
        self.Voltage.setText(_translate("Autopilot_SubW", "Voltage/V"))
        self.Current.setText(_translate("Autopilot_SubW", "Current/mA"))
