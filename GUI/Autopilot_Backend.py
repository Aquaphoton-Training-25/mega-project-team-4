import sys
import cv2
import serial
import time
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from Autopilot_Frontend import Ui_Autopilot_SubW  # Import your UI class

class MyApp_Auto(QtWidgets.QWidget):
    switch_signal = pyqtSignal()  # Define a custom signal

    def __init__(self, switch_handler=None):
        super().__init__()
        self.ui = Ui_Autopilot_SubW()
        self.ui.setupUi(self, switch_handler=self.handle_switch_mode)  # Setup UI

        # Initialize serial connection to HC-06
        try:
            self.ser = serial.Serial('COM7', 9600, timeout=1)  # Updated to COM7
            print("Connected to HC-06")
        except serial.SerialException as e:
            print(f"Error: {e}")
            self.ser = None

        # Send the 'A' command and print a confirmation message
        if self.ser:
            self.send_command('A\n')
            print("Sent command 'A' to start auto mode.")

        # Connect button actions
        self.ui.Webcam_button.clicked.connect(self.open_webcam)
        self.ui.Screenshot_button.clicked.connect(self.take_screenshot)
        self.ui.switch_button.clicked.connect(lambda: self.switch_to_manual_mode())
        self.ui.Stop.clicked.connect(self.stop)  # Connect Stop button

        # Initialize variables for camera and timer
        self.current_frame = None
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def switch_to_manual_mode(self):
        print("Performing additional action before switching to manual mode")

        # Example of an additional action

        print("Switching to manual mode")
        self.send_command('M\n')  # Switch to manual mode

    def send_command(self, command):
        if self.ser:
            try:
                self.ser.write(command.encode())
                time.sleep(0.1)  # Wait for the command to be processed
            except serial.SerialException as e:
                print(f"Serial Exception: {e}")
        else:
            print("Serial port not initialized.")

    def stop(self):
        print("Stop button clicked")
        self.send_command('0\n')  # Command to stop the car

    def handle_switch_mode(self):
        print("Switch mode button clicked")
        self.switch_signal.emit()  # Emit signal when button is clicked

    def open_webcam(self):
        # Open webcam and handle frame updates
        if self.capture is None or not self.capture.isOpened():
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                print("Error: Could not open webcam.")
                return

            # Start the timer to continuously update the frame
            self.timer.start(30)  # Update every 30 ms

    def update_frame(self):
        if self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                self.current_frame = frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.flip(rgb_frame, 1)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.ui.Webcam.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.ui.Webcam.width(), self.ui.Webcam.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            else:
                print("Error: Could not read frame from webcam.")
        else:
            print("Webcam not open.")

    def take_screenshot(self):
        if self.current_frame is not None:
            bgr_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            bgr_frame = cv2.flip(bgr_frame, 1)
            cv2.imwrite("screenshot.png", bgr_frame)
            print("Screenshot saved as screenshot.png")
            self.display_screenshot("screenshot.png")
        else:
            print("No frame available to take a screenshot.")

    def display_screenshot(self, filename):
        image = cv2.imread(filename)
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.ui.Screenshot.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.ui.Screenshot.width(), self.ui.Screenshot.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        if self.timer is not None:
            self.timer.stop()
        if self.ser and self.ser.is_open:
            self.ser.close()
        event.accept()
