import sys
import cv2
import serial
import time
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from Remote_Fend import Ui_Live_feed  # Import your UI class
from Bluetooth_Bend import Bluetooth_Bend  # Import the Bluetooth backend
from Bluetooth_Fend import Bluetooth_Fend  # Import the Bluetooth frontend

class MyApp_Remo(QtWidgets.QWidget):
    switch_signal = pyqtSignal()  # Define a custom signal

    def __init__(self, switch_handler=None):
        super().__init__()
        self.ui = Ui_Live_feed()
        self.ui.setupUi(self, switch_handler=self.handle_switch_mode)  # Setup UI
        # Send the 'M' command and print a confirmation message
        if self.ser:
            self.send_command('M\n')
            print("Sent command 'M' to start Manual mode.")

        # Initialize Bluetooth backend
        self.bluetooth_backend = Bluetooth_Bend()
        
        # Initialize Bluetooth frontend and connect to backend
        self.bluetooth_frontend = Bluetooth_Fend(self.bluetooth_backend)
        self.bluetooth_frontend.show()  # Show Bluetooth status window
        
        # Initialize serial connection to HC-06
        try:
            self.ser = serial.Serial('COM7', 9600, timeout=1)  # For Windows
            print("Connected to HC-06")
        except serial.SerialException as e:
            print(f"Error: {e}")
            self.ser = None

        # Connect button actions
        self.ui.Webcam_button.clicked.connect(self.open_webcam)
        self.ui.Screenshot_button.clicked.connect(self.take_screenshot)
        self.ui.Forward.clicked.connect(lambda: self.move('1', '128'))  # Move forward at full speed
        self.ui.Reverse.clicked.connect(lambda: self.move('2', '128'))  # Move backward at full speed
        self.ui.Left.clicked.connect(lambda: self.move('3', '128'))  # Turn left at half speed
        self.ui.Right.clicked.connect(lambda: self.move('4', '128'))  # Turn right at half speed
        self.ui.Accelerate.clicked.connect(lambda: self.accelerate(self.direction, self.speed))
        self.ui.Decelerate.clicked.connect(lambda: self.decelerate(self.direction, self.speed))
        self.ui.Stop.clicked.connect(self.stop)
        self.ui.switch_button.clicked.connect(lambda: self.switch_to_autonomous_mode('30'))

        # Initialize variables for camera and timer
        self.current_frame = None
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Initialize speed and direction
        self.speed = 80  # Default initial speed
        self.direction = '1'  # Default direction (forward)
        
        # Timer to update speed display every 100 ms
        self.speed_display_timer = QTimer(self)
        self.speed_display_timer.timeout.connect(self.update_speed_display)
        self.speed_display_timer.start(100)  # Update every 100 ms

    def accelerate(self, direction, speed):
        if direction == '1':
            if speed < 255:
                self.speed = min(speed + 50, 255)
                self.move('1', str(self.speed))
            else:
                print("Car already at Max Forward Speed")
        elif direction == '2':
            if speed < 255:
                self.speed = min(speed + 50, 255)
                self.move('2', str(self.speed))
            else:
                print("Car already at Max Reverse Speed")

    def decelerate(self, direction, speed):
        if direction == '1':
            if speed > 80:
                self.speed = max(speed - 50, 80)
                self.move('1', str(self.speed))
            else:
                print("Car already at Min Forward Speed")
        elif direction == '2':
            if speed > 80:
                self.speed = max(speed - 50, 80)
                self.move('2', str(self.speed))
            else:
                print("Car already at Min Reverse Speed")

    def update_speed_display(self):
        self.ui.lineSpeed.setText(str(self.speed))  # Update LineSpeed QLineEdit with the current speed

    def send_command(self, command):
        if self.ser:
            try:
                self.ser.write(command.encode())
                time.sleep(0.1)  # Wait for the command to be processed
            except serial.SerialException as e:
                print(f"Serial Exception: {e}")
        else:
            print("Serial port not initialized.")

    def move(self, direction, speed):
        self.direction = direction
        command = f"{direction}{speed}\n"
        print(f"Sending command: {command.strip()}")
        self.send_command(command)

    def stop(self):
        print("Stop button clicked")
        self.send_command('0\n')  # Command to stop the car

    def handle_switch_mode(self):
        print("Switch mode button clicked")
        self.switch_signal.emit()  # Emit signal when button is clicked

    def switch_to_manual_mode(self):
        print("Switching to manual mode")
        self.send_command('M\n')

    def switch_to_autonomous_mode(self, distance):
        command = f"A{distance}\n"
        print(f"Switching to autonomous mode with distance {distance} cm")
        self.send_command(command)
        self.disconnect()

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected from HC-06")
        else:
            print("Serial port already closed or not initialized")



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
        if self.speed_display_timer.isActive():
            self.speed_display_timer.stop()
        if self.ser and self.ser.is_open:
            self.ser.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp_Remo()
    window.show()
    sys.exit(app.exec())
