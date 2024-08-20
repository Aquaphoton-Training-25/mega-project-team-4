from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6 import uic
from Autopilot_Backend import MyApp_Auto  # Import your sub-window UI class
from Remote_Bend import MyApp_Remo
import sys

#with open("bobo.py", "w") as fout:
 # uic.compileUi("Webcam_Bend.ui", fout)

class UI(QMainWindow):
    def Switch_ModW(self):
      # Check if there is an active subwindow
     active_subwindow = self.mdi.activeSubWindow()

     if not active_subwindow:
        print("No active subwindow found")
        return

     # Get the title of the active subwindow
     active_title = active_subwindow.windowTitle()
     print(f"Active subwindow title: {active_title}")

     # Check if the title matches either "Remote" or "Autopilot"
     if active_title == "Remote":
        print("Switching from Remote to Autopilot")

        # Connect to the destroyed signal to open the new window after the current one closes
        active_subwindow.destroyed.connect(self.add_autopilot_window)

        # Close the Remote subwindow
        active_subwindow.close()

     elif active_title == "Autopilot":
        print("Switching from Autopilot to Remote")

        # Connect to the destroyed signal to open the new window after the current one closes
        active_subwindow.destroyed.connect(self.add_remote_window)

        # Close the Autopilot subwindow
        active_subwindow.close()

     else:
        print("Unknown subwindow title")



    def __init__(self):
        super().__init__()

        # Load UI file (assumed it is used for the main window)
        uic.loadUi("Multi_Fend.ui", self)
        
        # Define widgets
        self.mdi = self.findChild(QMdiArea, "mdiArea")
        self.remote = self.findChild(QPushButton, "Remote")
        self.autopilot = self.findChild(QPushButton, "Autopilot")
        self.logout = self.findChild(QPushButton, "Logout")

        # Connect buttons to their respective methods
        self.autopilot.clicked.connect(self.add_window)
        self.remote.clicked.connect(self.add_window)
        self.logout.clicked.connect(self.close)
    
        self.show()

    def add_window(self):
        # Capture the sender button
        sender_button = self.sender()

        
        if sender_button == self.remote:
            self.add_remote_window()

            
        elif sender_button == self.autopilot:
            # Directly add the Autopilot sub-window
             self.add_autopilot_window()

        # Disable the buttons while the subwindow is open
        self.remote.setEnabled(False)
        self.autopilot.setEnabled(False)

    def add_autopilot_window(self):
        # Create sub-window
        sub = QMdiSubWindow()
        sub.setFixedSize(int(1.6* 500),int(1.5 * 400))
        sub.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)  # Ensure proper deletion on close

        # Create an instance of the Autopilot Sub Window UI class
        self.ui_subwindow = MyApp_Auto(switch_handler=self.Switch_ModW)  # Pass switch_handler if needed
        self.ui_subwindow.switch_signal.connect(self.Switch_ModW)
        sub.setWidget(self.ui_subwindow)  # Set widget in the sub-window
        sub.setWindowTitle("Autopilot")
        
        # Add the Autopilot sub-window to the MDI area
        self.mdi.addSubWindow(sub)
        sub.show()

        # Re-enable the buttons when the subwindow is closed
        sub.destroyed.connect(lambda: self.remote.setEnabled(True))
        sub.destroyed.connect(lambda: self.autopilot.setEnabled(True))

    def add_remote_window(self):
        # Create sub-window
        sub = QMdiSubWindow()
        sub.setFixedSize(int(1.6* 500),int(1.5 * 400))
        sub.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # Create an instance of MyApp
        self.ui_subwindow = MyApp_Remo(switch_handler=self.Switch_ModW)  # Pass switch_handler if needed
         # Connect the switch_signal to the Switch_ModW method
        self.ui_subwindow.switch_signal.connect(self.Switch_ModW)
        sub.setWidget(self.ui_subwindow)  # Set MyApp directly as the widget
        sub.setWindowTitle("Remote")
        
        # Add the Remote sub-window to the MDI area
        self.mdi.addSubWindow(sub)
        sub.show()

        # Re-enable the buttons when the subwindow is closed
        sub.destroyed.connect(lambda: self.remote.setEnabled(True))
        sub.destroyed.connect(lambda: self.autopilot.setEnabled(True))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UI()
    window.setMinimumSize(1100, 650)
    window.show()
    sys.exit(app.exec())
