import cv2
import os
from tkinter import Tk, filedialog

class VideoStitching:
    def __init__(self, desired_width=1200, desired_height=920, frame_interval=2, frame_rate=15):
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.frame_interval = frame_interval
        self.frame_rate = frame_rate
        self.stitcher = cv2.Stitcher_create()
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def get_video_path(self):
        root = Tk()
        root.withdraw()
        return filedialog.askopenfilename(title="Select Video", filetypes=[("MP4 files", "*.mp4")])

    def stitch_videos(self, left_path, right_path, output_path='output.mp4'):
        capL = cv2.VideoCapture(left_path)
        capR = cv2.VideoCapture(right_path)

        frame_count = 0
        output = []

        out = cv2.VideoWriter(output_path, self.fourcc, self.frame_rate, (self.desired_width, self.desired_height))

        while True:
            retL, imageL = capL.read()
            retR, imageR = capR.read()

            if not retL or not retR:
                break

            if frame_count % self.frame_interval == 0:
                status, stitched_image = self.stitcher.stitch([imageL, imageR])
                if status == cv2.Stitcher_OK and stitched_image is not None:
                    resized_stitched_image = cv2.resize(stitched_image, (self.desired_width, self.desired_height))
                    output.append(resized_stitched_image)
                    out.write(resized_stitched_image)

            frame_count += 1

        capL.release()
        capR.release()
        out.release()
        cv2.destroyAllWindows()

        return output