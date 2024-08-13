import cv2
import os
from tkinter import Tk, filedialog

def get_video_path():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(title="Select Video", filetypes=[("MP4 files", "*.mp4")])

left_path = get_video_path()
right_path = get_video_path()

capL = cv2.VideoCapture(left_path)
capR = cv2.VideoCapture(right_path)

desired_width = 1200
desired_height = 920

stitcher = cv2.Stitcher_create()

frame_interval = 2
frame_rate = 15
frame_count = 0
output = []

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, frame_rate, (desired_width, desired_height))


while True:
    retL, imageL = capL.read()
    retR, imageR = capR.read()

    if not retL or not retR:
        break
     
    if frame_count % frame_interval == 0:
        status, stitched_image = stitcher.stitch([imageL, imageR])
        if status == cv2.Stitcher_OK and stitched_image is not None:

            resized_stitched_image = cv2.resize(stitched_image, (desired_width, desired_height))
            output.append(resized_stitched_image)
            out.write(resized_stitched_image)
           
           

    frame_count += 1

capL.release()
capR.release()
out.release()
cv2.destroyAllWindows()

