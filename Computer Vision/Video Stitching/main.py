import cv2
import numpy as np

from tkinter import Tk, filedialog

def get_video_path():
    root = Tk()
    root.withdraw() 
    return filedialog.askopenfilename(title="Select Video", filetypes=[("MP4 files", "*.mp4")])


left_video_path = get_video_path()
right_video_path = get_video_path()

left=cv2.VideoCapture(left_video_path)

right=cv2.VideoCapture(right_video_path)

frame_width=int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_rate=int(left.get(cv2.CAP_PROP_FPS))


fourcc=cv2.VideoWriter_fourcc(*'mp4v')
out=cv2.VideoWriter('output.mp4',fourcc,frame_rate,(frame_width*2,frame_height))

while True:
    ret1,frame1=left.read()

    ret2,frame2=right.read()

    if not ret1 and not ret2:
        break

    frame2=cv2.resize(frame2,(frame_width,frame_height))

    canvas=np.zeros((frame_height,frame_width*2,3),dtype=np.uint8)

    canvas[:,:frame_width]=frame1

    canvas[:,frame_width:]=frame2

    out.write(canvas)

left.release()
right.release()
out.release()    













