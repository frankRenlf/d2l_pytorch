# -*- coding: UTF-8 -*-
"""
    @Author : Frank.Ren
    @Project : pytorch 
    @Product : PyCharm
    @createTime : 2023/8/16 17:58 
    @Email : sc19lr@leeds.ac.uk
    @github : https://github.com/frankRenlf
    @Description : 
"""

import numpy as np
import cv2 as cv

if __name__ == "__main__":
    cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)

    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*"MP4V")
    out = cv.VideoWriter('output.mp4', fourcc, 15, size)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # write the flipped frame
        out.write(frame)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv.destroyAllWindows()
