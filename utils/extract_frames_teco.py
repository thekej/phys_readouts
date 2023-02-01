from os.path import isfile, join

import argparse
import cv2
import os
import numpy as np

def extract_frames(args):
    capture = cv2.VideoCapture(args.video_path)
    frameNr = 0
    while (True):
        success, frame = capture.read()
        if success:
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            cv2.imwrite(join(args.save_path, f'{frameNr}.jpg'), frame)

        if not success:
            break
        frameNr = frameNr+1
    capture.release()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video-path', type=str, default=None,
                        help='Path for video file')
    parser.add_argument('-s', '--save-path', type=str, default=None,
                        help='Path for output-directory')
    args = parser.parse_args()
    extract_frames(args)