#!/usr/bin/env python3
"""
Python script to calibrate rgb camera using Aruco board
"""

import cv2
import glob
import pickle
import argparse
import numpy as np
from calibration import calibrate_camera_arucoboard


argparser = argparse.ArgumentParser()
argparser.add_argument("--devID", type=int, default=2, help="Camera device Id.", required=False)
argparser.add_argument("--calibrateOnly", type=bool, default=False, help="Calibration flag, else capture image.", required=False)
argparser.add_argument("--imgPath", type=str, default="./data/ImgData", help="Images path", required=False)
camID = argparser.parse_args().devID
calibFlag = argparser.parse_args().calibrateOnly
image_path = argparser.parse_args().imgPath

if (calibFlag):
    print("Performing calibration\n")
else:
    print("Starting image collection\n[s] -> save image frame\n[q] -> quit image collection\n")

def main():
    if (not calibFlag):
        cam = cv2.VideoCapture(camID)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640.0)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480.0)

        if (not cam.isOpened()):
            print(f"Not able to open camera; device id: {camID}")
            exit(0)

        imageCollection = []
        count = 0

        # image collection loop
        while True:
            _, bgr = cam.read()
            if (not _):
                print("Did not get proper image!")
                continue
            cv2.imshow("Image collection window", bgr)
            key = cv2.waitKey(1)
            if (key == ord('s')):
                imageCollection.append(bgr)
                # save image
                cv2.imwrite(image_path + f"/image{count+1}.png", bgr)
                print(f"Collected image: {count+1}")
                count+=1
            if (key == ord('q')):
                break
        
        cv2.destroyAllWindows()
    else:
        # read image from folder
        images = glob.glob(f'{image_path}/*.png')
        imageCollection = []

        for image in images:
            imageCollection.append(cv2.imread(image))


    # performing calibration
    mtx, dist = calibrate_camera_arucoboard(imageCollection, 4, 3, 0.052, 0.005, cv2.aruco.DICT_5X5_250)
    
    print("Calibration completed")
    print(f"Mat:\n{mtx}\nDist:\n{dist}")

    # save image
    fp = open("./camera_matrix.pkl", "wb")
    pickle.dump(mtx, fp)
    fp1 = open("./dist_coef.pkl", "wb")
    pickle.dump(dist, fp1)


if __name__ == "__main__":
    main()