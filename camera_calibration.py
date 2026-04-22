#!/usr/bin/env python3

"""
Python script to calibrate rgb camera using Chessboard or checkerboard
"""

import cv2
import glob
import yaml
import argparse
import numpy as np


### helper function

def calibrate_camera_checkerboard(images, cols, rows, square_size, verbose=True):
    """Calibrates camera to get camera matrix and distortion coefficients."""

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp = objp * square_size

    # arrays to store object points and image points from all the images
    objpoints = []
    imgpoints = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            objpoints.append(objp)

            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            out_img = img.copy()
            if verbose:
                cv2.drawChessboardCorners(out_img, (cols, rows), corners, ret)
                cv2.imshow("Calibration", out_img)
                cv2.waitKey(0)

    cv2.destroyAllWindows()

    rmse, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return rmse, camera_matrix, dist_coeffs

###


argparser = argparse.ArgumentParser()
argparser.add_argument("--devID", type=int, default=2, help="Camera device Id.", required=False)
argparser.add_argument("--imgPath", type=str, default="./data/ImgData", help="Images path", required=False)
argparser.add_argument("--calibrateOnly", type=str, default="False", help="Calibration flag, else capture image.", required=False)
argparser.add_argument("--config_file", type=str, required=True, help="Configuration file path")
camID = argparser.parse_args().devID
calibFlag = True if argparser.parse_args().calibrateOnly == "True" else False
image_path = argparser.parse_args().imgPath
config_file = argparser.parse_args().config_file

# extract paramters
config = yaml.safe_load(open(config_file))
rows = config["no_rows"]
cols = config["no_cols"]
square_size = config["square_size"]
cam_width = config["cam_width"]
cam_height = config["cam_height"]


if (calibFlag):
    print("Performing calibration\n")
else:
    print("Starting image collection\n[s] -> save image frame\n[q] -> quit image collection\n")

def main():
    if (not calibFlag):
        cam = cv2.VideoCapture(camID)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

        if (not cam.isOpened()):
            print(f"Not able to open camera; device id: {camID}")
            exit(0)

        imageCollection = []
        count = 0

        # image collection loop
        while True:
            _, bgr = cam.read()
            bgr = cv2.resize(bgr, [int(cam_width), int(cam_height)])
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
            img = cv2.resize(cv2.imread(image), [int(cam_width), int(cam_height)])
            imageCollection.append(img)


    # performing calibration
    rmse, mtx, dist = calibrate_camera_checkerboard(imageCollection, cols, rows, square_size, False)
    
    print("Calibration completed")
    print(f"Mat:\n{mtx}\nDist:\n{dist}")

    # save image
    np.save("./camera_matrix.pkl", mtx)
    np.save("./dist_coef.pkl", dist)


if __name__ == "__main__":
    main()