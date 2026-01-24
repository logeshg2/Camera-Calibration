#!/home/logesh/robotic_toolbox_ws/toolbox_env/bin/python3

import os
import cv2
import pickle
import numpy as np
import spatialmath as sm
from ComDependencies.robot_controller import robot


def save_calib_data(calib_data: object, calib_data_path: str):
    """Saves calibration data."""
    with open(calib_data_path, "wb") as f:
        pickle.dump(calib_data, f)


def load_calib_data(calib_data_path: str):
    """Loads calibration data."""
    with open(calib_data_path, "rb") as f:
        calib_data = pickle.load(f)
    return calib_data


def draw_axis(img, corners, imgpts, thickness=5):
    """Draws xyz axis."""
    # NOTE: opencv in BGR order, B-z, G-y, R-x
    corner = tuple(corners[0].astype(int).ravel())
    img = cv2.line(
        img,
        corner,
        tuple(imgpts[0].astype(int).ravel()),
        (0, 0, 255),
        thickness=thickness,
    )
    img = cv2.line(
        img,
        corner,
        tuple(imgpts[1].astype(int).ravel()),
        (0, 255, 0),
        thickness=thickness,
    )
    img = cv2.line(
        img,
        corner,
        tuple(imgpts[2].astype(int).ravel()),
        (255, 0, 0),
        thickness=thickness,
    )
    return img


def collect_checker_board_images(camera, save_dir):
    """Collects checkerboard images."""

    print("Type [c] to capture, [q] to exit.")
    array = []
    count = 1
    while True:
        _, frame = camera.read()
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("c"):
            cv2.imwrite(os.path.join(save_dir, f"image_{count}.png"), frame)
            array.append(frame)
            print(f"Saved data #{count}")
            count += 1

    camera.release()
    cv2.destroyAllWindows()
    return array


def calibrate_camera_arucoboard(images, cols, rows, marker_length, marker_separation, marker_dict=cv2.aruco.DICT_4X4_50, verbose=True):
    """Calibrates camera to get camera matrix and distortion coefficients, using aruco board."""

    aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict)
    board = cv2.aruco.GridBoard((rows, cols), marker_length, marker_separation, aruco_dict)
    arucoParams = cv2.aruco.DetectorParameters()

    corners_array = []
    ids_array = []
    counter = []
    imgShape = None

    for idx, image in enumerate(images):
        print(f"Calibrating image number: {idx+1}")
        # gray image conversion
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(grayImage, aruco_dict, parameters=arucoParams)

        if (ids is None):
            continue

        corners_array.extend(corners)
        ids_array.extend(ids.flatten())
        counter.append(len(ids))

        if imgShape is None:
            imgShape = grayImage.shape

    ids_array = np.array(ids_array, dtype=np.int32).reshape(-1, 1)
    counter = np.array(counter, dtype=np.int32)

    # calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(corners_array, ids_array, np.array(counter), board, imgShape, None, None)
    print(f"Reprojection Error: {ret}")
    
    return mtx, dist


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

### NOTE:
# calibration aruco -> DICT_6X6_100
# pick and place aruco -> DICT_4X4_50

def find_aruco_pose(frame, camera_matrix, dist_coeffs, marker_length, marker_dict=cv2.aruco.DICT_4X4_50):
    """Finds aruco marker pose."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict)          
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    (corners, ids, rejected) = detector.detectMarkers(gray)

    # pose of aruco
    marker_size = marker_length  # in mm
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    R_target2cam, t_target2cam = None, None
    if np.all(ids is not None):
        for i in range(len(ids)):
            ret, rvec, tvec = cv2.solvePnP(marker_points, corners[i], camera_matrix, dist_coeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)    
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec,30,3)
            R_target2cam, _ = cv2.Rodrigues(rvec)
            t_target2cam = tvec.squeeze()

    return R_target2cam, t_target2cam


def find_checkerboard_pose(frame, camera_matrix, dist_coeffs, cols, rows, square_size):
    """Finds checkerboard pose."""
    R_target2cam, t_target2cam = None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp = objp * square_size

    axis = np.float32(
        [[3 * square_size, 0, 0], [0, 3 * square_size, 0], [0, 0, -3 * square_size]]
    ).reshape(-1, 3)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if ret:
        # refine corners
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # find pose
        ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
        R_target2cam, _ = cv2.Rodrigues(rvec)
        t_target2cam = tvec.squeeze()

        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
        frame = draw_axis(frame, corners, imgpts)

    return R_target2cam, t_target2cam


def collect_eye_hand_data(
    camera,
    camera_matrix,
    dist_coeffs,
    marker_length,
    robot,
    aruco,
    cols,
    rows,
    square_size,
):
    """Collects eye hand calibration data."""
    target_poses = []
    robot_ee_poses = []
    count = 0
    while True:
        _, frame = camera.read()
        if aruco:
            R_target2cam, t_target2cam = find_aruco_pose(
                frame=frame,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                marker_length=marker_length,
            )
        else:
            R_target2cam, t_target2cam = find_checkerboard_pose(
                frame=frame,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                cols=cols,
                rows=rows,
                square_size=square_size,
            )
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("c"):
            if R_target2cam is None:
                print("None pose detected, try again")
            else:
                target_poses.append((R_target2cam, t_target2cam))

                robot.connect()
                xyzrpw = robot.get_curpos()
                robot_ee_poses.append(xyzrpw)
                count += 1
                print(f"Collected data: {count}")

    return target_poses, robot_ee_poses


def collect_eye_hand_data_fanuc(
    camera,
    camera_matrix,
    dist_coeffs,
    marker_length,
    aruco,
    cols,
    rows,
    square_size,
    image_save_dir
):  
    """Collects eye hand calibration data."""
    target_poses = []
    robot_ee_poses = []
    count = 0

    # robot model
    bot = robot('192.168.1.9')
    
    while True:
        _, frame = camera.read()
        if aruco:
            R_target2cam, t_target2cam = find_aruco_pose(
                frame=frame,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                marker_length=marker_length,
                marker_dict=cv2.aruco.DICT_6X6_100
            )
        else:
            R_target2cam, t_target2cam = find_checkerboard_pose(
                frame=frame,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                cols=cols,
                rows=rows,
                square_size=square_size,
            )
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("c"):
            if R_target2cam is None:
                print("None pose detected, try again")
            else:
                target_poses.append((R_target2cam, t_target2cam))

                # using fanuc robot_controller
                cartPose = bot.read_current_cartesian_pose()        # [X, Y, Z, W, P, R]
                trans = np.array(cartPose[0:3])
                rotm = sm.SO3.RPY(np.deg2rad(cartPose[3:6])).R
                ee_pose = [rotm, trans]                             # rotm, translation
                # debug
                # print(ee_pose)

                robot_ee_poses.append(ee_pose)
                count += 1
                print(f"Collected data: {count}")
                # save the image while performing hand eye calibration
                cv2.imwrite(f"{image_save_dir}/image{count}.png", frame)  
                
    return target_poses, robot_ee_poses


def calibrate_eye_hand(
    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True
):
    tts = []
    if eye_to_hand:
        # change coordinates from gripper2base to base2gripper
        R_base2gripper, t_base2gripper = [], []
        for R, t in zip(R_gripper2base, t_gripper2base):
            R_b2g = R.T
            t_b2g = -R_b2g @ t
            tts.append(t_b2g)
            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)

        # change parameters values
        R_gripper2base = R_base2gripper
        t_gripper2base = t_base2gripper

    # calibrate
    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS     # ANDREFF method give accurate values
    )

    return R, t

if __name__ == "__main__":
    R_g2b = load_calib_data("/home/logesh/fanuc_ws/src/Camera-Calibration/data/calib_data_finger/hand_eye_data/R_g2b.pkl")
    R_t2c = load_calib_data("/home/logesh/fanuc_ws/src/Camera-Calibration/data/calib_data_finger/hand_eye_data/R_t2c.pkl")
    t_g2b = load_calib_data("/home/logesh/fanuc_ws/src/Camera-Calibration/data/calib_data_finger/hand_eye_data/t_g2b.pkl")
    t_t2c = load_calib_data("/home/logesh/fanuc_ws/src/Camera-Calibration/data/calib_data_finger/hand_eye_data/t_t2c.pkl")

    R, t = calibrate_eye_hand(
        R_gripper2base=R_g2b,
        t_gripper2base=t_g2b,
        R_target2cam=R_t2c,
        t_target2cam=t_t2c,
        eye_to_hand=False
    )
    print("ANDREFF: ")
    print(R,t,sep='\n')
    # print(np.rad2deg(sm.base.tr2rpy(R)))

    # save_calib_data(R, "/home/logesh/fanuc_ws/src/Camera-Calibration/data/calib_data_finger/hand_eye_rotm.pkl")
    # save_calib_data(t, "/home/logesh/fanuc_ws/src/Camera-Calibration/data/calib_data_finger/hand_eye_trans.pkl")