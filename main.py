import json
import os
import numpy as np
import cv2 as cv
import logging

rows=7
columns=10
end=30
objpoints = []
imgpoints = []

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_images(image_dir, extensions=('.png', '.jpg', '.bmp')):
    try:
        images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                  if f.lower().endswith(extensions)]
        logging.info(f"Loaded {len(images)} images from {image_dir}")
        return images
    except Exception as e:
        logging.error(f"Error loading images: {e}")
        return []

def find_corners(image, pattern_size):
    """Find chessboard corners in an image."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners2
    else:
        return None

def calibrate_camera(objpoints, imgpoints, image_shape):
    """Calibrate the camera and return the calibration parameters."""
    try:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_shape[::-1], None, None)
        logging.info(f"Camera calibrated with return value: {ret}")
        return ret, mtx, dist, rvecs, tvecs
    except Exception as e:
        logging.error(f"Calibration error: {e}")
        return None, None, None, None, None
    
def save_calibration_data(file_path, calibration_data):
    """Save calibration data to a JSON file."""
    try:
        with open(file_path, 'w') as json_file:
            json.dump(calibration_data, json_file, indent=4)
        logging.info(f"Calibration data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving calibration data: {e}")

def calibrate_chessboard(image_dir):
    """Calibrate the camera using chessboard images."""
    global criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * 28.67

    images = load_images(image_dir)

    for count, fname in enumerate(images[:end], start=1):
        logging.info(f"Processing image {count}/{end}: {fname}")
        img = cv.imread(fname)

        corners = find_corners(img, (columns, rows))
        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv.drawChessboardCorners(img, (columns, rows), corners, True)
            cv.imshow('Corners', img)
            cv.waitKey(100)

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, img.shape[:2])
    return ret, mtx, dist, rvecs, tvecs  # Return the lists as well

def undistort_image(image_path, mtx, dist):
    """Undistort an image using the camera matrix and distortion coefficients."""
    img = cv.imread(image_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.4, (w, h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    return dst[y:y + h, x:x + w]

def calculate_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    """Calculate the mean reprojection error."""
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    logging.info(f"Mean reprojection error: {mean_error}")
    return mean_error

if __name__ == "__main__":
    image_dir = os.path.join(os.getcwd(), 'Dataset', 'Chessboard', 'Mono 1', 'cam4')

    ret, mtx, dist, rvecs, tvecs = calibrate_chessboard(image_dir)

    if mtx is not None:
        mean_error = calculate_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)
        calibration_data = {
            "mean_reprojection_error": mean_error,
            "ret": ret,
            "mtx": mtx.tolist(),
            "dist": dist.tolist(),
            "rvecs": [rvec.tolist() for rvec in rvecs],
            "tvecs": [tvec.tolist() for tvec in tvecs]
        }
        save_calibration_data('calibration_data.json', calibration_data)

        # Test undistortion
        undistorted_image = undistort_image(os.path.join(image_dir, "58.png"), mtx, dist)
        cv.imshow('Undistorted Image', undistorted_image)
        cv.waitKey(0)
        cv.destroyAllWindows()