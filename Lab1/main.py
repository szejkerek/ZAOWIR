import json
import os
import numpy as np
import cv2 as cv
import logging
import time

debugMode = True
squareSizeInMilimeters = 28.67
rows = 7
columns = 10
end = 0 
alpha = 0.3
image_dir = os.path.join(os.getcwd(), 'Dataset', 'Chessboard', 'Mono 1', 'cam4')

objpoints = []
imgpoints = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

def CalculateError(mtx, dist, rvecs, tvecs):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    logging.info(f"Mean reprojection error: {mean_error:.3f}")
    return mean_error

def SaveToJson(ret, mtx, dist, rvecs, tvecs, mean_error):
    calibration_data = {
        "mean_reprojection_error": mean_error,
        "ret": ret,
        "mtx": mtx.tolist(),
        "dist": dist.tolist(),
        "rvecs": [rvec.tolist() for rvec in rvecs],
        "tvecs": [tvec.tolist() for tvec in tvecs]
    }

    with open('calibration_data.json', 'w') as json_file:
        json.dump(calibration_data, json_file, indent=4)

def UndistortPhoto(mtx, dist):
    img = cv.imread(os.path.join(image_dir, "58.png"))
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    undistorted_img = dst[y:y + h, x:x + w]
    return undistorted_img

def RemappImage(mtx, dist):
    img = cv.imread(os.path.join(image_dir, "58.png"))
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))

    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    
    # crop the image
    x, y, w, h = roi
    remappedImage = dst[y:y+h, x:x+w]
    return remappedImage

if __name__ == "__main__":
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * squareSizeInMilimeters

    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', 
                                    '.tiff', '.tif', '.webp', 
                                    '.gif', '.ppm', '.pgm', '.pbm'))]

    if end <= 0 or end > len(images):
        end = len(images)

    for count, fname in enumerate(images[:end], start=1):
        logging.info(f"Processing image {count}/{end}: {os.path.basename(fname)}")
        img = cv.imread(fname)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (columns, rows), None)

        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            if debugMode:
                cv.drawChessboardCorners(img, (columns, rows), corners2, True)
                cv.imshow('Corners', img)
                cv.waitKey(100)

    cv.destroyAllWindows()

    start_time = time.time()
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[:2][::-1], None, None)
    end_time = time.time()
    logging.info(f"Camera calibrated in {end_time - start_time:.2f} seconds with return value: {ret:.2f}")

    mean_error = CalculateError(mtx, dist, rvecs, tvecs)
    SaveToJson(ret, mtx, dist, rvecs, tvecs, mean_error)


    undistorted_img = UndistortPhoto(mtx, dist)
    remapped_image = RemappImage(mtx, dist)

    cv.imwrite('undistorted_image.png', undistorted_img)
    cv.imwrite('remapped_image.png', remapped_image)

