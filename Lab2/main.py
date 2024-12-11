if __name__ == "__main__":
    from zaowr_polsl_kisiel import are_params_valid, stereo_rectify, stereo_calibration

    left_camera_path = r"Dataset/Chessboard/Stereo 2/cam2/"
    right_camera_path = r"Dataset/Chessboard/Stereo 2/cam3/"

    params_path_left = "calibration_params/calibration_params_left.json"
    params_path_right = "calibration_params/calibration_params_right.json"
    params_path_stereo = "calibration_params/stereo_calibration_params.json"

    # Validate existing calibration files
    valid_left, left_params = are_params_valid(params_path_left)
    valid_right, right_params = are_params_valid(params_path_right)
    valid_stereo, stereo_params = are_params_valid(params_path_stereo)

    # Perform calibration if parameters are invalid or missing
    if not (valid_left and valid_right and valid_stereo):
        stereo_calibration(
            chessBoardSize=(10, 7),
            squareRealDimensions=50.0,
            calibImgDirPath_left=left_camera_path,
            calibImgDirPath_right=right_camera_path,
            globImgExtension="png",
            saveCalibrationParams=True,
            calibrationParamsPath_left=params_path_left,
            calibrationParamsPath_right=params_path_right,
            saveStereoCalibrationParams=True,
            stereoCalibrationParamsPath=params_path_stereo,
        )

        # Revalidate parameters after calibration
        valid_left, left_params = are_params_valid(params_path_left)
        valid_right, right_params = are_params_valid(params_path_right)
        valid_stereo, stereo_params = are_params_valid(params_path_stereo)

    # Ensure parameters are valid
    if not (valid_left and valid_right and valid_stereo):
        raise RuntimeError("Calibration unsuccessful. Parameters remain invalid.")

    rectified_output_dir = "./tests/rectified_images"

    stereo_rectify(
        calibImgDirPath_left=left_camera_path,
        calibImgDirPath_right=right_camera_path,
        imgPoints_left=left_params.get("imgPoints"),
        imgPoints_right=right_params.get("imgPoints"),
        loadStereoCalibrationParams=True,
        stereoCalibrationParamsPath=params_path_stereo,
        saveRectifiedImages=True,
        rectifiedImagesDirPath=rectified_output_dir,
        whichImage=5,
        drawEpipolarLinesParams=(20,3,2),
    )