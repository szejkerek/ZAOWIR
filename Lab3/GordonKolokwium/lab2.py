# Bartłomiej Gordon
if __name__ == "__main__":
    from zaowr_polsl_kisiel import stereo_calibration, stereo_rectify, are_params_valid #package co-author ;p

    left_cam = "C:\\Projekty\\ZaowrKolos\\kolokwium-zestaw\\left"
    right_cam = "C:\\Projekty\\ZaowrKolos\\kolokwium-zestaw\\right"

    left_cam_params = "C:\\Projekty\\ZaowrKolos\\leftCalib.json"
    right_cam_params = "C:\\Projekty\\ZaowrKolos\\rightCalib.json"
    stereo_cam_params = "C:\\Projekty\\ZaowrKolos\\stereoCalib.json"

    rectified_images_dir = "C:\\Projekty\\ZaowrKolos\\rect"

    left_valid, params_left = are_params_valid(left_cam_params)
    right_valid, params_right = are_params_valid(right_cam_params)
    stereo_valid, stereo_params = are_params_valid(stereo_cam_params)

    if not left_valid or not right_valid or not stereo_valid:
        stereo_calibration(
            chessBoardSize=(10, 7),
            squareRealDimensions=50.0,
            calibImgDirPath_left=left_cam,
            calibImgDirPath_right=right_cam,
            globImgExtension="png",
            saveCalibrationParams=True,
            calibrationParamsPath_left=left_cam_params,
            calibrationParamsPath_right=right_cam_params,
            saveStereoCalibrationParams=True,
            stereoCalibrationParamsPath=stereo_cam_params,
        )

        left_valid, params_left = are_params_valid(left_cam_params)
        right_valid, params_right = are_params_valid(right_cam_params)
        stereo_valid, stereo_params = are_params_valid(stereo_cam_params)

    stereo_rectify(
        calibImgDirPath_left=left_cam,
        calibImgDirPath_right=right_cam,
        imgPoints_left=params_left["imgPoints"],
        imgPoints_right=params_right["imgPoints"],
        loadStereoCalibrationParams=True,
        stereoCalibrationParamsPath=stereo_cam_params,
        saveRectifiedImages=True,
        rectifiedImagesDirPath=rectified_images_dir,
        whichImage=5,
        drawEpipolarLinesParams=(25, 2, 2)
    )


