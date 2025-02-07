import cv2 as cv
import numpy as np
import time


def sparse_optical_flow(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0, 255, (200, 3))

    ret, old_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    if p0 is None:
        print("Error: No good features to track.")
        return

    mask = np.zeros_like(old_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Compute optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is None or st is None:
            break

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        output = cv.add(frame, mask)
        cv.imshow('Sparse Optical Flow', output)

        key = cv.waitKey(30) & 0xFF
        if key == 27:  # ESC key to exit
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        # Refresh features if too many points are lost
        if len(p0) < 10:
            p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            if p0 is None:
                break

    cap.release()
    cv.destroyAllWindows()


def dense_optical_flow(video_path):
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    ret, frame1 = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        return

    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # Set full saturation for clear visualization

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break

        next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # Calculate dense optical flow
        flow = cv.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Map the angles to HSV color space
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # Convert HSV to BGR for display
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        # Display output with dynamic resizing
        cv.imshow('Dense Optical Flow', cv.resize(bgr, (800, 600)))

        key = cv.waitKey(30) & 0xFF
        if key == 27:  # ESC key to exit
            break

        prvs = next_frame  # Update previous frame

    cap.release()
    cv.destroyAllWindows()


def moving_objects_detection(video_path):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    ret, frame1 = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        return

    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break

        next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # Compute optical flow
        flow = cv.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalize motion magnitude
        motion_magnitude = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # Adaptive threshold for motion detection
        _, motion_mask = cv.threshold(motion_magnitude, 25, 255, cv.THRESH_BINARY)

        # Convert to uint8 for processing
        motion_mask = motion_mask.astype(np.uint8)

        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_OPEN, kernel)
        motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_CLOSE, kernel)

        # Find contours of moving objects
        contours, _ = cv.findContours(motion_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv.contourArea(cnt) > 800:  # Ignore small motions
                x, y, w, h = cv.boundingRect(cnt)
                cv.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame2, "Moving Object", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2, cv.LINE_AA)

        # Overlay motion mask (optional)
        blended = cv.addWeighted(frame2, 0.8, cv.cvtColor(motion_mask, cv.COLOR_GRAY2BGR), 0.2, 0)

        # Display result
        cv.imshow('Motion Detection', blended)

        key = cv.waitKey(30) & 0xFF
        if key == 27:  # ESC key to exit
            break

        prvs = next_frame  # Update previous frame

    cap.release()
    cv.destroyAllWindows()

def real_time_analysis(source=0):
    cap = cv.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    ret, frame1 = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        return

    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break

        next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalize magnitude for visualization
        motion_magnitude = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # Adaptive threshold for motion mask
        _, motion_mask = cv.threshold(motion_magnitude, 25, 255, cv.THRESH_BINARY)
        motion_mask = motion_mask.astype(np.uint8)

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_OPEN, kernel)
        motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_CLOSE, kernel)

        # Find contours of moving objects
        contours, _ = cv.findContours(motion_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv.contourArea(cnt) > 1000:  # Ignore small movements
                x, y, w, h = cv.boundingRect(cnt)

                # Compute average motion speed
                avg_speed = np.mean(mag[np.where(motion_mask[y:y+h, x:x+w] > 0)])

                # Compute average motion direction
                avg_angle = np.mean(ang[np.where(motion_mask[y:y+h, x:x+w] > 0)]) * (180 / np.pi)

                # Draw bounding box
                cv.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display motion information
                text_color = (0, 255, 0)
                cv.putText(frame2, f"Speed: {avg_speed:.2f} px/frame", (x, y - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                cv.putText(frame2, f"Dir: {avg_angle:.1f} deg", (x, y - 30), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Display processed frame
        cv.imshow('Real-Time Motion Analysis', frame2)

        # Exit on ESC key
        if cv.waitKey(30) & 0xFF == 27:
            break

        prvs = next_frame
        frame_count += 1

    elapsed_time = time.time() - start_time
    if frame_count > 0:
        print(f'Average processing time per frame: {elapsed_time / frame_count:.4f} seconds')

    cap.release()
    cv.destroyAllWindows()


videoPath = r"C:\#Projects\ZAOWIR\Lab5\car.mp4"

#sparse_optical_flow(videoPath)
#dense_optical_flow(videoPath)
#moving_objects_detection(videoPath)
real_time_analysis(0)
