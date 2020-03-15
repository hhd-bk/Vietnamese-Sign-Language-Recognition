import cv2
import numpy as np

# Matrix for performing dilate and erode
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Extract the skin color
def skin_extract(frame, lower, upper):
    # Convert frame to HSV coordinate
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Mask skin by the value in HSV coordinate
    mask = cv2.inRange(hsv_frame, lower, upper)
    # Take the skin color only
    skin_mask = cv2.bitwise_and(frame, frame, mask=mask)
    # Reduce noise outside the hand
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    # Reduce noise inside the hand
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    return skin_mask

# Take the HSV value from and save it by pressing s
def get_hsv_value():
    # Do nothing just pass to the next code
    def nothing(x):
        pass
    # Create HSV track bars
    cv2.namedWindow('Track bars')
    cv2.resizeWindow('Track bars', 800, 310)
    # Min values
    cv2.createTrackbar('H min', 'Track bars', 0, 179, nothing)
    cv2.createTrackbar('S min', 'Track bars', 0, 255, nothing)
    cv2.createTrackbar('V min', 'Track bars', 0, 255, nothing)
    # Max values (S and V max are always 255)
    cv2.createTrackbar('H max', 'Track bars', 0, 179, nothing)
    cv2.createTrackbar('S max', 'Track bars', 0, 255, nothing)
    cv2.createTrackbar('V max', 'Track bars', 0, 255, nothing)
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        # Get all min values
        h_min = cv2.getTrackbarPos('H min', 'Track bars')
        s_min = cv2.getTrackbarPos('S min', 'Track bars')
        v_min = cv2.getTrackbarPos('V min', 'Track bars')
        # Get all max values
        h_max = cv2.getTrackbarPos('H max', 'Track bars')
        s_max = cv2.getTrackbarPos('S max', 'Track bars')
        v_max = cv2.getTrackbarPos('V max', 'Track bars')
        # Extract skin by using skin color
        lower = np.array([h_min, s_min, v_min], np.uint8)
        upper = np.array([h_max, s_max, v_max], np.uint8)
        # Convert frame to hsv coordinate and feed to mask
        skin_data = skin_extract(frame, lower, upper)
        all_frame = np.hstack([frame, skin_data])
        all_frame = cv2.resize(all_frame, (800, 300))
        cv2.imshow('Track bars', all_frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            np.save('skin_value.npy', [lower, upper])
            print([lower, upper])
            break
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return

