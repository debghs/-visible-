import cv2
import numpy as np
import time

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    return cap

def capture_background(cap):
    print("Capturing background. Please wait...")
    time.sleep(3)
    background = None
    for _ in range(30):
        _, background = cap.read()
    if background is None:
        print("Error: Could not capture background.")
    else:
        print("Background captured successfully.")
        return np.flip(background, axis=1)

def replace_background(cap, background):
    start_time = time.time()
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        img = np.flip(img, axis=1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Example for red cloak color
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask = create_mask(hsv, lower_red1, upper_red1, lower_red2, upper_red2)
        
        img = replace_pixels(img, mask, background)

        cv2.imshow('Display', img)
        
        k = cv2.waitKey(10)
        if k == 27:  # Press 'ESC' to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()

def create_mask(hsv, lower1, upper1, lower2, upper2):
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = mask1 + mask2
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask

def replace_pixels(img, mask, background):
    img[np.where(mask == 255)] = background[np.where(mask == 255)]
    return img

def main():
    cap = initialize_camera()
    if cap is None:
        return
    
    background = capture_background(cap)
    if background is None:
        cap.release()
        return
    
    replace_background(cap, background)

if __name__ == "__main__":
    main()
