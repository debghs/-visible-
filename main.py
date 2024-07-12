import cv2
import numpy as np
import time

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    return cap

def capture_roi_color(cap):
    print("Capturing ROI color in 3seconds")
    time.sleep(3)    
    # Define ROI parameters
    roi_x, roi_y, roi_w, roi_h = 200, 200, 100, 100  # You can change these values as needed
    
    start_time = time.time()
    color_samples = []
    
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        avg_color = np.mean(roi, axis=(0, 1))
        color_samples.append(avg_color)
        
        # Draw the ROI on the frame
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
        cv2.imshow('Capture ROI Color', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to quit early
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(color_samples) == 0:
        print("Error: No color samples captured.")
        return None
    
    avg_color = np.mean(color_samples, axis=0)
    print(f"Average ROI Color: {avg_color}")
    return avg_color

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

def replace_background(cap, background, trigger_color):
    start_time = time.time()
    
    lower_trigger = np.array(trigger_color) - 50  # Adjust threshold values as needed
    upper_trigger = np.array(trigger_color) + 50
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        img = np.flip(img, axis=1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define color ranges for detecting the trigger color
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask = create_mask(hsv, lower_red1, upper_red1, lower_red2, upper_red2)
        
        # Add trigger color detection
        mask_trigger = cv2.inRange(hsv, lower_trigger, upper_trigger)
        mask = mask | mask_trigger
        
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
    
    # Capture color for invisibility trigger
    trigger_color = capture_roi_color(cap)
    if trigger_color is None:
        cap.release()
        return
    
    # Convert trigger_color to HSV for comparison
    trigger_color_hsv = cv2.cvtColor(np.uint8([[trigger_color]]), cv2.COLOR_BGR2HSV)[0,0]
    lower_trigger_hsv = np.array([trigger_color_hsv[0]-10, 100, 100])
    upper_trigger_hsv = np.array([trigger_color_hsv[0]+10, 255, 255])

    cp = initialize_camera()
    if cp is None:
        return
    
    # Capture the background for replacement
    background = capture_background(cp)
    #print(background)
    if background is None:
        cp.release()
        return
    
    # Replace background with the captured one using the trigger color
    replace_background(cp, background, trigger_color_hsv)

if __name__ == "__main__":
    main()
