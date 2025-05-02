import cv2
import os

# Create a directory to save images if it does not exist
output_folder = 'cap_chessboard'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize the camera
# NEW: changing resolution to 1280x720 for better quality of photos
camera = cv2.VideoCapture("/dev/video4")
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width in pixels
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Counter for the number of images captured
image_count = 0
max_images = 8

# Define the function to capture and save an image
def capture_image():
    global image_count
    ret, frame = camera.read()
    if ret:
        # Use .format() for string formatting
        image_path = os.path.join(output_folder, 'image_{}.jpg'.format(image_count + 1))
        cv2.imwrite(image_path, frame)
        print("Captured image {} and saved to {}".format(image_count + 1, image_path))
        image_count += 1
    else:
        print("Failed to capture image")

# Main loop
print("Press the spacebar to capture images. Capturing 5 images.")
while image_count < max_images:
    ret, frame = camera.read()
    if ret:
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space key
            capture_image()
        elif key == 27:  # Escape key to exit early
            break
    else:
        print("Failed to read frame from camera")

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
print("Captured {} images. Exiting.".format(image_count))