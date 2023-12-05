from PIL import Image
import cv2

def emotion_test():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Add processing to the frame
        # processed_frame = # Your processing code here

        # Convert the processed frame to a PIL Image object
        image = Image.fromarray(frame)

        # Yield the image to the calling function
        yield image
