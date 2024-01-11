import cv2
import mediapipe as mp
import numpy as np

# Import necessary libraries for using Mediapipe Pose solution
mp_pose = mp.solutions.pose

# Display an opening screen to keep the console open for a certain period and indicate that the program is running
load = "loading"
for i in range(3):
    load = i * "-" + "loading" + i * "-"
    print(load)
    cv2.waitKey(200)

print("Program ready.")  # Notify the user that the program is ready

# Create a VideoCapture object to capture video from the specified camera index
camera_index = int(input("Enter the camera index: "))
cap = cv2.VideoCapture(camera_index)

# Use the Mediapipe Pose solution to detect landmarks
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Let the user define landmark indices
    landmark_indices_str = input("Enter landmark indices separated by commas (e.g., 12,11,24,23): ")
    landmark_indices = [int(index) for index in landmark_indices_str.split(',')]

    # Specify the path of the image to be placed and the landmarks to be considered
    imageSource = input("Enter the file path of the overlay image: ")
    image_texture = cv2.imread(imageSource, cv2.IMREAD_UNCHANGED)

    # Create a window for displaying the video feed in fullscreen mode
    cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the read frame to RGB format
        image.flags.writeable = False  # Make the image unwriteable
        results = pose.process(image)  # Process the image to detect landmarks
        image.flags.writeable = True  # Make the image writeable again
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert the image back to BGR format

        try:
            if results.pose_landmarks:
                # Calculate the x and y coordinates of the landmarks
                landmark_points = [(int(results.pose_landmarks.landmark[i].x * frame.shape[1]),
                                    int(results.pose_landmarks.landmark[i].y * frame.shape[0]))
                                    for i in landmark_indices]

                # Check if the landmarks are within the image boundaries
                all_within_bounds = all(0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0] for point in landmark_points)
                if all_within_bounds:
                    landmark_points = np.array(landmark_points, np.int32)

                    x, y, w, h = cv2.boundingRect(landmark_points)
                    image_texture_resized = cv2.resize(image_texture, (w, h))

                    # Overlay operation by checking the alpha channel
                    overlay_mask = image_texture_resized[:, :, 3] > 0
                    image[y:y + h, x:x + w][overlay_mask] = image_texture_resized[:, :, :3][overlay_mask]
        except:
            continue

        cv2.imshow('Mediapipe Feed', image)  # Display the results on the screen

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()  # Release the video capture
    cv2.destroyAllWindows()  # Close all windows
