import cv2
import mediapipe as mp
import numpy as np
import threading
import time
mp_pose = mp.solutions.pose

"""

Alper Melkeli: alpermelkeli@hacettepe.edu.tr

"""


#Exe dosyasında konsol açıldığında bir süre bekletiyor ondan dolayı açılma ekranı ekledim. Programın açıldığından emin olmak için.

load = "loading"

for i in range(3):
    
    load = i*"-" + "loading" + i*"-"
    
    print(load)

    time.sleep(0.2)

print("Program hazır.")

camera_index = int(input("Kamera indexini giriniz: "))
imageSource = input("Yerleştirilecek Resmin Dosya Yolunu Giriniz: ")

cap = cv2.VideoCapture(camera_index)

image_texture = cv2.imread(imageSource, cv2.IMREAD_UNCHANGED)
landmark_indices = [12, 11, 24, 23]

cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:

            if results.pose_landmarks:
                landmark_points = [(int(results.pose_landmarks.landmark[i].x * frame.shape[1]),
                                    int(results.pose_landmarks.landmark[i].y * frame.shape[0]))
                                for i in landmark_indices]

                all_within_bounds = all(0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0] for point in landmark_points)
                if all_within_bounds:
                    landmark_points = np.array(landmark_points, np.int32)
                    landmark_points[0][0] -= 30
                    landmark_points[1][0] += 30
                    x, y, w, h = cv2.boundingRect(landmark_points)
                    image_texture_resized = cv2.resize(image_texture, (w, h))

                    # Overlay the resized image_texture on the ROI with transparency using NumPy operations
                    overlay_mask = image_texture_resized[:, :, 3] > 0  # Check the alpha channel
                    image[y - 35:y + h - 35, x:x + w][overlay_mask] = image_texture_resized[:, :, :3][overlay_mask]
        except:
            continue
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
