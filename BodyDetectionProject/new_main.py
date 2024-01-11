import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Pose çözümünü kullanabilmek için gerekli kütüphanelerin import edilmesi
mp_pose = mp.solutions.pose

# Konsolun belirli bir süre açık kalmasını sağlamak ve programın çalıştığını belirtmek için açılış ekranı
load = "yükleniyor"
for i in range(3):
    load = i * "-" + "yükleniyor" + i * "-"
    print(load)
    cv2.waitKey(200)

print("Program hazır.")  # Programın hazır olduğunu kullanıcıya bildirir

# Kameradan görüntü almak için VideoCapture objesinin oluşturulması
camera_index = int(input("Kamera indexini giriniz: "))
cap = cv2.VideoCapture(camera_index)

# Mediapipe Pose çözümüyle landmarkları algılama işlemi
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Landmark indices'inin kullanıcı tarafından belirlenmesi
    landmark_indices_str = input("Landmark indices'ini virgülle ayırarak giriniz (örn: 12,11,24,23): ")
    landmark_indices = [int(index) for index in landmark_indices_str.split(',')]

    # Yerleştirilecek resmin yolu ve alınacak olan landmarkların belirlenmesi
    imageSource = input("Yerleştirilecek Resmin Dosya Yolunu Giriniz: ")
    image_texture = cv2.imread(imageSource, cv2.IMREAD_UNCHANGED)

    # Görüntünün tam ekran modunda gösterilmesi için pencerenin oluşturulması
    cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()  # Kameradan bir kare okunur
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Okunan kare RGB formatına dönüştürülür
        image.flags.writeable = False  # Görüntüyü değiştirilemez hale getirilir
        results = pose.process(image)  # Landmarkların tespiti için görüntü işlenir
        image.flags.writeable = True  # Görüntüyü tekrar değiştirilebilir hale getirilir
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Görüntü BGR formatına dönüştürülür

        try:
            if results.pose_landmarks:
                # Landmarkların x ve y koordinatlarının hesaplanması
                landmark_points = [(int(results.pose_landmarks.landmark[i].x * frame.shape[1]),
                                    int(results.pose_landmarks.landmark[i].y * frame.shape[0]))
                                    for i in landmark_indices]

                # Landmarkların görüntü sınırları içinde olup olmadığının kontrolü
                all_within_bounds = all(0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0] for point in landmark_points)
                if all_within_bounds:
                    landmark_points = np.array(landmark_points, np.int32)
                    landmark_points[0][0] -= 30
                    landmark_points[1][0] += 30
                    x, y, w, h = cv2.boundingRect(landmark_points)
                    image_texture_resized = cv2.resize(image_texture, (w, h))

                    # Alpha kanalını kontrol ederek overlay işlemi
                    overlay_mask = image_texture_resized[:, :, 3] > 0
                    image[y - 35:y + h - 35, x:x + w][overlay_mask] = image_texture_resized[:, :, :3][overlay_mask]
        except:
            continue

        cv2.imshow('Mediapipe Feed', image)  # Sonuçlar ekrana gösterilir

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()  # Video yakalama işlemi sonlandırılır
    cv2.destroyAllWindows()  # Tüm pencereler kapatılır
