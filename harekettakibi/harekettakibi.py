import cv2
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("video6.mp4")
# cap = cv2.VideoCapture(0)  # Canlı kamera için

if not cap.isOpened():
    raise IOError("Video veya kamera açılamadı!")

p_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("Video bitti veya kamera görüntüsü alınamadı.")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            if id == 13:  # Sol dirsek noktası
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
    p_time = c_time

    cv2.putText(img, f"FPS: {int(fps)}", (10, 65),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Pose Detection", img)

    # 'q' tuşuyla çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Kullanıcı çıkış yaptı.")
        break

cap.release()
cv2.destroyAllWindows()
